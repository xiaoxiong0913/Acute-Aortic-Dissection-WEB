import streamlit as st
import pandas as pd
import pickle
import os
from sklearn.preprocessing import StandardScaler

# 处理版本警告
import warnings

warnings.filterwarnings("ignore", category=UserWarning, module="sklearn")

# 加载模型、标准化器和特征列表
model_path = r"gbm_model.pkl"
scaler_path = r"scaler.pkl"
features_path = r"features.txt"

# 使用 pickle 加载模型和标准化器
try:
    with open(model_path, 'rb') as model_file:
        model = pickle.load(model_file)

    with open(scaler_path, 'rb') as scaler_file:
        scaler = pickle.load(scaler_file)

    # 检查 scaler 是否为 StandardScaler 实例
    if not isinstance(scaler, StandardScaler):
        raise ValueError("加载的 scaler 不是 StandardScaler 实例。")

    # 加载特征列表
    with open(features_path, 'r') as f:
        features = f.read().splitlines()

except Exception as e:
    st.error(f"加载模型、标准化器或特征时发生错误: {e}")
    st.stop()

import streamlit as st
import pandas as pd
import pickle
from sklearn.preprocessing import StandardScaler
import warnings

warnings.filterwarnings("ignore", category=UserWarning, module="sklearn")

# 页面配置
st.set_page_config(layout="wide", page_icon="❤️")
st.title("Aortic Dissection Mortality Prediction System")

# 自定义CSS样式
st.markdown("""
<style>
.protocol-card {
    padding: 15px;
    border-radius: 10px;
    margin: 10px;
    box-shadow: 0 2px 4px rgba(0,0,0,0.1);
}
.critical {
    border-left: 5px solid #dc3545;
    background-color: #fff5f5;
}
.alert {
    border-left: 5px solid #ffc107;
    background-color: #fff9e6;
}
.green-header {
    color: #28a745;
    font-size: 1.2em;
}
.red-header {
    color: #dc3545;
    font-size: 1.2em;
}
</style>
""", unsafe_allow_html=True)

# 简介部分
st.markdown("""
## Introduction
This clinical decision support tool integrates CT radiomics, electrocardiographic biomarkers, and laboratory parameters 
to predict 3-year mortality risk in aortic dissection patients. Validated with **AUC 0.89 (0.84-0.94)** and **88.09% accuracy**.
""")

# 横向排列的临床路径协议
cols = st.columns(3)
with cols[0]:
    st.markdown("""
    <div class='protocol-card critical'>
        <h4 class='red-header'>🚨 High Risk Criteria</h4>
        <ul style='padding-left:20px'>
            <li>Probability ≥20.2%</li>
            <li>Any aortic lesion/hematoma</li>
            <li>Requires ICU admission</li>
        </ul>
    </div>
    
    <div class='protocol-card'>
        <h4 class='green-header'>🔬 Laboratory Alerts</h4>
        <ul style='padding-left:20px'>
            <li>Creatinine >200 μmol/L → Renal consult</li>
            <li>AST >3×ULN → Hepatic workup</li>
        </ul>
    </div>
    """, unsafe_allow_html=True)

with cols[1]:
    st.markdown("""
    <div class='protocol-card alert'>
        <h4 class='red-header'>⚕️ Surgical Indications</h4>
        <ul style='padding-left:20px'>
            <li>Ascending aorta involvement → Emergency surgery</li>
            <li>Rapid hematoma expansion → Endovascular repair</li>
        </ul>
    </div>
    
    <div class='protocol-card'>
        <h4 class='green-header'>📋 Standard Protocol</h4>
        <ul style='padding-left:20px'>
            <li>CT follow-up q72h</li>
            <li>BP target: SBP <120 mmHg</li>
            <li>Neuro checks q4h</li>
        </ul>
    </div>
    """, unsafe_allow_html=True)

with cols[2]:
    st.markdown("""
    <div class='protocol-card'>
        <h4 class='green-header'>👁️ Monitoring Protocol</h4>
        <ul style='padding-left:20px'>
            <li>Hourly vital signs</li>
            <li>Neuro checks q4h</li>
            <li>Daily CT ×3 days</li>
        </ul>
    </div>
    """, unsafe_allow_html=True)

# 加载模型资源
try:
    model = pickle.load(open("gbm_model.pkl", "rb"))
    scaler = pickle.load(open("scaler.pkl", "rb"))
    ordered_features = [
        'CT-lesion involving ascending aorta', 'NEU', 'Age', 'CT-peritoneal effusion',
        'AST', 'CREA', 'Escape beat', 'DBP', 'CT-intramural hematoma'
    ]
except Exception as e:
    st.error(f"Initialization failed: {str(e)}")
    st.stop()

# 输入面板
with st.sidebar:
    st.markdown("## Patient Parameters")
    with st.form("input_form"):
        inputs = {}
        
        # 连续变量
        inputs['Age'] = st.slider("Age (Years)", 18, 100, 50)
        inputs['NEU'] = st.slider("NEU (10⁹/L)", 0.1, 25.0, 5.0)
        inputs['AST'] = st.slider("AST (U/L)", 0, 500, 30)
        inputs['CREA'] = st.slider("CREA (μmol/L)", 30, 200, 80)
        inputs['DBP'] = st.slider("DBP (mmHg)", 40, 120, 80)
        
        # 分类变量
        inputs['CT-lesion involving ascending aorta'] = st.selectbox("CT lesion involving ascending aorta", ["No", "Yes"])
        inputs['CT-peritoneal effusion'] = st.selectbox("CT peritoneal effusion", ["No", "Yes"])
        inputs['Escape beat'] = st.selectbox("Escape beat", ["No", "Yes"])
        inputs['CT-intramural hematoma'] = st.selectbox("CT intramural hematoma", ["No", "Yes"])
        
        submitted = st.form_submit_button("Predict Risk")

# 处理预测
if submitted:
    try:
        # 数据预处理
        input_data = {k: (1 if v == "Yes" else 0) if isinstance(v, str) else v for k, v in inputs.items()}
        df = pd.DataFrame([input_data], columns=ordered_features)
        df_scaled = scaler.transform(df)
        prob = model.predict_proba(df_scaled)[:, 1][0]
        risk_status = "High risk" if prob >= 0.202 else "Low risk"

        # 预测结果展示
        st.markdown(f"""
<div style='border-radius:10px; padding:20px; background-color:#f8f9fa; margin:20px 0;'>
    <h2 style='color:{"#dc3545" if risk_status == "High risk" else "#28a745"};'>
        Predicted Mortality Risk: {prob*100:.1f}% ({risk_status})
    </h2>
    <p>High risk of mortality within 3 years.</p>
    <h4>📊 Parameter Assessment</h4>
    <ul>
        <li>CREA (μmol/L): <span style='color:{"#dc3545" if input_data["CREA"] > 200 else "inherit"}'>
            {input_data['CREA']} {"⚠️" if input_data['CREA'] > 200 else ""}
        </span></li>
        <li>AST (U/L): <span style='color:{"#dc3545" if input_data["AST"] > 120 else "inherit"}'>
            {input_data['AST']} {"⚠️" if input_data['AST'] > 120 else ""}
        </span></li>
        <li>DBP (mmHg): {input_data['DBP']}</li>
    </ul>
    <h4>📝 Recommendations</h4>
    <div style='padding-left:20px'>
        <div style='color:#6c757d; margin:5px 0'>• Regular cardiovascular follow-up</div>
        <div style='color:#6c757d; margin:5px 0'>• Optimize antihypertensive therapy</div>
        {"<div style='color:#dc3545; margin:5px 0'>• Immediate surgical consultation</div>" if risk_status == "High risk" else ""}
    </div>
</div>
""", unsafe_allow_html=True)
        
    except Exception as e:
        st.error(f"Prediction error: {str(e)}")

# 页脚
st.markdown("---")
st.markdown("<div style='text-align: center; color: gray;'>Developed by Yichang Central People's Hospital</div>", 
            unsafe_allow_html=True)
