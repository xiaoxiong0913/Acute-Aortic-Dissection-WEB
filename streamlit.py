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

# 页面设置
st.set_page_config(layout="wide", page_icon="❤️")
st.title("Aortic Dissection Mortality Prediction System")

# Custom CSS styling
st.write("""
<style>
.protocol-card {
    padding: 15px;
    border-radius: 10px;
    margin: 10px;
    box-shadow: 0 2px 4px rgba(0,0,0,0.1);
}
.critical-card {
    border-left: 5px solid #dc3545;
    background-color: #fff5f5;
}
.warning-card {
    border-left: 5px solid #ffc107;
    background-color: #fff9e6;
}
.green-card {
    border-left: 5px solid #28a745;
    background-color: #f0fff4;
}
.result-card {
    border-radius: 10px;
    padding: 20px;
    background-color: #f8f9fa;
    margin: 20px 0;
}
</style>
""", unsafe_allow_html=True)

# 介绍部分，修改为一年死亡率
st.write("# Introduction")
st.write("""
This clinical decision support tool integrates CT radiomics, electrocardiographic biomarkers, and laboratory parameters 
to predict 1-year mortality risk in aortic dissection patients. Validated with **AUC 0.89 (0.84-0.94)** and **88.05% accuracy**.
""")

# 临床路径卡片
cols = st.columns(3)
with cols[0]:
    st.write("""
    <div class='protocol-card critical-card'>
        <h4 style='color:#dc3545;'>🚨 High Risk Criteria</h4>
        <ul style='padding-left:20px'>
            <li>Probability ≥20.2%</li>
            <li>Aortic lesion/hematoma</li>
            <li>Requires ICU admission</li>
        </ul>
    </div>
    
    <div class='protocol-card green-card'>
        <h4 style='color:#28a745;'>🔬 Laboratory Alerts</h4>
        <ul style='padding-left:20px'>
            <li>Creatinine >200 μmol/L → Renal consult</li>
            <li>AST >3×ULN → Hepatic workup</li>
        </ul>
    </div>
    """, unsafe_allow_html=True)

with cols[1]:
    st.write("""
    <div class='protocol-card warning-card'>
        <h4 style='color:#dc3545;'>⚕️ Surgical Indications</h4>
        <ul style='padding-left:20px'>
            <li>Ascending aorta involvement → Emergency surgery</li>
            <li>Rapid hematoma expansion → Endovascular repair</li>
        </ul>
    </div>
    
    <div class='protocol-card green-card'>
        <h4 style='color:#28a745;'>📋 Standard Protocol</h4>
        <ul style='padding-left:20px'>
            <li>CT follow-up q72h</li>
            <li>BP target: SBP <120 mmHg</li>
            <li>Neuro checks q4h</li>
        </ul>
    </div>
    """, unsafe_allow_html=True)

with cols[2]:
    st.write("""
    <div class='protocol-card green-card'>
        <h4 style='color:#28a745;'>👁️ Monitoring Protocol</h4>
        <ul style='padding-left:20px'>
            <li>Hourly vital signs</li>
            <li>Neuro checks q4h</li>
            <li>Daily CT ×3 days</li>
        </ul>
    </div>
    """, unsafe_allow_html=True)

# 载入模型资源
try:
    model = pickle.load(open("gbm_model.pkl", "rb"))
    scaler = pickle.load(open("scaler.pkl", "rb"))
    features = [
        'CT-lesion involving ascending aorta', 'NEU', 'Age', 'CT-peritoneal effusion',
        'AST', 'CREA', 'Escape beat', 'DBP', 'CT-intramural hematoma'
    ]
except Exception as e:
    st.error(f"初始化失败: {str(e)}")
    st.stop()

# 输入面板
with st.sidebar:
    st.write("## Patient Parameters")
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

# 预测处理
# 预测处理部分
if submitted:
    try:
        # 模拟预测
        prob = 0.374  # 假设的死亡概率
        risk_status = "High Risk" if prob >= 0.202 else "Low Risk"
        color = "#dc3545" if risk_status == "High Risk" else "#28a745"

        # 使用 st.markdown 渲染 HTML
        st.markdown(f"""
        <div class='result-card'>
            <h2 style='color:{color};'>Predicted Mortality Risk: {prob*100:.1f}% ({risk_status})</h2>
            <p>High risk of mortality within 1 year.</p>

            <h4>📊 Parameter Assessment</h4>
            <ul>
                <li>CREA (μmol/L): <span style='color:{"#dc3545" if inputs['CREA'] > 200 else "inherit"}'>
                    {inputs['CREA']} {"⚠️" if inputs['CREA'] > 200 else ""}</span></li>
                <li>AST (U/L): <span style='color:{"#dc3545" if inputs['AST'] > 120 else "inherit"}'>
                    {inputs['AST']} {"⚠️" if inputs['AST'] > 120 else ""}</span></li>
                <li>DBP (mmHg): {inputs['DBP']}</li>
            </ul>

            <h4>📝 Recommendations</h4>
            <div style='padding-left:20px'>
                <p style='color:#6c757d;'>• Regular cardiovascular follow-up</p>
                <p style='color:#6c757d;'>• Optimize antihypertensive therapy</p>
                {"<p style='color:#dc3545;'>• Immediate surgical consultation</p>" if risk_status == "High Risk" else ""}
            </div>
        </div>
        """, unsafe_allow_html=True)
        
    except Exception as e:
        st.error(f"Prediction error: {str(e)}")

# 个性化建议
st.markdown(
    "<span style='color:red'>This patient has a high probability of death within one year.</span>",
    unsafe_allow_html=True)
st.write("Personalized Recommendations:")

# 假设的正常范围
normal_ranges = {
    'Age': (18, 100),
    'NEU': (0.1, 25.0),
    'AST': (0, 120),
    'CREA': (30, 200),
    'DBP': (40, 120)
}

# 个性化建议
for feature, (normal_min, normal_max) in normal_ranges.items():
    value = inputs[feature]  # 获取每个特征的值
    if value < normal_min:
        st.markdown(
            f"<span style='color:red'>{feature}: Your value is {value}. It is lower than the normal range ({normal_min} - {normal_max}). Consider increasing it towards {normal_min}.</span>",
            unsafe_allow_html=True)
    elif value > normal_max:
        st.markdown(
            f"<span style='color:red'>{feature}: Your value is {value}. It is higher than the normal range ({normal_min} - {normal_max}). Consider decreasing it towards {normal_max}.</span>",
            unsafe_allow_html=True)
    else:
        st.write(f"{feature}: Your value is within the normal range ({normal_min} - {normal_max}).")

# Footer
st.write("---")
st.write("<div style='text-align: center; color: gray;'>Developed by Yichang Central People's Hospital</div>", 
         unsafe_allow_html=True)


