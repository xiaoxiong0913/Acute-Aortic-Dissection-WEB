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

# Page configuration
st.set_page_config(layout="wide", page_icon="❤️")
st.title("Aortic Dissection Mortality Prediction System")

# 固定显示的临床路径协议
st.markdown("""
## Clinical Pathway Protocol

**High Risk Criteria:**
- Probability ≥20.2%
- Any aortic lesion/hematoma
- Requires ICU admission

**Surgical Indications:**
- Ascending aorta involvement → Emergency surgery
- Rapid hematoma expansion → Endovascular repair

**Laboratory Alert Levels:**
- Creatinine >200 μmol/L → Renal consult
- AST >3×ULN → Hepatic workup

**Monitoring Protocol:**
- Hourly vital signs
- 4-hourly neurovascular checks
- Daily CT for first 72hrs

**Standard Management Protocol:**
- All patients: CT follow-up every 72 hours
- Blood pressure target: SBP <120 mmHg
- Priority neurovascular assessment q4h
""")

# Introduction
st.markdown("""
## Introduction
This clinical decision support tool integrates CT radiomics, electrocardiographic biomarkers, and laboratory parameters 
to predict 3-year mortality risk in aortic dissection patients. Validated with **AUC 0.89 (0.84-0.94)** and **88.09% accuracy**.
""")

# Load resources
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

# Input panel
with st.sidebar:
    st.markdown("## Patient Parameters")
    with st.form("input_form"):
        inputs = {}
        
        inputs['Age'] = st.slider("Age (Years)", 18, 100, 50)
        inputs['NEU'] = st.slider("NEU (10⁹/L)", 0.1, 25.0, 5.0)
        inputs['AST'] = st.slider("AST (U/L)", 0, 500, 30)
        inputs['CREA'] = st.slider("CREA (μmol/L)", 30, 200, 80)
        inputs['DBP'] = st.slider("DBP (mmHg)", 40, 120, 80)
        
        inputs['CT-lesion involving ascending aorta'] = st.selectbox("CT lesion involving ascending aorta", ["No", "Yes"])
        inputs['CT-peritoneal effusion'] = st.selectbox("CT peritoneal effusion", ["No", "Yes"])
        inputs['Escape beat'] = st.selectbox("Escape beat", ["No", "Yes"])
        inputs['CT-intramural hematoma'] = st.selectbox("CT intramural hematoma", ["No", "Yes"])
        
        submitted = st.form_submit_button("Predict Risk")

# Process prediction
if submitted:
    try:
        # Preprocess data
        input_data = {k: (1 if v == "Yes" else 0) if isinstance(v, str) else v for k, v in inputs.items()}
        df = pd.DataFrame([input_data], columns=ordered_features)
        df_scaled = scaler.transform(df)
        prob = model.predict_proba(df_scaled)[:, 1][0]
        risk_status = "High risk" if prob >= 0.202 else "Low risk"

        # 按指定格式显示预测结果
        st.markdown(f"""
        ## Predicted Mortality Risk: {prob*100:.1f}% ({risk_status})
        
        **Risk Summary:**  
        {risk_status} of mortality within 1 years.
        
        ### Parameter Assessment
        CREA (μmol/L): {'Normal' if 64 <= input_data['CREA'] <= 104 else 'Abnormal'}  
        AST (U/L): {'Normal' if 8 <= input_data['AST'] <= 40 else 'Abnormal'}  
        DBP (mmHg): {'Normal' if 60 <= input_data['DBP'] <= 80 else 'Abnormal'}
        
        ### Further Recommendations:
        1. Regular follow-up with cardiovascular specialist
        2. Consider CTA imaging for aortic monitoring
        3. Optimize antihypertensive therapy
        """)

        # 高风险额外建议
        if risk_status == "High risk":
            st.markdown("""
            ### Critical Care Recommendations:
            - Immediate cardiothoracic surgery consultation
            - Prepare ICU admission
            - Initiate beta-blocker therapy
            - Monitor for signs of rupture q1h
            """)
            
    except Exception as e:
        st.error(f"Prediction error: {str(e)}")

# Footer
st.markdown("---")
st.markdown("<div style='text-align: center; color: gray;'>Developed by Yichang Central People's Hospital</div>", 
            unsafe_allow_html=True)
