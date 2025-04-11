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

# Page configuration
st.set_page_config(layout="wide", page_icon="❤️")
st.title("Aortic Dissection Mortality Prediction System")

# Introduction section with uniform font size
st.markdown("""
<style>
.intro-text {
    font-size: 16px;
}
</style>

<div class='intro-text'>
## Introduction
This clinical decision support tool integrates CT radiomics, electrocardiographic biomarkers, and laboratory parameters 
to predict 3-year mortality risk in aortic dissection patients. Validated with <b>AUC 0.89 (0.84-0.94)</b> and <b>88.09% accuracy</b>.
</div>
""", unsafe_allow_html=True)

# Load model resources
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
        
        # Continuous variables
        inputs['Age'] = st.slider("Age (Years)", 18, 100, 50)
        inputs['NEU'] = st.slider("NEU (10⁹/L)", 0.1, 25.0, 5.0)
        inputs['AST'] = st.slider("AST (U/L)", 0, 500, 30)
        inputs['CREA'] = st.slider("CREA (μmol/L)", 30, 200, 80)
        inputs['DBP'] = st.slider("DBP (mmHg)", 40, 120, 80)
        
        # Categorical variables
        inputs['CT-lesion involving ascending aorta'] = st.selectbox("CT lesion involving ascending aorta", ["No", "Yes"])
        inputs['CT-peritoneal effusion'] = st.selectbox("CT peritoneal effusion", ["No", "Yes"])
        inputs['Escape beat'] = st.selectbox("Escape beat", ["No", "Yes"])
        inputs['CT-intramural hematoma'] = st.selectbox("CT intramural hematoma", ["No", "Yes"])
        
        submitted = st.form_submit_button("Predict Risk")

# Result display
if submitted:
    try:
        # Data preprocessing
        input_data = {k: (1 if v == "Yes" else 0) if isinstance(v, str) else v for k, v in inputs.items()}
        df = pd.DataFrame([input_data], columns=ordered_features)
        df_scaled = scaler.transform(df)
        prob = model.predict_proba(df_scaled)[:, 1][0]

        # Main result display
        st.markdown("## Prediction Result")
        risk_status = "High Risk" if prob >= 0.202 else "Low Risk"
        color = "#dc3545" if risk_status == "High Risk" else "#28a745"
        st.markdown(f"<h2 style='color:{color}'>3-Year Mortality Probability: <b>{prob*100:.1f}%</b></h2>", 
                    unsafe_allow_html=True)
        
        # High-risk recommendations
        if risk_status == "High Risk":
            st.markdown("""
            <div style='border-left: 5px solid #dc3545; padding: 10px; margin: 15px 0;'>
            <h4 style='color:#dc3545'>🚨 High Risk Management Protocol</h4>
            """, unsafe_allow_html=True)
            
            # Laboratory alerts
            lab_alerts = []
            if input_data['CREA'] > 200:
                lab_alerts.append("⚠️ **Creatinine >200 μmol/L** → Immediate nephrology consult")
            if input_data['AST'] > 120:
                lab_alerts.append("⚠️ **AST >120 U/L** → Initiate hepatic protection protocol")
                
            if lab_alerts:
                st.markdown("""
                <div style='background-color:#f8d7da; padding:10px; border-radius:5px; margin:10px 0;'>
                <h5>CRITICAL LAB VALUES</h5>
                """ + "<br>".join(lab_alerts) + "</div>", unsafe_allow_html=True)
            
            # Imaging alerts
            if input_data['CT-lesion involving ascending aorta']:
                st.markdown("""
                <div style='background-color:#dc3545; color:white; padding:10px; border-radius:5px; margin:10px 0;'>
                <h5>🚨 ASCENDING AORTA INVOLVEMENT</h5>
                1. Activate cardiothoracic surgery team<br>
                2. Prepare emergency OR<br>
                3. Hourly vital sign monitoring
                </div>
                """, unsafe_allow_html=True)
                
            st.markdown("</div>", unsafe_allow_html=True)

        # Clinical Pathway Protocol
        st.markdown("---")
        st.markdown("""  
        **Clinical Pathway Protocol**  
        1. **High Risk Criteria**:  
           - Probability ≥20.2%  
           - Any aortic lesion/hematoma  
           - Requires ICU admission  

        2. **Surgical Indications**:  
           - Ascending aorta involvement → Emergency surgery  
           - Rapid hematoma expansion → Endovascular repair  

        3. **Laboratory Alert Levels**:  
           - Creatinine >200 μmol/L → Renal consult  
           - AST >3×ULN → Hepatic workup  

        4. **Monitoring Protocol**:  
           - Hourly vital signs  
           - 4-hourly neurovascular checks  
           - Daily CT for first 72hrs  
        """)

        # Standard recommendations
        st.markdown("""
        **Standard Management Protocol**
        - All patients: CT follow-up every 72 hours
        - Blood pressure target: SBP <120 mmHg
        - Priority neurovascular assessment q4h
        """)
        
    except Exception as e:
        st.error(f"System error: {str(e)}")

# Footer
st.markdown("---")
st.markdown("<div style='text-align: center; color: gray;'>Developed by Yichang Central People's Hospital</div>", 
            unsafe_allow_html=True)
