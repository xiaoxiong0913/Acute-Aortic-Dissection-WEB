import streamlit as st
import pandas as pd
import pickle
from sklearn.preprocessing import StandardScaler

# 处理版本警告
import warnings
warnings.filterwarnings("ignore", category=UserWarning, module="sklearn")

# 页面设置
st.set_page_config(layout="wide", page_icon="❤️")
st.title("Aortic Dissection Mortality Prediction System")

# 自定义CSS样式
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

# 介绍部分，修改为一年死亡率
st.write("# Introduction")
st.write("""
This clinical decision support tool integrates CT radiomics, electrocardiographic biomarkers, and laboratory parameters 
to predict 1-year mortality risk in aortic dissection patients. Validated with **AUC 0.89 (0.84-0.94)** and **88.05% accuracy**.
""")

# 参数输入部分
with st.sidebar:
    st.write("## Patient Parameters")
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

# 模拟模型预测
if submitted:
    try:
        # 数据预处理
        input_data = {k: 1 if v == "Yes" else 0 if isinstance(v, str) else v for k, v in inputs.items()}
        df = pd.DataFrame([input_data], columns=features)
        df_scaled = scaler.transform(df)
        prob = model.predict_proba(df_scaled)[:, 1][0]
        risk_status = "High Risk" if prob >= 0.202 else "Low Risk"
        color = "#dc3545" if risk_status == "High Risk" else "#28a745"

        # 使用 Streamlit 控件输出参数评估和建议
        st.markdown(f"""
        **Predicted Mortality Risk:** {prob*100:.1f}% ({risk_status})
        """, unsafe_allow_html=True)

        st.write("### Parameter Assessment:")
        st.write(f"**CREA (μmol/L):** {inputs['CREA']} - {'⚠️ High' if inputs['CREA'] > 200 else 'Normal'}")
        st.write(f"**AST (U/L):** {inputs['AST']} - {'⚠️ High' if inputs['AST'] > 120 else 'Normal'}")
        st.write(f"**DBP (mmHg):** {inputs['DBP']} - {'⚠️ High' if inputs['DBP'] > 100 else 'Normal'}")

        st.write("### Recommendations:")
        st.write("- Regular cardiovascular follow-up")
        st.write("- Optimize antihypertensive therapy")

        if risk_status == "High Risk":
            st.write("- **Immediate surgical consultation**")

    except Exception as e:
        st.error(f"Prediction error: {str(e)}")

# 个性化建议部分
st.markdown(
    "<span style='color:red'>This patient has a high probability of death within one year.</span>",
    unsafe_allow_html=True)
st.write("Personalized Recommendations:")

# 假设的正常范围
normal_ranges = {
    'NEU': (0.1, 25.0),
    'AST': (0, 120),
    'CREA': (30, 200),
    'DBP': (40, 120)
}

# 个性化建议
for feature, (normal_min, normal_max) in normal_ranges.items():
    value = inputs[feature]  # 获取每个特征的值
    if value < normal_min:
        st.write(f"**{feature}:** Your value is {value}. It is lower than the normal range ({normal_min} - {normal_max}). Consider increasing it towards {normal_min}.")
    elif value > normal_max:
        st.write(f"**{feature}:** Your value is {value}. It is higher than the normal range ({normal_min} - {normal_max}). Consider decreasing it towards {normal_max}.")
    else:
        st.write(f"**{feature}:** Your value is within the normal range ({normal_min} - {normal_max}).")

# Footer
st.write("---")
st.write("<div style='text-align: center; color: gray;'>Developed by Yichang Central People's Hospital</div>", 
         unsafe_allow_html=True)
