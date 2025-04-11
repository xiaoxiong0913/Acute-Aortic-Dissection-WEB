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

# 页面布局
st.set_page_config(layout="wide", page_icon="❤️")

# 正确的特征顺序
ordered_features = [
    'CT-lesion involving ascending aorta', 'NEU', 'Age', 'CT-peritoneal effusion', 'AST', 
    'CREA', 'Escape beat', 'DBP', 'CT-intramural hematoma'
]

# 定义连续特征和分类特征
continuous_features = ['NEU', 'Age', 'AST', 'CREA', 'DBP']
categorical_features = ['CT-lesion involving ascending aorta', 'CT-peritoneal effusion', 'Escape beat', 'CT-intramural hematoma']

# 输入面板
with st.sidebar:
    st.markdown("## Patient Parameters")
    with st.form("input_form"):
        # 动态生成输入选项（基于上传的特征列表）
        inputs = {}

        # 处理连续变量并为其加上单位
        for feature in continuous_features:
            if feature == 'Age':
                inputs[feature] = st.slider(f'{feature} (Years)', min_value=18, max_value=100, value=50)
            elif feature == 'NEU':
                inputs[feature] = st.slider(f'{feature} (10^9/L)', min_value=0.1, max_value=20.0, value=5.0)
            elif feature == 'AST':
                inputs[feature] = st.slider(f'{feature} (U/L)', min_value=0, max_value=500, value=30)
            elif feature == 'CREA':
                inputs[feature] = st.slider(f'{feature} (μmol/L)', min_value=30, max_value=200, value=80)
            elif feature == 'DBP':
                inputs[feature] = st.slider(f'{feature} (mmHg)', min_value=40, max_value=120, value=80)

        # 处理分类变量
        for feature in categorical_features:
            inputs[feature] = st.selectbox(feature, ['No', 'Yes'])

        # 提交按钮
        submitted = st.form_submit_button("Predict Risk")

# 结果面板
if submitted:
    col1, col2 = st.columns([2, 3])

    with col1:
        st.markdown("## 3-Year Mortality Prediction for Aortic Dissection")
        st.markdown(""" 
        This web-based calculator was developed based on the gradient boosting model, with an AUC of 0.89 and the following metrics:
        - Accuracy: 88.05%
        - F1-score: 0.65
        - Brier Score: 0.10
        """)

    with col2:
        try:
            # 构建输入数据（将 "Yes"/"No" 转换为 1/0）
            input_data = {}
            for feature in inputs:
                if inputs[feature] == 'Yes':
                    input_data[feature] = 1
                elif inputs[feature] == 'No':
                    input_data[feature] = 0
                else:
                    input_data[feature] = inputs[feature]

            # 创建严格排序的DataFrame
            df = pd.DataFrame([input_data], columns=ordered_features)

            # 标准化处理
            df_scaled = scaler.transform(df)

            # 预测概率
            prob = model.predict_proba(df_scaled)[:, 1][0]  # 获取类别为1的预测概率

            # 显示结果
            st.markdown(f"""
            ### Predicted Mortality Risk: {prob * 100:.2f}%
            """, unsafe_allow_html=True)

            # 风险评估并给出建议
            if prob >= 0.202:
                st.markdown("<span style='color:red'>High risk: This patient is classified as a high-risk patient.</span>", unsafe_allow_html=True)
                st.subheader("Personalized Recommendations:")
                
                # 给出个性化建议（仅在高风险情况下）
                st.markdown(f"**Age**: {inputs['Age']} (Normal range: 18-100)")
                st.markdown(f"**NEU**: {inputs['NEU']} (Normal range: 2.0 - 7.5 x10^9/L)")

                # 如果患者的数值低于正常范围，显示相应建议
                if inputs['Age'] < 40:
                    st.markdown(f"<span style='color:red'>Age: Value is below normal range (18-100 years). Consider further monitoring.</span>", unsafe_allow_html=True)
                if inputs['NEU'] < 2.0:
                    st.markdown(f"<span style='color:red'>NEU: Value is below normal range (2.0 - 7.5 x10^9/L). Consider additional testing.</span>", unsafe_allow_html=True)
                
            else:
                st.markdown("<span style='color:green'>Low risk: This patient is classified as a low-risk patient.</span>", unsafe_allow_html=True)

        except Exception as e:
            st.error(f"System Error: {str(e)}")

    # 临床路径指南
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

    # 底部版权信息
    st.markdown("---")
    st.markdown("<span style='color:grey;'>This application was developed by Yichang Central People's Hospital.</span>", unsafe_allow_html=True)
