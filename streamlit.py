import streamlit as st
import pandas as pd
import pickle
from sklearn.preprocessing import StandardScaler
import warnings
import subprocess
import sys

# 处理版本警告
warnings.filterwarnings("ignore", category=UserWarning, module="sklearn")

# 安装所需包函数
def install(package):
    result = subprocess.run([sys.executable, "-m", "pip", "install", package, "-i", "https://pypi.tuna.tsinghua.edu.cn/simple"], check=True)
    if result.returncode == 0:
        print(f"{package} successfully installed.")
    else:
        print(f"Failed to install {package}.")

# 列出所有需要安装的包
packages = ['pip']
for package in packages:
    install(package)

# 加载模型和标准化器
model_path = r"gbm_model.pkl"
scaler_path = r"scaler.pkl"

with open(model_path, 'rb') as model_file, open(scaler_path, 'rb') as scaler_file:
    model = pickle.load(model_file)
    scaler = pickle.load(scaler_file)

# 特征名称直接在代码中定义
feature_names = ['CT-lesion involving ascending aorta', 'NEU', 'Age', 'CT-peritoneal effusion', 'AST', 'CREA', 'Escape beat', 'DBP', 'CT-intramural hematoma']

# 创建Web应用的标题
st.title('Machine learning-based model predicts 1-year mortality in patients with Aortic Dissection')

# 添加介绍部分
st.markdown("""
## Introduction
This web-based calculator was developed based on the Gradient Boosting Model (GBM) for predicting 1-year mortality in patients with Aortic Dissection based on various clinical and radiological features.
""")

# 创建输入表单
st.markdown("## Selection Panel")
st.markdown("Please select the parameters to predict mortality risk.")

with st.form("prediction_form"):
    # 动态生成输入选项（基于上传的特征列表）
    inputs = {}

    # 处理连续变量并为其加上单位
    age = st.slider('Age', min_value=18, max_value=100, value=50)
    neu = st.slider('NEU (10^9/L)', min_value=0.1, max_value=20.0, value=5.0)
    ast = st.slider('AST (U/L)', min_value=0, max_value=500, value=30)
    crea = st.slider('CREA (μmol/L)', min_value=30, max_value=200, value=80)
    dbp = st.slider('DBP (mmHg)', min_value=40, max_value=120, value=80)
    
    # 处理分类变量（0=No, 1=Yes）
    ct_lesion = st.selectbox('CT-lesion involving ascending aorta', options=['No', 'Yes'])
    ct_peritoneal = st.selectbox('CT-peritoneal effusion', options=['No', 'Yes'])
    escape_beat = st.selectbox('Escape beat', options=['No', 'Yes'])
    ct_intramural = st.selectbox('CT-intramural hematoma', options=['No', 'Yes'])

    # 提交按钮
    submit_button = st.form_submit_button("Predict Risk")

# 当用户提交表单时
if submit_button:
    # 构建请求数据，将 "Yes"/"No" 转换为 1/0
    data = {
        'Age': age,
        'NEU': neu,
        'AST': ast,
        'CREA': crea,
        'DBP': dbp,
        'CT-lesion involving ascending aorta': 1 if ct_lesion == 'Yes' else 0,
        'CT-peritoneal effusion': 1 if ct_peritoneal == 'Yes' else 0,
        'Escape beat': 1 if escape_beat == 'Yes' else 0,
        'CT-intramural hematoma': 1 if ct_intramural == 'Yes' else 0
    }

    try:
        # 将数据转换为 DataFrame，并按特征文件中的顺序排列列
        data_df = pd.DataFrame([data], columns=feature_names)

        # 打印输入数据列名，便于调试
        st.write(f"Prediction data columns: {data_df.columns.tolist()}")

        # 应用标准化
        data_scaled = scaler.transform(data_df)

        # 进行预测
        prediction = model.predict_proba(data_scaled)[:, 1][0]  # 获取类别为1的预测概率

        # 显示预测结果
        st.write(f'Prediction: {prediction * 100:.2f}%')  # 将概率转换为百分比

        # 提供个性化建议
        if prediction >= 0.202:
            st.markdown(
                "<span style='color:red'>This patient has a high probability of death within one year.</span>",
                unsafe_allow_html=True)
            st.write("Personalized Recommendations:")
            st.write("Consider more intensive monitoring or treatment.")
        else:
            st.markdown(
                "<span style='color:green'>This patient has a high probability of survival within one year.</span>",
                unsafe_allow_html=True)

    except Exception as e:
        st.write(f'Error: {str(e)}')
