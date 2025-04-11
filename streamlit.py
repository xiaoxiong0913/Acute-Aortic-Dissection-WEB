import streamlit as st
import pandas as pd
import joblib
import warnings
import subprocess
import sys

# 配置警告信息
warnings.filterwarnings("ignore", category=UserWarning)

# 定义安装依赖函数
def install(package):
    subprocess.run([sys.executable, "-m", "pip", "install", package, "-i", "https://pypi.tuna.tsinghua.edu.cn/simple"])

# 安装必要依赖
required_packages = ['streamlit', 'pandas', 'scikit-learn', 'joblib']
for pkg in required_packages:
    install(pkg)

# 加载模型和标准化器
model_path = r"D:\WEB汇总\Acute Aortic Dissection WEB\gbm_model.pkl"
scaler_path = r"D:\WEB汇总\Acute Aortic Dissection WEB\scaler.pkl"

model = joblib.load(model_path)
scaler = joblib.load(scaler_path)

# 定义特征顺序（连续变量在前，分类在后）
features = [
    'NEU (10^9/L)',     # 中性粒细胞计数
    'Age (years)',      # 年龄
    'AST (U/L)',        # 天门冬氨酸氨基转移酶
    'CREA (μmol/L)',    # 肌酐
    'DBP (mmHg)',       # 舒张压
    'CT-lesion involving ascending aorta',  # CT累及升主动脉
    'CT-peritoneal effusion',   # CT腹膜积液
    'Escape beat',              # 逸搏
    'CT-intramural hematoma'    # CT壁内血肿
]

# 国际标准正常范围
normal_ranges = {
    'NEU (10^9/L)': (2.0, 7.5),    # 中性粒细胞计数
    'AST (U/L)': (0, 40),          # 肝酶标准
    'CREA (μmol/L)': (44, 115),    # 肌酐（综合男女范围）
    'DBP (mmHg)': (60, 80)         # 舒张压
}

# 创建网页界面
st.title('Interpretable Machine Learning for 1-Year Mortality Prediction in Acute Aortic Dissection')

# 介绍部分
st.markdown("""
## Clinical Decision Support System
**Multimodal Model Integrating CT Radiomics and Electrocardiographic Biomarkers**

Model Performance Metrics:
- **ROC AUC**: 0.89 (95% CI: 0.84-0.94)
- **AUC-PR**: 0.71
- **Accuracy**: 88.05%
- **F1-score**: 0.65
- **Brier Score**: 0.10

Threshold for High Risk: Probability ≥ 0.202
""")

# 创建输入面板
st.markdown("## Patient Parameters")
with st.form("aortic_form"):
    # 连续变量输入
    st.markdown("### Laboratory & Vital Signs")
    neu = st.slider('NEU (10^9/L)', 0.0, 30.0, 5.0)
    age = st.slider('Age (years)', 18, 100, 60)
    ast = st.slider('AST (U/L)', 0, 500, 30)
    crea = st.slider('CREA (μmol/L)', 30, 1000, 80)
    dbp = st.slider('DBP (mmHg)', 30, 150, 75)

    # 分类变量输入
    st.markdown("### Imaging & ECG Features")
    ct_lesion = st.selectbox('CT Lesion Involving Ascending Aorta', ['No', 'Yes'])
    ct_effusion = st.selectbox('CT Peritoneal Effusion', ['No', 'Yes'])
    escape_beat = st.selectbox('Escape Beat', ['No', 'Yes'])
    ct_hematoma = st.selectbox('CT Intramural Hematoma', ['No', 'Yes'])

    # 提交按钮
    submitted = st.form_submit_button("Predict Mortality Risk")

if submitted:
    # 转换分类变量
    data = {
        'NEU (10^9/L)': neu,
        'Age (years)': age,
        'AST (U/L)': ast,
        'CREA (μmol/L)': crea,
        'DBP (mmHg)': dbp,
        'CT-lesion involving ascending aorta': 1 if ct_lesion == 'Yes' else 0,
        'CT-peritoneal effusion': 1 if ct_effusion == 'Yes' else 0,
        'Escape beat': 1 if escape_beat == 'Yes' else 0,
        'CT-intramural hematoma': 1 if ct_hematoma == 'Yes' else 0
    }

    try:
        # 创建DataFrame并标准化
        df = pd.DataFrame([data], columns=features)
        scaled_data = scaler.transform(df)

        # 预测概率
        prob = model.predict_proba(scaled_data)[0][1]
        risk_level = "High Risk" if prob >= 0.202 else "Low Risk"

        # 显示结果
        st.markdown(f"## Prediction Result: **{risk_level}**")
        st.markdown(f"### Mortality Probability: **{prob:.2f}**")

        # 异常值建议
        st.markdown("### Clinical Recommendations")
        for feature in normal_ranges:
            value = data[feature]
            min_val, max_val = normal_ranges[feature]

            if value < min_val:
                st.markdown(f"""
                <div style='background-color:#fff3cd; padding:10px; border-radius:5px;'>
                ⚠️ **{feature}**: {value} (Below normal range {min_val}-{max_val})  
                Consider: Infection screening, Bone marrow evaluation
                </div>
                """, unsafe_allow_html=True)

            elif value > max_val:
                st.markdown(f"""
                <div style='background-color:#f8d7da; padding:10px; border-radius:5px;'>
                ⚠️ **{feature}**: {value} (Above normal range {min_val}-{max_val})  
                Consider: 
                {{
                    'NEU': 'Infection control',
                    'AST': 'Hepatic assessment',
                    'CREA': 'Renal consultation',
                    'DBP': 'Blood pressure management'
                }}[feature.split()[0]]
                </div>
                """, unsafe_allow_html=True)

        # 影像特征建议
        st.markdown("### Imaging Findings Alert")
        if ct_lesion == 'Yes':
            st.warning("Ascending aortic involvement requires urgent surgical evaluation")
        if ct_hematoma == 'Yes':
            st.warning("Intramural hematoma indicates high risk of progression")

    except Exception as e:
        st.error(f"Prediction Error: {str(e)}")

# 侧边栏信息
st.sidebar.markdown("""
**Clinical Guidance**  
1. 高风险患者建议收治ICU监护  
2. 升主动脉受累需在24小时内会诊心外科  
3. 肌酐>200 μmol/L需启动肾脏保护方案  
4. 所有异常影像学发现需MDT会诊
""")