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
model_path = r"gbm_model.pkl"
scaler_path = r"scaler.pkl"

model = joblib.load(model_path)
scaler = joblib.load(scaler_path)

# 原始特征名称（必须与训练时完全一致）
original_features = [
    'NEU',  # 中性粒细胞计数
    'Age',
    'AST',
    'CREA',
    'DBP',
    'CT-lesion involving ascending aorta',
    'CT-peritoneal effusion',
    'Escape beat',
    'CT-intramural hematoma'
]

# 带单位的显示名称
display_names = {
    'NEU': 'Neutrophil Count (10⁹/L)',
    'Age': 'Age (years)',
    'AST': 'Aspartate Aminotransferase (U/L)',
    'CREA': 'Creatinine (μmol/L)',
    'DBP': 'Diastolic Blood Pressure (mmHg)',
    'CT-lesion involving ascending aorta': 'CT: Ascending Aorta Lesion',
    'CT-peritoneal effusion': 'CT: Peritoneal Effusion',
    'Escape beat': 'ECG: Escape Beat',
    'CT-intramural hematoma': 'CT: Intramural Hematoma'
}

# 国际标准正常范围
normal_ranges = {
    'NEU': (2.0, 7.5),
    'AST': (0, 40),
    'CREA': (44, 115),
    'DBP': (60, 80)
}

# ================= 页面布局 =================
st.set_page_config(layout="wide")

# 左侧边栏 - 输入部分
with st.sidebar:
    st.markdown("## Patient Parameters")
    with st.form("aortic_form"):
        # 连续变量输入
        neu = st.slider(display_names['NEU'], 0.0, 30.0, 5.0)
        age = st.slider(display_names['Age'], 18, 100, 60)
        ast = st.slider(display_names['AST'], 0, 500, 30)
        crea = st.slider(display_names['CREA'], 30, 1000, 80)
        dbp = st.slider(display_names['DBP'], 30, 150, 75)

        # 分类变量输入
        ct_lesion = st.selectbox(display_names['CT-lesion involving ascending aorta'], ['No', 'Yes'])
        ct_effusion = st.selectbox(display_names['CT-peritoneal effusion'], ['No', 'Yes'])
        escape_beat = st.selectbox(display_names['Escape beat'], ['No', 'Yes'])
        ct_hematoma = st.selectbox(display_names['CT-intramural hematoma'], ['No', 'Yes'])

        submitted = st.form_submit_button("Predict Mortality Risk")

# 右侧主区域
col1, col2 = st.columns([1, 3])

with col1:
    # 标题和介绍
    st.title('Aortic Dissection Mortality Predictor')
    st.markdown("""
    ## Multimodal Predictive Model
    **Integrating CT Radiomics and ECG Biomarkers**
    
    Model Performance:
    - **AUC**: 0.89 (0.84-0.94)
    - **Accuracy**: 88.05%
    - **F1-score**: 0.65
    - **Risk Threshold**: ≥0.202
    """)

with col2:
    if submitted:
        try:
            # 数据转换
            data = {
                'NEU': neu,
                'Age': age,
                'AST': ast,
                'CREA': crea,
                'DBP': dbp,
                'CT-lesion involving ascending aorta': 1 if ct_lesion == 'Yes' else 0,
                'CT-peritoneal effusion': 1 if ct_effusion == 'Yes' else 0,
                'Escape beat': 1 if escape_beat == 'Yes' else 0,
                'CT-intramural hematoma': 1 if ct_hematoma == 'Yes' else 0
            }

            # 创建DataFrame并标准化
            df = pd.DataFrame([data], columns=original_features)
            scaled_data = scaler.transform(df)

            # 预测概率
            prob = model.predict_proba(scaled_data)[0][1]
            risk_level = "High Risk" if prob >= 0.202 else "Low Risk"

            # 显示结果
            st.markdown(f"## Prediction Result: **{risk_level}**")
            st.markdown(f"### 1-Year Mortality Probability: **{prob*100:.1f}%**")

            # 异常值建议
            st.markdown("### Clinical Recommendations")
            for feature in normal_ranges:
                value = data[feature]
                min_val, max_val = normal_ranges[feature]
                display_name = display_names[feature]

                if value < min_val:
                    st.markdown(f"""
                    <div style='background-color:#fff3cd; padding:10px; border-radius:5px; margin:10px 0;'>
                    ⚠️ **{display_name}**: {value} (Below normal range {min_val}-{max_val})  
                    Recommended Actions:  
                    • Perform infection screening  
                    • Evaluate bone marrow function
                    </div>
                    """, unsafe_allow_html=True)

                elif value > max_val:
                    st.markdown(f"""
                    <div style='background-color:#f8d7da; padding:10px; border-radius:5px; margin:10px 0;'>
                    ⚠️ **{display_name}**: {value} (Above normal range {min_val}-{max_val})  
                    Recommended Actions:  
                    • {{
                        'NEU': 'Initiate infection control protocol',
                        'AST': 'Conduct hepatic function assessment',
                        'CREA': 'Consult nephrology specialist',
                        'DBP': 'Optimize antihypertensive therapy'
                    }}[feature]
                    </div>
                    """, unsafe_allow_html=True)

            # 影像特征建议
            st.markdown("### Critical Imaging Findings")
            if ct_lesion == 'Yes':
                st.error("""
                **Ascending Aortic Involvement**  
                Immediate Actions:  
                1. Contact cardiothoracic surgery  
                2. Prepare for possible emergency intervention  
                3. Monitor for pericardial effusion
                """)
                
            if ct_hematoma == 'Yes':
                st.error("""
                **Intramural Hematoma Detected**  
                Required Measures:  
                1. Serial CT monitoring (24/48/72hrs)  
                2. Strict blood pressure control  
                3. Assess for end-organ malperfusion
                """)

        except Exception as e:
            st.error(f"Prediction Error: {str(e)}")

# 底部临床指南
st.markdown("---")
st.markdown("""
**Clinical Protocol Guidance**  
• High-risk patients require ICU admission  
• Surgical consultation within 2hrs for ascending aorta involvement  
• Renal protection protocol for creatinine >200 μmol/L  
• MDT consultation for all positive imaging findings
""")
