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

# 确保与训练时完全一致的特征名称（不要修改）
original_features = [
    'CT-lesion involving ascending aorta',
    'NEU', 
    'Age',
    'CT-peritoneal effusion',
    'AST',
    'CREA',
    'Escape beat',
    'DBP',
    'CT-intramural hematoma'
]

# 带单位的显示名称映射
display_mapping = {
    'NEU': 'Neutrophil Count (10⁹/L)',
    'Age': 'Age (years)',
    'AST': 'AST (U/L)',
    'CREA': 'Creatinine (μmol/L)',
    'DBP': 'Diastolic BP (mmHg)',
    'CT-lesion involving ascending aorta': 'CT: Ascending Aorta Lesion',
    'CT-peritoneal effusion': 'CT: Peritoneal Effusion',
    'Escape beat': 'ECG: Escape Beat',
    'CT-intramural hematoma': 'CT: Intramural Hematoma'
}

# 国际标准正常范围
normal_ranges = {
    'NEU': (2.0, 7.5),    # 中性粒细胞
    'AST': (8, 40),       # 天门冬氨酸氨基转移酶
    'CREA': (64, 104),    # 肌酐（男性）
    'DBP': (60, 80)       # 舒张压
}

# ================= 页面布局 =================
st.set_page_config(layout="wide")

# 左侧边栏 - 输入部分
with st.sidebar:
    st.markdown("## Patient Parameters")
    with st.form("aortic_form"):
        # 连续变量
        neu = st.slider(display_mapping['NEU'], 0.0, 30.0, 5.0)
        age = st.slider(display_mapping['Age'], 18, 100, 60)
        ast = st.slider(display_mapping['AST'], 0, 500, 30)
        crea = st.slider(display_mapping['CREA'], 30, 1000, 80)
        dbp = st.slider(display_mapping['DBP'], 30, 150, 75)

        # 分类变量
        ct_lesion = st.selectbox(display_mapping['CT-lesion involving ascending aorta'], ['No', 'Yes'])
        ct_effusion = st.selectbox(display_mapping['CT-peritoneal effusion'], ['No', 'Yes'])
        escape_beat = st.selectbox(display_mapping['Escape beat'], ['No', 'Yes'])
        ct_hematoma = st.selectbox(display_mapping['CT-intramural hematoma'], ['No', 'Yes'])

        submitted = st.form_submit_button("Predict Mortality Risk")

# 右侧主区域
col1, col2 = st.columns([1, 3])

with col1:
    st.title('Aortic Dissection Mortality Prediction')
    st.markdown("""
    ## Multimodal Predictive Model
    **Integrating CT Radiomics and ECG Biomarkers**
    
    ### Model Performance:
    - **AUC**: 0.89 (95% CI: 0.84-0.94)
    - **Accuracy**: 88.05%  
    - **F1-score**: 0.65  
    - **Risk Threshold**: ≥0.202
    """)

with col2:
    if submitted:
        try:
            # 构建与训练时完全一致的数据结构
            data = {
                'CT-lesion involving ascending aorta': 1 if ct_lesion == 'Yes' else 0,
                'NEU': neu,
                'Age': age,
                'CT-peritoneal effusion': 1 if ct_effusion == 'Yes' else 0,
                'AST': ast,
                'CREA': crea,
                'Escape beat': 1 if escape_beat == 'Yes' else 0,
                'DBP': dbp,
                'CT-intramural hematoma': 1 if ct_hematoma == 'Yes' else 0
            }

            # 创建DataFrame（严格保持训练时的特征顺序）
            df = pd.DataFrame([data], columns=original_features)
            
            # 标准化处理
            scaled_data = scaler.transform(df)

            # 预测概率
            prob = model.predict_proba(scaled_data)[0][1]
            risk_level = "High Risk" if prob >= 0.202 else "Low Risk"

            # 显示结果
            st.markdown(f"## Prediction Result: **{risk_level}**")
            st.markdown(f"### 1-Year Mortality Probability: **{prob*100:.1f}%**")

            # 实验室异常建议
            st.markdown("### Clinical Recommendations")
            for feature in ['NEU', 'AST', 'CREA', 'DBP']:
                value = data[feature]
                min_val, max_val = normal_ranges[feature]
                
                if value < min_val:
                    st.markdown(f"""
                    <div style='background-color:#fff3cd; padding:10px; border-radius:5px; margin:10px 0;'>
                    ⚠️ **{display_mapping[feature]}**: {value}  
                    *Below normal range ({min_val}-{max_val})*  
                    Recommended actions:  
                    {{
                        'NEU': '• Infection screening\n• Bone marrow evaluation',
                        'AST': '• Repeat liver function tests\n• Viral hepatitis panel',
                        'CREA': '• Renal ultrasound\n• Urinalysis',
                        'DBP': '• Volume status assessment\n• Cardiac evaluation'
                    }}[feature]
                    </div>
                    """, unsafe_allow_html=True)
                    
                elif value > max_val:
                    st.markdown(f"""
                    <div style='background-color:#f8d7da; padding:10px; border-radius:5px; margin:10px 0;'>
                    ⚠️ **{display_mapping[feature]}**: {value}  
                    *Above normal range ({min_val}-{max_val})*  
                    Recommended actions:  
                    {{
                        'NEU': '• Infection control protocol\n• Consider sepsis workup',
                        'AST': '• Hepatology consultation\n• Abdominal ultrasound',
                        'CREA': '• Nephrology consultation\n• Stop nephrotoxic drugs',
                        'DBP': '• Antihypertensive therapy adjustment\n• End-organ damage evaluation'
                    }}[feature]
                    </div>
                    """, unsafe_allow_html=True)

            # 影像学警报
            st.markdown("### Critical Imaging Findings")
            if ct_lesion == 'Yes':
                st.error("""
                **Ascending Aorta Involvement**  
                Immediate Actions Required:  
                1. Emergency cardiothoracic surgery consult  
                2. Bedside echocardiography  
                3. Prepare operating room  
                """)
                
            if ct_hematoma == 'Yes':
                st.error("""
                **Intramural Hematoma**  
                Monitoring Protocol:  
                1. Serial CT angiography (24/48/72 hrs)  
                2. Target SBP <120 mmHg  
                3. Neurological assessment q4h  
                """)

        except Exception as e:
            st.error(f"Prediction Error: {str(e)}")

# 底部指南
st.markdown("---")
st.markdown("""
**Clinical Protocol**  
1. High-risk patients: Immediate ICU transfer  
2. Ascending aorta involvement: Surgical consult within 2 hours  
3. Creatinine >200 μmol/L: Initiate renal protection protocol  
4. All imaging abnormalities: Mandatory MDT review  
""")
