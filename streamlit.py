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

# 严格匹配训练时的特征名称和顺序（必须与训练数据完全一致）
original_features = [
    'CT-lesion involving ascending aorta',  # 必须与训练数据列名完全一致
    'NEU',
    'Age',
    'CT-peritoneal effusion',
    'AST',
    'CREA',
    'Escape beat',  # 注意这里是下划线还是空格
    'DBP',
    'CT-intramural hematoma'
]

# 带单位的显示名称映射（仅用于界面显示）
display_mapping = {
    'CT-lesion involving ascending aorta': 'CT: Ascending Aorta Lesion',
    'NEU': 'Neutrophil (10⁹/L)',
    'Age': 'Age (years)',
    'CT-peritoneal effusion': 'CT: Peritoneal Effusion',
    'AST': 'AST (U/L)',
    'CREA': 'Creatinine (μmol/L)',
    'Escape beat': 'ECG: Escape Beat',
    'DBP': 'Diastolic BP (mmHg)',
    'CT-intramural hematoma': 'CT: Intramural Hematoma'
}

# ================= 专业医学布局 =================
st.set_page_config(layout="wide", page_icon="❤️")

# 左侧输入面板
with st.sidebar:
    st.markdown("## Patient Parameters")
    with st.form("input_form"):
        # 分类特征（保持原始名称）
        ct_lesion = st.selectbox(display_mapping['CT-lesion involving ascending aorta'], ['No', 'Yes'])
        ct_effusion = st.selectbox(display_mapping['CT-peritoneal effusion'], ['No', 'Yes'])
        escape_beat = st.selectbox(display_mapping['Escape beat'], ['No', 'Yes'])
        ct_hematoma = st.selectbox(display_mapping['CT-intramural hematoma'], ['No', 'Yes'])
        
        # 连续特征（带单位显示）
        neu = st.slider(display_mapping['NEU'], 0.0, 30.0, 5.0)
        age = st.slider(display_mapping['Age'], 18, 100, 60)
        ast = st.slider(display_mapping['AST'], 0, 500, 30)
        crea = st.slider(display_mapping['CREA'], 30, 1000, 80)
        dbp = st.slider(display_mapping['DBP'], 30, 150, 75)

        submitted = st.form_submit_button("Predict Risk")

# 右侧结果面板
col1, col2 = st.columns([1, 2])

with col1:
    st.markdown("## Aortic Dissection Mortality Predictor")
    st.markdown("""
    **Multimodal Model Integrating:**
    - CT Radiomics Features
    - Electrocardiographic Biomarkers
    - Clinical Laboratory Data
    
    **Validation Metrics:**
    - AUC: 0.89 (0.84-0.94)
    - Accuracy: 88.05%
    - F1-score: 0.65
    - Brier Score: 0.10
    """)

with col2:
    if submitted:
        try:
            # 构建与训练数据完全一致的结构
            input_data = {
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
            
            # 创建严格排序的DataFrame
            df = pd.DataFrame([input_data], columns=original_features)
            
            # 标准化处理
            scaled_data = scaler.transform(df)
            
            # 预测概率
            prob = model.predict_proba(scaled_data)[0][1]
            risk_status = "High Risk" if prob >= 0.202 else "Low Risk"
            
            # 显示结果
            st.markdown(f"""
            ### Prediction Result: <span style='color:red'>{risk_status}</span>
            ##### 1-Year Mortality Probability: {prob*100:.1f}%
            """, unsafe_allow_html=True)
            
            # 医学建议系统
            st.markdown("### Clinical Decision Support")
            
            # 实验室异常检测
            lab_ranges = {
                'NEU': (2.0, 7.5),
                'AST': (8, 40),
                'CREA': (64, 104),
                'DBP': (60, 80)
            }
            
            for param in lab_ranges:
                value = input_data[param]
                low, high = lab_ranges[param]
                if value < low:
                    st.markdown(f"""
                    <div style='background-color:#fff3cd; padding:10px; border-radius:5px; margin:10px 0;'>
                    ⚠️ **{display_mapping[param]}**: {value} (Low)  
                    Recommended: {{
                        'NEU': 'Infection screening',
                        'AST': 'Repeat LFTs',
                        'CREA': 'Renal ultrasound',
                        'DBP': 'Volume assessment'
                    }}[param]
                    </div>
                    """, unsafe_allow_html=True)
                elif value > high:
                    st.markdown(f"""
                    <div style='background-color:#f8d7da; padding:10px; border-radius:5px; margin:10px 0;'>
                    ⚠️ **{display_mapping[param]}**: {value} (High)  
                    Required: {{
                        'NEU': 'Sepsis protocol',
                        'AST': 'Hepatology consult',
                        'CREA': 'Nephrology consult',
                        'DBP': 'BP management'
                    }}[param]
                    </div>
                    """, unsafe_allow_html=True)
            
            # 影像学危急值处理
            if ct_lesion == 'Yes':
                st.markdown("""
                <div style='background-color:#dc3545; color:white; padding:10px; border-radius:5px; margin:10px 0;'>
                🚨 **Ascending Aorta Involvement**  
                Immediate Actions:  
                1. Call cardiothoracic surgery  
                2. Prepare OR  
                3. Monitor for rupture signs  
                </div>
                """, unsafe_allow_html=True)
                
            if ct_hematoma == 'Yes':
                st.markdown("""
                <div style='background-color:#dc3545; color:white; padding:10px; border-radius:5px; margin:10px 0;'>
                🚨 **Intramural Hematoma**  
                Priority Measures:  
                1. Serial CT monitoring  
                2. Strict BP control (SBP <120 mmHg)  
                3. Assess organ perfusion  
                </div>
                """, unsafe_allow_html=True)
                
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
