import streamlit as st
import pandas as pd
import pickle
from sklearn.preprocessing import StandardScaler
import warnings

warnings.filterwarnings("ignore", category=UserWarning, module="sklearn")

# Load model, scaler, and feature list
model_path = r"gbm_model.pkl"
scaler_path = r"scaler.pkl"
features_path = r"features.txt"

# Load model and scaler using pickle
try:
    with open(model_path, 'rb') as model_file:
        model = pickle.load(model_file)

    with open(scaler_path, 'rb') as scaler_file:
        scaler = pickle.load(scaler_file)

    # Check if scaler is an instance of StandardScaler
    if not isinstance(scaler, StandardScaler):
        raise ValueError("Loaded scaler is not an instance of StandardScaler.")

    # Load feature list
    with open(features_path, 'r') as f:
        features = f.read().splitlines()

except Exception as e:
    st.error(f"Error loading model, scaler, or features: {e}")
    st.stop()

st.set_page_config(layout="wide", page_icon="❤️")
st.title("Acute Aortic Dissection Mortality Prediction System")

# Custom CSS styling for beautification
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

# Introduction section
st.write("# Introduction")
st.write("""
This clinical decision support tool integrates CT radiomics, electrocardiographic biomarkers, and laboratory parameters 
to predict 1-year mortality risk in aortic dissection patients. Validated with **AUC 0.89 (0.84-0.94)** and **88.05% accuracy**.
""")

# Clinical pathway cards
cols = st.columns(3)

# First column: High Risk Criteria
with cols[0]:
    st.write("""
    <div class='protocol-card critical-card'>
        <h4 style='color:#dc3545;'>High Risk Criteria</h4>
        <ul style='padding-left:20px'>
            <li>Probability ≥20.2%</li>
        </ul>
    </div>
    """, unsafe_allow_html=True)

# Second column: Laboratory Alerts and Surgical Indications
with cols[1]:
    st.write("""
    <div class='protocol-card green-card'>
        <h4 style='color:#28a745;'>Laboratory Alerts</h4>
        <ul style='padding-left:20px'>
            <li>Creatinine >200 μmol/L → Renal consult</li>
            <li>AST >3×ULN → Hepatic workup</li>
        </ul>
    </div>
    
    <div class='protocol-card warning-card'>
        <h4 style='color:#ffc107;'>Surgical Indications</h4>
        <ul style='padding-left:20px'>
            <li>Ascending aorta involvement → Emergency surgery</li>
            <li>Rapid hematoma expansion → Endovascular repair</li>
        </ul>
    </div>
    """, unsafe_allow_html=True)

# Third column: Monitoring Protocol and Combined Protocol
with cols[2]:
    st.write("""
    <div class='protocol-card green-card'>
        <h4 style='color:#28a745;'>Monitoring & Standard Protocol</h4>
        <ul style='padding-left:20px'>
            <li>CT follow-up every 3.0 days</li>
            <li>Hourly vital signs</li>
            <li>Neuro checks every 0.17 days</li>
            <li>Daily CT ×3 days</li>
            <li>Additional supportive care and monitoring based on individual patient condition</li>
        </ul>
    </div>
    """, unsafe_allow_html=True)

# Define normal value ranges for continuous variables
normal_ranges = {
    'NEU': (1.8, 7.8),  # Normal range for NEU (10⁹/L)
    'AST': (10, 40),    # Normal range for AST (U/L)
    'CREA': (44, 106),  # Normal range for CREA (μmol/L)
    'DBP': (60, 90),    # Normal range for DBP (mmHg)
}

# Load model resources
try:
    model = pickle.load(open("gbm_model.pkl", "rb"))
    scaler = pickle.load(open("scaler.pkl", "rb"))
    features = [
        'CT-lesion involving ascending aorta', 'NEU', 'Age', 'CT-peritoneal effusion',
        'AST', 'CREA', 'Escape beat', 'DBP', 'CT-intramural hematoma'
    ]
except Exception as e:
    st.error(f"Initialization failed: {str(e)}")
    st.stop()

with st.sidebar:
    st.write("## Patient Parameters")
    with st.form("input_form"):
        inputs = {}
        
        # Continuous variables with normal ranges
        inputs['Age'] = st.slider("Age (Years)", 18, 100, 50)
        inputs['NEU'] = st.slider("NEU (10⁹/L)", 0.1, 25.0, 5.0)
        inputs['AST'] = st.slider("AST (U/L)", 0, 500, 30)
        inputs['CREA'] = st.slider("CREA (μmol/L)", 30, 200, 80)
        inputs['DBP'] = st.slider("DBP (mmHg)", 40, 160, 56)
        
        # Categorical variables
        inputs['CT-lesion involving ascending aorta'] = st.selectbox("CT lesion involving ascending aorta", ["No", "Yes"])
        inputs['CT-peritoneal effusion'] = st.selectbox("CT peritoneal effusion", ["No", "Yes"])
        inputs['Escape beat'] = st.selectbox("Escape beat", ["No", "Yes"])
        inputs['CT-intramural hematoma'] = st.selectbox("CT intramural hematoma", ["No", "Yes"])
        
        submitted = st.form_submit_button("Predict Risk")

# Process prediction
if submitted:
    try:
        # Data preprocessing
        input_data = {k: 1 if v == "Yes" else 0 if isinstance(v, str) else v for k, v in inputs.items()}
        df = pd.DataFrame([input_data], columns=features)
        df_scaled = scaler.transform(df)
        prob = model.predict_proba(df_scaled)[:, 1][0]
        risk_status = "High Risk" if prob >= 0.202 else "Low Risk"
        color = "#dc3545" if risk_status == "High Risk" else "#28a745"
        
        # Check for abnormal values and highlight them
        abnormal_vars = []
        advice = []

        for var, value in inputs.items():
            if var in normal_ranges:
                lower, upper = normal_ranges[var]
                if value < lower or value > upper:
                    abnormal_vars.append(var)
                    # Add the advice message
                    advice.append(f"<b>{var}:</b> Your value is {value}, while the normal range is {lower}-{upper}.")
                    if var == 'NEU':
                        advice.append("High NEU may indicate infection or inflammation. You should consider consulting a healthcare provider for further evaluation.")
                    elif var == 'AST':
                        advice.append("Elevated AST levels might indicate liver dysfunction. A liver function test is recommended to assess the underlying cause.")
                    elif var == 'CREA':
                        advice.append("High CREA levels could be a sign of kidney dysfunction. It is advisable to consult a nephrologist for proper evaluation and management.")
                    elif var == 'DBP':
                        advice.append("Elevated DBP (diastolic blood pressure) may indicate hypertension. Lifestyle modifications, including a balanced diet and regular exercise, may help lower blood pressure. Please consult your healthcare provider for further advice.")

        # Display results
        st.markdown(f"""
        <div class='result-card'>
            <h2 style='color:{color};'>Predicted Mortality Risk: {prob*100:.1f}% ({risk_status})</h2>
            <p>High risk of mortality within 1 year.</p>
        </div>
        """, unsafe_allow_html=True)
        
        # Display abnormal variables with advice
        if abnormal_vars:
            st.markdown("<h4 style='color: red;'>Personalized Advice:</h4>", unsafe_allow_html=True)
            st.markdown("<p style='color: red;'>Below are the abnormal values and their corresponding advice for correction:</p>", unsafe_allow_html=True)
            for adv in advice:
                st.markdown(f"<p>{adv}</p>", unsafe_allow_html=True)

    except Exception as e:
        st.error(f"Prediction error: {str(e)}")

# Footer
st.write("---")
st.write("<div style='text-align: center; color: gray;'>Developed by Yichang Central People's Hospital</div>", 
         unsafe_allow_html=True)
