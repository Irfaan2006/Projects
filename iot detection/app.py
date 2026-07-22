import streamlit as st
import pandas as pd
import joblib
import os

st.set_page_config(page_title="IoT Equipment Monitoring", page_icon="🏭", layout="wide")

st.markdown("""
<style>
.stApp{background:linear-gradient(135deg,#eef6ff,#ffffff);}
[data-testid="stSidebar"]{background:#0f172a;}
[data-testid="stSidebar"] *{color:white;}
.block-container{padding-top:1rem;}
.card{background:white;padding:18px;border-radius:14px;box-shadow:0 2px 10px rgba(0,0,0,.1);}
.stButton>button{width:100%;background:#2563eb;color:white;border-radius:10px;height:48px;}
</style>
""", unsafe_allow_html=True)

BASE=os.path.dirname(os.path.abspath(__file__))
model=joblib.load(os.path.join(BASE,"iot_fault_detection_model.pkl"))

st.title("🏭 IoT Equipment Monitoring Dashboard")
page=st.sidebar.radio("Navigation",["Dashboard","Prediction","Dataset","About"])

if page=="Dashboard":
    c1,c2,c3,c4=st.columns(4)
    c1.metric("Model","Loaded")
    c2.metric("Sensors","Live")
    c3.metric("Status","Monitoring")
    c4.metric("Version","1.0")
    st.info("Use the sidebar to predict equipment faults or explore the dataset.")

elif page=="Prediction":
    col1,col2=st.columns(2)
    with col1:
        sensor_id=st.text_input("Sensor ID","S001")
        temperature=st.number_input("Temperature",25.0)
        vibration=st.number_input("Vibration",0.25)
        pressure=st.number_input("Pressure",101.3)
        voltage=st.number_input("Voltage",220.0)
        current=st.number_input("Current",10.0)
        fft1=st.number_input("FFT Feature 1",0.0)
    with col2:
        fft2=st.number_input("FFT Feature 2",0.0)
        norm_temp=st.number_input("Normalized Temp",0.5)
        norm_vib=st.number_input("Normalized Vibration",0.5)
        norm_press=st.number_input("Normalized Pressure",0.5)
        norm_voltage=st.number_input("Normalized Voltage",0.5)
        norm_current=st.number_input("Normalized Current",0.5)
        anomaly=st.number_input("Anomaly Score",0.1)
    fault_type=st.selectbox("Fault Type",["Normal","Bearing Fault","Voltage Drop","Pressure Leak","Current Spike","Overheat"])
    if st.button("Predict"):
        X=pd.DataFrame({
            "Sensor_ID":[sensor_id],"Temperature":[temperature],"Vibration":[vibration],
            "Pressure":[pressure],"Voltage":[voltage],"Current":[current],
            "FFT_Feature1":[fft1],"FFT_Feature2":[fft2],
            "Normalized_Temp":[norm_temp],"Normalized_Vibration":[norm_vib],
            "Normalized_Pressure":[norm_press],"Normalized_Voltage":[norm_voltage],
            "Normalized_Current":[norm_current],"Anomaly_Score":[anomaly],
            "Fault_Type":[fault_type]
        })
        pred=model.predict(X)[0]
        if pred==1:
            st.error("🚨 Fault Detected")
        else:
            st.success("✅ Sensor is Normal")

elif page=="Dataset":
    path=os.path.join(BASE,"iot_equipment_monitoring_dataset.csv")
    if os.path.exists(path):
        df=pd.read_csv(path)
        st.dataframe(df,use_container_width=True)
        st.write(df.describe())
    else:
        st.warning("Dataset file not found.")
else:
    st.write("Industrial IoT Fault Detection System built with Streamlit.")
