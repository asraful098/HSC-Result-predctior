import pickle
 
import numpy as np
import pandas as pd
import streamlit as st
 
st.set_page_config(page_title="HSC Result Predictor", page_icon="🎓")
 
ALL_COLUMNS = [
    "gender", "age", "address", "famsize", "Pstatus", "M_Edu", "F_Edu",
    "M_Job", "F_Job", "relationship", "smoker", "tuition_fee",
    "time_friends", "ssc_result",
]
 
 
@st.cache_resource
def load_model():
    with open("student_rf_pipeline.pkl", "rb") as f:
        return pickle.load(f)
 
 
model = load_model()
 
st.title("🎓 HSC Result Predictor")
st.caption("Fill in the student's details and click Predict.")
 
with st.form("predict_form"):
    col1, col2 = st.columns(2)
 
    with col1:
        gender = st.radio("Gender", ["M", "F"], horizontal=True)
        age = st.number_input("Age", min_value=10, max_value=40, value=18, step=1)
        address = st.radio("Address", ["Rural", "Urban"], horizontal=True)
        famsize = st.radio("Family size", ["GT3", "LE3"], horizontal=True)
        pstatus = st.radio("Parents' status", ["Together", "Apart"], horizontal=True)
        rls = st.radio("Relationship", ["Yes", "NO"], horizontal=True)
        smoker = st.radio("Smoker", ["Yes", "NO"], horizontal=True)
 
    with col2:
        m_edu = st.slider("Mother's education", 0, 4, 2, step=1)
        f_edu = st.slider("Father's education", 0, 4, 2, step=1)
        m_job = st.selectbox(
            "Mother's job", ["At_home", "Health", "Other", "Services", "Teacher"]
        )
        f_job = st.selectbox(
            "Father's job", ["Business", "Health", "Farmer", "Services", "Teacher"]
        )
        tuition_fee = st.number_input("Tuition fee", min_value=0.0, value=0.0, step=100.0)
        time_friends = st.slider("Time with friends", 0, 5, 2, step=1)
        ssc_result = st.number_input(
            "SSC result (GPA)", min_value=0.0, max_value=5.0, value=4.0, step=0.01
        )
 
    submitted = st.form_submit_button("Predict", use_container_width=True)
 
if submitted:
    input_df = pd.DataFrame(
        [[
            gender, age, address, famsize, pstatus, m_edu, f_edu, m_job, f_job,
            rls, smoker, tuition_fee, time_friends, ssc_result,
        ]],
        columns=ALL_COLUMNS,
    )
 
    prediction = model.predict(input_df)[0]
    result = float(np.clip(prediction, 0, 5))
 
    st.success(f"Predicted HSC result: **{result:.2f}**")
    st.progress(result / 5)