import streamlit as st
import pickle
import numpy as np
import pandas as pd

# Page Configuration
st.set_page_config(
    page_title="Student Performance Predictor",
    page_icon="🎓",
    layout="centered"
)

# Custom CSS for proper shadows, layout, and styling
st.markdown("""
    <style>
    .main {
        background-color: #f8f9fa;
    }
    .card {
        background-color: #ffffff;
        padding: 30px;
        border-radius: 12px;
        box-shadow: 0 4px 12px rgba(0, 0, 0, 0.08);
        margin-bottom: 20px;
    }
    .stButton>button {
        width: 100%;
        background-color: #ff4b4b;
        color: white;
        font-weight: bold;
        border-radius: 8px;
        padding: 0.6rem;
        border: none;
        box-shadow: 0 4px 6px rgba(255, 75, 75, 0.2);
    }
    .stButton>button:hover {
        background-color: #e03e3e;
    }
    h1, h2, h3 {
        color: #1f2937;
    }
    </style>
""", unsafe_allow_html=True)

# Load the trained model safely
@st.cache_resource
def load_model():
    with open('knn.pkl', 'rb') as file:
        model = pickle.load(file)
    return model

try:
    model = load_model()
except Exception as e:
    st.error(f"Error loading 'knn.pkl': {e}")
    st.stop()

# App Header
st.markdown("<h1 style='text-align: center;'>🎓 Student Performance Predictor</h1>", unsafe_allow_html=True)
st.markdown("<p style='text-align: center; color: #6b7280;'>Enter student details below to estimate performance using a K-Nearest Neighbors (KNN) Regressor.</p>", unsafe_allow_html=True)
st.markdown("---")

# Main Input Layout inside a structured card container
with st.container():
    st.markdown("### 📝 Student Information Input")
    
    col1, col2 = st.columns(2)
    
    with col1:
        age = st.number_input("Age", min_value=10, max_value=100, value=20, step=1)
        gender = st.selectbox("Gender", ["Male", "Female", "Other"])
        course = st.selectbox("Course / Field of Study", ["Engineering", "Science", "Arts", "Business", "Medicine"])
        study_hours = st.slider("Weekly Study Hours", min_value=0.0, max_value=60.0, value=15.0, step=0.5)
        class_attendance = st.slider("Class Attendance (%)", min_value=0.0, max_value=100.0, value=85.0, step=1.0)
        internet_access = st.selectbox("Internet Access", ["Yes", "No"])
        
    with col2:
        sleep_hours = st.slider("Average Sleep Hours (Daily)", min_value=0.0, max_value=14.0, value=7.0, step=0.5)
        sleep_quality = st.selectbox("Sleep Quality", ["Good", "Average", "Poor"])
        study_method = st.selectbox("Primary Study Method", ["Self-study", "Group study", "Online videos", "Textbooks"])
        facility_rating = st.slider("Campus Facility Rating (1-10)", min_value=1, max_value=10, value=7, step=1)
        exam_difficulty = st.selectbox("Exam Difficulty Level", ["Easy", "Moderate", "Hard"])

    st.markdown("<br>", unsafe_allow_html=True)
    predict_btn = st.button("Predict Performance")

# Prediction and Results Section
if predict_btn:
    # Build a DataFrame containing the exact feature columns expected by scikit-learn
    # Note: If your training pipeline used one-hot encoding or a ColumnTransformer, 
    # adjust the mapping below to match the exact format used during training.
    input_data = pd.DataFrame({
        'age': [age],
        'gender': [gender],
        'course': [course],
        'study_hours': [study_hours],
        'class_attendance': [class_attendance],
        'internet_access': [internet_access],
        'sleep_hours': [sleep_hours],
        'sleep_quality': [sleep_quality],
        'study_method': [study_method],
        'facility_rating': [facility_rating],
        'exam_difficulty': [exam_difficulty]
    })

    try:
        # Generate prediction
        prediction = model.predict(input_data)[0]
        
        st.markdown("---")
        st.markdown("### 📊 Prediction Result")
        st.success(f"### Estimated Performance Score: **{prediction:.2f}**")
        
    except Exception as e:
        st.error(f"Prediction failed. Note: If your model expects numerical encoded values instead of raw text categories, make sure to apply the same label encoder used during training. Details: {e}")
