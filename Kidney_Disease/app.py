import streamlit as st
import pandas as pd
import joblib
from sklearn.preprocessing import StandardScaler

# Load the trained models and scaler
logistic_model = joblib.load("logistic_regression_model.pkl")
decision_tree_model = joblib.load("decision_tree_model.pkl")
random_forest_model = joblib.load("random_forest_model.pkl")
svm_model = joblib.load("svm_model.pkl")
knn_model = joblib.load("knn_model.pkl")
xgboost_model = joblib.load("xgboost_model.pkl")
adaboost_model = joblib.load("adaboost_model.pkl")
scaler = joblib.load("scaler.pkl")

# Set page title and header
st.set_page_config(
    page_title="Kidney Disease Prediction",
    page_icon=":hospital:",
    layout="centered",
    initial_sidebar_state="expanded"
)
# Set background image

st.markdown(
    """
    <style>
    body {
        background-image: url('.//one.jpg');
        background-size: cover;
    }
    </style>
    """,
    unsafe_allow_html=True
)  
# Set sidebar background color and text color
st.markdown(
    """
    <style>
    body {
        background-image: url('one.jpg');
        background-repeat: no-repeat;
        background-size: cover;
    }
    </style>
    """,
    unsafe_allow_html=True
)


# Set main content background color and text color
st.markdown(
    """
    <style>
    .block-container {
        background-color: rgba(255, 255, 255, 0.8);
        color: #333333;
    }
    </style>
    """,
    unsafe_allow_html=True
)

# Set header styles
st.markdown(
    """
    <style>
    h1, h2, h3 {
        color: #4287f5;
    }
    </style>
    """,
    unsafe_allow_html=True
)

# Set sidebar header styles
st.markdown(
    """
    <style>
    .sidebar .sidebar-content .sidebar-title {
        color: #4287f5;
    }
    </style>
    """,
    unsafe_allow_html=True
)

# Set sidebar radio button styles
st.markdown(
    """
    <style>
    .sidebar .sidebar-content .radio-container .radio-label {
        color: #333333;
    }
    .sidebar .sidebar-content .radio-container .radio-button input:checked+span::before {
        border-color: #4287f5;
        background-color: #4287f5;
    }
    </style>
    """,
    unsafe_allow_html=True
)

# Set footer styles
st.markdown(
    """
    <style>
    footer {
        font-size: 14px;
        color: #666666;
        text-align: center;
        padding: 10px;
    }
    </style>
    """,
    unsafe_allow_html=True
)
# Main Heading
st.title("A Novel Machine Learning Based System for the Diagnosis of Chronic Kidney Disease")

# Input patient information
patient_name = st.text_input("Patient Name", "John Doe")
date_of_checkup = st.date_input("Date of Checkup")

# Model selection
model_name = st.sidebar.radio("Select Model", ("Logistic Regression", "Decision Tree", "Random Forest", "SVC", "KNN", "XGBoost", "AdaBoost"))

# Subheading
st.subheader("Input Features")

# Input form
with st.form(key='prediction_form'):
    # Input fields
    specific_gravity = st.number_input("Specific Gravity")
    albumin = st.number_input("Albumin")
    haemoglobin = st.number_input("Haemoglobin")
    packed_cell_volume = st.number_input("Packed Cell Volume")
    hypertension = st.number_input("Hypertension")
    diabetes_mellitus = st.number_input("Diabetes Mellitus")
    
    # Submit button
    submit_button = st.form_submit_button(label='Predict')

# Perform prediction on form submission
if submit_button:
    # Prepare input data
    input_data = pd.DataFrame({
        'specific_gravity': [specific_gravity],
        'albumin': [albumin],
        'haemoglobin': [haemoglobin],
        'packed_cell_volume': [packed_cell_volume],
        'hypertension': [hypertension],
        'diabetes_mellitus': [diabetes_mellitus]
    })
    
    # Scale the input data
    input_data_scaled = scaler.transform(input_data)
    

    # Perform prediction based on selected model
    if model_name == "Logistic Regression":
        prediction = logistic_model.predict(input_data_scaled)
        if prediction[0] == 1:
            st.subheader("Prediction Result")
            st.write("Person is suffering from Chronic Kidney Disease.")
        else:
            st.subheader("Prediction Result")
            st.write("Person is healthy and not suffering from Chronic Kidney Disease.")
    elif model_name == "Decision Tree":
        prediction = decision_tree_model.predict(input_data_scaled)
        if prediction[0] == 1:
            st.subheader("Prediction Result")
            st.write("Person is suffering from Chronic Kidney Disease.")
        else:
            st.subheader("Prediction Result")
            st.write("Person is healthy and not suffering from Chronic Kidney Disease.")
    elif model_name == "Random Forest":
        prediction = random_forest_model.predict(input_data_scaled)
        if prediction[0] == 1:
            st.subheader("Prediction Result")
            st.write("Person is suffering from Chronic Kidney Disease.")
        else:
            st.subheader("Prediction Result")
            st.write("Person is healthy and not suffering from Chronic Kidney Disease.")
    elif model_name == "SVC":
        prediction = svm_model.predict(input_data_scaled)
        if prediction[0] == 1:
            st.subheader("Prediction Result")
            st.write("Person is suffering from Chronic Kidney Disease.")
        else:
            st.subheader("Prediction Result")
            st.write("Person is healthy and not suffering from Chronic Kidney Disease.")
    elif model_name == "KNN":
        prediction = knn_model.predict(input_data_scaled)
        if prediction[0] == 1:
            st.subheader("Prediction Result")
            st.write("Person is suffering from Chronic Kidney Disease.")
        else:
            st.subheader("Prediction Result")
            st.write("Person is healthy and not suffering from Chronic Kidney Disease.")
    elif model_name == "XGBoost":
        prediction = xgboost_model.predict(input_data_scaled)
        if prediction[0] == 1:
            st.subheader("Prediction Result")
            st.write("Person is suffering from Chronic Kidney Disease.")
        else:
            st.subheader("Prediction Result")
            st.write("Person is healthy and not suffering from Chronic Kidney Disease.")
    elif model_name == "AdaBoost":
        prediction = adaboost_model.predict(input_data_scaled)
        if prediction[0] == 1:
            st.subheader("Prediction Result")
            st.write("Person is suffering from Chronic Kidney Disease.")
        else:
            st.subheader("Prediction Result")
            st.write("Person is healthy and not suffering from Chronic Kidney Disease.")


# Footer
st.markdown(
    """
    <footer>
    _________________________________________________
    
    </footer>
    """,
    unsafe_allow_html=True
)
