import streamlit as st
import numpy as np
import pickle

# Load the serialized preprocessing pipeline
preprocessing_filename = "scaler.pkl"  # Use .pkl extension
with open(preprocessing_filename, "rb") as f:
    loaded_prepro = pickle.load(f)

# Load the serialized KNN model
model_filename = "knn_HD.pkl"  # Use .pkl extension
with open(model_filename, "rb") as f:
    loaded_knn_model = pickle.load(f)

st.title("Heart Disease Prediction")

# User details panel
st.sidebar.header("User Details")
name = st.sidebar.text_input("Full Name:")
age = st.sidebar.number_input("Age:", value=40, min_value=0, max_value=100)
phone = st.sidebar.text_input("Phone Number:")
address = st.sidebar.text_area("Address:")

# User inputs
sex = st.radio("Sex:", ["Male", "Female"])
cp = st.selectbox("Chest Pain Type:", [0, 1, 2, 3])
trestbps = st.number_input("Resting Blood Pressure (mm Hg):", value=120, min_value=0)
chol = st.number_input("Cholesterol Serum (mg/dl):", value=200, min_value=0)
fbs = st.radio("Fasting Blood Sugar > 120 mg/dl:", ["False", "True"])
restecg = st.selectbox("Resting Electrocardiographic Results:", [0, 1, 2])
thalach = st.number_input("Max Heart Rate Achieved:", value=150, min_value=0)
exang = st.radio("Exercise Induced Angina:", ["No", "Yes"])
oldpeak = st.number_input("Depression Induced by Exercise:", value=0, min_value=0)
slope = st.selectbox("Slope of ST Segment:", [0, 1, 2])
ca = st.selectbox("Number of Major Vessels Colored by Fluoroscopy:", [0, 1, 2, 3])
thal = st.selectbox("Thalassemia:", [1, 2, 7])

# Convert categorical features to numeric values
sex_encoded = 0 if sex == "Male" else 1
fbs_encoded = 1 if fbs == "True" else 0
exang_encoded = 1 if exang == "Yes" else 0

# Create a dictionary with feature names and values
input_dict = {
    "Age": age,
    "RestingBP": trestbps,
    "Cholesterol": chol,
    "MaxHeartRate": thalach,
    "OldPeak": oldpeak,
    "NumMajorVessels": ca,
    "Gender_0": 1 - sex_encoded,
    "Gender_1": sex_encoded,
    "ChestPain_0": 1 if cp == 0 else 0,
    "ChestPain_1": 1 if cp == 1 else 0,
    "ChestPain_2": 1 if cp == 2 else 0,
    "ChestPain_3": 1 if cp == 3 else 0,
    "FastingSugar_0": 1 - fbs_encoded,
    "FastingSugar_1": fbs_encoded,
    "RestECG_0": 1 if restecg == 0 else 0,
    "RestECG_1": 1 if restecg == 1 else 0,
    "RestECG_2": 1 if restecg == 2 else 0,
    "ExerciseInducedAngina_0": 1 - exang_encoded,
    "ExerciseInducedAngina_1": exang_encoded,
    "Slope_0": 1 if slope == 0 else 0,
    "Slope_1": 1 if slope == 1 else 0,
    "Slope_2": 1 if slope == 2 else 0,
    "Thalassemia_0": 1 if thal == 0 else 0,
    "Thalassemia_1": 1 if thal == 1 else 0,
    "Thalassemia_2": 1 if thal == 2 else 0,
    "Thalassemia_3": 1 if thal == 3 else 0,
}

# Prediction button
if st.button("Predict"):
    # Transform user inputs using the loaded preprocessing pipeline
    input_array = np.array(list(input_dict.values())).reshape(1, -1)
    input_prepared = loaded_prepro.transform(input_array)

    # Make predictions using the loaded KNN model
    prediction = loaded_knn_model.predict(input_prepared)

    # Display user details
    st.write("User Details:")
    st.write(f"Name: {name}")
    st.write(f"Age: {age}")
    st.write(f"Phone Number: {phone}")
    st.write(f"Address: {address}")

    # Display the input features and prediction


    st.write("Prediction:", "Heart Disease" if prediction else "No Heart Disease")
