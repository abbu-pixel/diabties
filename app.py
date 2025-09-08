import streamlit as st
import numpy as np
import pickle

# Load the trained machine learning model
try:
    with open("model.pkl", "rb") as f:
        model = pickle.load(f)
except FileNotFoundError:
    st.error("Error: model.pkl not found. Please run model.py first to create the model file.")
    st.stop()

# Streamlit app
st.title("Diabetes Prediction App")
st.write("Enter the patient details below to predict diabetes:")

# Collect input values from user
glucose = st.number_input("Glucose Level", min_value=0.0, step=0.1)
blood_pressure = st.number_input("Blood Pressure", min_value=0.0, step=0.1)
skin_thickness = st.number_input("Skin Thickness", min_value=0.0, step=0.1)
insulin = st.number_input("Insulin", min_value=0.0, step=0.1)
bmi = st.number_input("BMI", min_value=0.0, step=0.1)
diabetes_pedigree_function = st.number_input("Diabetes Pedigree Function", min_value=0.0, step=0.01)
age = st.number_input("Age", min_value=0, step=1)

# Predict button
if st.button("Predict"):
    try:
        # Prepare features
        features = [
            glucose,
            blood_pressure,
            skin_thickness,
            insulin,
            bmi,
            diabetes_pedigree_function,
            age,
        ]
        final_features = np.array(features).reshape(1, -1)

        # Prediction
        prediction = model.predict(final_features)[0]

        if prediction == 1:
            st.error("Based on the data, the patient is predicted to have diabetes.")
        else:
            st.success("Based on the data, the patient is predicted to be non-diabetic.")

    except Exception as e:
        st.error(f"An error occurred: {str(e)}. Please check your input.")
