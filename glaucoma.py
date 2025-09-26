# %%

import streamlit as st
import pandas as pd
import pickle
import numpy as np
from sklearn.preprocessing import LabelEncoder

# Set page title
st.title("Glaucoma Prediction App")

# Load the trained glaucoma model
try:
    with open('glaucoma_model.pkl', 'rb') as file:
        glaucoma_model= pickle.load(file)
except FileNotFoundError:
    st.error("Model file 'glaucoma_model.pkl' not found. Please ensure it is in the same directory.")
    st.stop()

# Define the label encoder for decoding predictions
label_encoder = LabelEncoder()
label_encoder.classes_ = np.array(['Glaucoma', 'No Glaucoma'])  # adjust based on your dataset encoding

# Sidebar for user inputs
st.sidebar.header("Enter Patient Details")

# Example numerical features (adjust according to glaucoma dataset)
age = st.sidebar.slider("Age", min_value=20, max_value=80, value=50)
intraocular_pressure = st.sidebar.slider("Intraocular Pressure (mmHg)", min_value=10, max_value=40, value=20)
cup_disc_ratio = st.sidebar.slider("Cup-to-Disc Ratio", min_value=0.2, max_value=0.9, value=0.5, step=0.01)
central_corneal_thickness = st.sidebar.slider("Central Corneal Thickness (µm)", min_value=450, max_value=650, value=540)
visual_field_index = st.sidebar.slider("Visual Field Index (%)", min_value=0, max_value=100, value=85)

# Example categorical features
family_history = st.sidebar.selectbox("Family History of Glaucoma", options=["Yes", "No"])
diabetes = st.sidebar.selectbox("Diabetes", options=["Yes", "No"])
hypertension = st.sidebar.selectbox("Hypertension", options=["Yes", "No"])
gender = st.sidebar.selectbox("Gender", options=["Male", "Female"])

# Function to preprocess input data
def preprocess_input(age, intraocular_pressure, cup_disc_ratio, central_corneal_thickness, visual_field_index,
                     family_history, diabetes, hypertension, gender):
    data = {
        'Age': age,
        'IntraocularPressure': intraocular_pressure,
        'CupDiscRatio': cup_disc_ratio,
        'CentralCornealThickness': central_corneal_thickness,
        'VisualFieldIndex': visual_field_index,
        'FamilyHistory_Yes': 1 if family_history == 'Yes' else 0,
        'FamilyHistory_No': 1 if family_history == 'No' else 0,
        'Diabetes_Yes': 1 if diabetes == 'Yes' else 0,
        'Diabetes_No': 1 if diabetes == 'No' else 0,
        'Hypertension_Yes': 1 if hypertension == 'Yes' else 0,
        'Hypertension_No': 1 if hypertension == 'No' else 0,
        'Gender_Female': 1 if gender == 'Female' else 0,
        'Gender_Male': 1 if gender == 'Male' else 0
    }
    df = pd.DataFrame([data])

    # Expected columns (make sure they match your trained model)
    expected_columns = [
        'Age', 'IntraocularPressure', 'CupDiscRatio', 'CentralCornealThickness', 'VisualFieldIndex',
        'FamilyHistory_No', 'FamilyHistory_Yes',
        'Diabetes_No', 'Diabetes_Yes',
        'Hypertension_No', 'Hypertension_Yes',
        'Gender_Female', 'Gender_Male'
    ]
    df = df.reindex(columns=expected_columns, fill_value=0)
    return df

# Button to make prediction
if st.sidebar.button("Predict"):
    input_df = preprocess_input(
        age, intraocular_pressure, cup_disc_ratio, central_corneal_thickness, visual_field_index,
        family_history, diabetes, hypertension, gender
    )
    
    try:
        prediction = glaucoma_model.predict(input_df)
        predicted_label = label_encoder.inverse_transform(prediction)[0]
        
        # Display result
        st.subheader("Prediction Result")
        st.write(f"The predicted outcome is: **{predicted_label}**")
        if predicted_label == "No Glaucoma":
            st.write("The patient is unlikely to have glaucoma.")
        else:
            st.write("The patient may have **Glaucoma**. Further medical examination is advised.")
    except Exception as e:
        st.error(f"Error making prediction: {str(e)}")

# Display instructions
st.write("""
### Instructions
1. Use the sidebar to enter the patient's clinical details.
2. Adjust the sliders for numerical features like Age, Intraocular Pressure, etc.
3. Select appropriate options for categorical features like Family History, Diabetes, etc.
4. Click the 'Predict' button to see the predicted glaucoma outcome.
""")
