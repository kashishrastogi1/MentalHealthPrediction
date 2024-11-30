# import streamlit as st
# import pickle
# import numpy as np
# import joblib

# # Load the model, scaler, and encoder
# with open("model.pkl", "rb") as model_file:
#     model = joblib.load(model_file)

# with open("le.pkl", "rb") as le_file:
#     le = joblib.load(le_file)

# with open("preprocessor.pkl", "rb") as preprocessor_file:
#     preprocessor = joblib.load(preprocessor_file)
    
# print(le.classes_)

# st.title("Mental Health Prediction")
# st.header("Please provide your details:")

# # User input
# gender = st.selectbox("Gender", ['Female' ,'Male'])
# age = st.number_input("Age", min_value=1, max_value=120, value=25)
# working_status = st.selectbox("Working Professional or Student", ["Working Professional", "Student"])
# profession = st.selectbox("Profession", [
#     "Teacher", "Content Writer", "Architect", "Consultant", "HR Manager", "Pharmacist", 
#     "Doctor", "Business Analyst", "Entrepreneur", "Chemist", "Chef", "Educational Consultant", 
#     "Data Scientist", "Researcher", "Lawyer", "Customer Support", "Marketing Manager", 
#     "Pilot", "Travel Consultant", "Plumber", "Sales Executive", "Manager", "Judge", 
#     "Electrician", "Financial Analyst", "Software Engineer", "Civil Engineer", "UX/UI Designer", 
#     "Digital Marketer", "Accountant", "Finanancial Analyst", "Mechanical Engineer", 
#     "Graphic Designer", "Research Analyst", "Investment Banker"
# ])
# cgpa = st.number_input("CGPA", min_value=0.0, max_value=10.0)
# sleep_duration = st.selectbox("Sleep Duration", ['More than 7 hours', 'Less than 7 hours'])
# dietary_habits = st.selectbox("Dietary Habits", ["Healthy", "Unhealthy", "Moderate"])
# suicidal_thoughts = st.selectbox("Have you ever had suicidal thoughts?", ["No", "Yes"])
# work_study_hours = st.slider("Work/Study Hours", 1, 12)
# financial_stress = st.slider("Financial Stress", 1, 5)
# family_history = st.selectbox("Family History of Mental Illness", ["No", "Yes"])
# pressure = st.slider("Work/Academic Pressure", 1, 5)
# satisfaction = st.slider("Job/Study Satisfaction", 1, 5)

# if st.button("Predict"):
#     # Prepare the data for prediction (categorical and numerical inputs separately)
    
#     # Binary categorical features (use LabelEncoder)
#     cat_binary_features = {
#         "Gender": gender,
#         "Working Professional or Student": working_status,
#         "Sleep Duration": sleep_duration,
#         "Have you ever had suicidal thoughts ?": suicidal_thoughts,
#         "Family History of Mental Illness": family_history
#     }
    
#     # Multi-categorical features (e.g., Profession, Dietary Habits)
#     cat_multi_features = {
#         "Profession": profession,
#         "Dietary Habits": dietary_habits
#     }
    
#     # Numerical features
#     num_features = {
#         "Age": age,
#         "CGPA": cgpa,
#         "Work/Study Hours": work_study_hours,
#         "Financial Stress": financial_stress,
#         "Work/Academic Pressure": pressure,
#         "Job/Study Satisfaction": satisfaction
#     }
    
#     # Encoding binary categorical features using LabelEncoder
    
#     cat_binary_features_transformed = []
#     for col in cat_binary_features:
#     # Transform the feature using the label encoder and append to the list
#         cat_binary_features_transformed.append(le.transform([cat_binary_features[col]])[0])
    
#     # One-hot encoding for multi-categorical features (e.g., Profession, Dietary Habits)
#     cat_multi_data = np.array([cat_multi_features['Profession'], cat_multi_features['Dietary Habits']]).reshape(1, -1)
    
#     # Apply one-hot encoding for 'Profession' and 'Dietary Habits' using the preprocessor
#     cat_multi_features_transformed = preprocessor.transform(cat_multi_data)
    
#     # Prepare the numerical features
#     num_features_array = np.array(list(num_features.values())).reshape(1, -1)
    
#     # Preprocess the numerical features using the preprocessor (which includes StandardScaler)
#     num_features_transformed = preprocessor.transform(num_features_array)
    
#     # Reshape binary features for concatenation
#     cat_binary_features_transformed = np.array(cat_binary_features_transformed).reshape(1, -1)
    
#     # Combine the binary, one-hot encoded, and numerical features
#     final_input = np.concatenate([cat_binary_features_transformed, cat_multi_features_transformed, num_features_transformed], axis=1)
    
#     # Make the prediction
#     prediction = model.predict(final_input)
    
#     # Show the result
#     if prediction == 1:
#         st.write("The model predicts that the person might be depressed.")
#     else:
#         st.write("The model predicts that the person is not depressed.")

import streamlit as st
import pickle
import joblib
import numpy as np
import pandas as pd

# Load the model, preprocessor, and saved LabelEncoders
model = joblib.load("model.pkl")
preprocessor = joblib.load("preprocessor.pkl")

# Load the LabelEncoders
gender_le = joblib.load("Gender_le.pkl")
working_status_le = joblib.load("Working_Professional_or_Student_le.pkl")
sleep_duration_le = joblib.load("Sleep_Duration_le.pkl")
suicidal_thoughts_le = joblib.load("Have_you_ever_had_suicidal_thoughts__le.pkl")
family_history_le = joblib.load("Family_History_of_Mental_Illness_le.pkl")

st.title("Mental Health Prediction")
st.header("Please provide your details:")

# User input form
gender = st.selectbox("Gender", ["Male", "Female"])
age = st.number_input("Age", min_value=1, max_value=120, value=25)
working_status = st.selectbox("Working Professional or Student", ["Working Professional", "Student"])
profession = st.selectbox("Profession", [
    "Teacher", "Content Writer", "Architect", "Consultant", "HR Manager", "Pharmacist", 
    "Doctor", "Business Analyst", "Entrepreneur", "Chemist", "Chef", "Educational Consultant", 
    "Data Scientist", "Researcher", "Lawyer", "Customer Support", "Marketing Manager", 
    "Pilot", "Travel Consultant", "Plumber", "Sales Executive", "Manager", "Judge", 
    "Electrician", "Financial Analyst", "Software Engineer", "Civil Engineer", "UX/UI Designer", 
    "Digital Marketer", "Accountant", "Financial Analyst", "Mechanical Engineer", 
    "Graphic Designer", "Research Analyst", "Investment Banker"
])
cgpa = st.number_input("CGPA", min_value=0.0, max_value=10.0)
sleep_duration = st.selectbox("Sleep Duration", ['More than 7 hours', 'Less than 7 hours'])
dietary_habits = st.selectbox("Dietary Habits", ["Healthy", "Unhealthy", "Moderate"])
suicidal_thoughts = st.selectbox("Have you ever had suicidal thoughts?", ["No", "Yes"])
work_study_hours = st.slider("Work/Study Hours", 1, 12)
financial_stress = st.slider("Financial Stress", 1, 5)
family_history = st.selectbox("Family History of Mental Illness", ["No", "Yes"])
pressure = st.slider("Work/Academic Pressure", 1, 5)
satisfaction = st.slider("Job/Study Satisfaction", 1, 5)

if st.button("Predict"):
    # Prepare the data for prediction (categorical and numerical inputs separately)
    cat_binary_features = {
        "Gender": gender,
        "Working Professional or Student": working_status,
        "Sleep Duration": sleep_duration,
        "Have you ever had suicidal thoughts ?": suicidal_thoughts,
        "Family History of Mental Illness": family_history
    }

    cat_multi_features = {
        "Profession": profession,
        "Dietary Habits": dietary_habits
    }

    num_features = {
        "Age": age,
        "CGPA": cgpa,
        "Work/Study Hours": work_study_hours,
        "Financial Stress": financial_stress,
        "Work/Academic Pressure": pressure,
        "Job/Study Satisfaction": satisfaction
    }

    # Encode binary categorical features using LabelEncoder
    cat_binary_features_transformed = [
        gender_le.transform([cat_binary_features["Gender"]])[0],
        working_status_le.transform([cat_binary_features["Working Professional or Student"]])[0],
        sleep_duration_le.transform([cat_binary_features["Sleep Duration"]])[0],
        suicidal_thoughts_le.transform([cat_binary_features["Have you ever had suicidal thoughts ?"]])[0],
        family_history_le.transform([cat_binary_features["Family History of Mental Illness"]])[0]
    ]

    # Create a dictionary of all the features
    all_features = {**cat_binary_features, **cat_multi_features, **num_features}

    # Convert the features into a pandas DataFrame
    all_features_df = pd.DataFrame([all_features])

    # Apply the preprocessor to all features
    all_features_transformed = preprocessor.transform(all_features_df)

    # Make the prediction
    prediction = model.predict(all_features_transformed)
    
    # Show the result
    if prediction == 1:
        st.write("The model predicts that the person might be depressed.")
    else:
        st.write("The model predicts that the person is not depressed.")





