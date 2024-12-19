# -*- coding: utf-8 -*-

import numpy as np
import pandas as pd
import streamlit as st
from sklearn.preprocessing import LabelEncoder, MinMaxScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

# Streamlit setup
st.title("Student Performance Prediction")
st.write("This app predicts student performance based on input features.")

# Input data from the user
gender = st.selectbox("Gender", ["male", "female"])
race_ethnicity = st.selectbox("Race/Ethnicity", ["group A", "group B", "group C", "group D", "group E"])
parental_education = st.selectbox("Parental Level of Education", [
    "some high school", "high school", "some college", "associate's degree", "bachelor's degree", "master's degree"
])
lunch = st.selectbox("Lunch", ["standard", "free/reduced"])
test_prep = st.selectbox("Test Preparation Course", ["none", "completed"])
math_score = st.slider("Math Score", 0, 100, 75)
reading_score = st.slider("Reading Score", 0, 100, 75)
writing_score = st.slider("Writing Score", 0, 100, 75)

# Combine inputs into a DataFrame
input_data = pd.DataFrame({
    'gender': [gender],
    'race/ethnicity': [race_ethnicity],
    'parental level of education': [parental_education],
    'lunch': [lunch],
    'test preparation course': [test_prep],
    'math score': [math_score],
    'reading score': [reading_score],
    'writing score': [writing_score]
})

# Load dataset
data = pd.read_csv('StudentsPerformance.csv')

# Check if 'performance' column exists
if 'performance' not in data.columns:
    st.error("The dataset is missing the 'performance' column. Please ensure the dataset is correct.")
else:
    # Encoding categorical features
    encoders = {}
    for col in ['gender', 'race/ethnicity', 'parental level of education', 'lunch', 'test preparation course']:
        le = LabelEncoder()
        data[col] = le.fit_transform(data[col])
        encoders[col] = le

    # Scaling numerical features
    scaler = MinMaxScaler()
    numerical_columns = ['math score', 'reading score', 'writing score']
    data[numerical_columns] = scaler.fit_transform(data[numerical_columns])

    # Splitting data for training and testing
    X = data.drop('performance', axis=1)
    y = data['performance']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Model training
    model = RandomForestClassifier(random_state=42)
    model.fit(X_train, y_train)

    # Testing accuracy (optional for display)
    y_pred = model.predict(X_test)
    st.write("Model Accuracy on Test Data:", accuracy_score(y_test, y_pred))

    # Preprocess user input
    def preprocess_input(input_df):
        for col, encoder in encoders.items():
            input_df[col] = encoder.transform(input_df[col])
        input_df[numerical_columns] = scaler.transform(input_df[numerical_columns])
        return input_df

    input_data_processed = preprocess_input(input_data)

    # Predict
    if st.button("Predict"):
        prediction = model.predict(input_data_processed)
        st.write("Predicted Outcome:", "High Performance" if prediction[0] == 1 else "Low Performance")
