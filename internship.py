# -*- coding: utf-8 -*-
"""Untitled28.ipynb

Automatically generated by Colab.

Original file is located at
    https://colab.research.google.com/drive/1Rxqe5WEfW6mjQeQTLII-n9leZ_HNn_LD
"""

import numpy as np
import pandas as pd
import streamlit as st
from sklearn.preprocessing import LabelEncoder, MinMaxScaler
from sklearn.ensemble import RandomForestClassifier

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

# Encoding and model setup
encoders = {
    'gender': LabelEncoder().fit(['male', 'female']),
    'race/ethnicity': LabelEncoder().fit(['group A', 'group B', 'group C', 'group D', 'group E']),
    'parental level of education': LabelEncoder().fit([
        'some high school', 'high school', 'some college', "associate's degree", "bachelor's degree", "master's degree"
    ]),
    'lunch': LabelEncoder().fit(['standard', 'free/reduced']),
    'test preparation course': LabelEncoder().fit(['none', 'completed'])
}

scaler = MinMaxScaler()
scaler.fit(np.random.rand(1000, 3))

model = RandomForestClassifier()
model.fit(np.random.rand(1000, 8), np.random.randint(0, 2, 1000))

# Preprocess input data
for col, encoder in encoders.items():
    input_data[col] = encoder.transform(input_data[col])

numerical_columns = ['math score', 'reading score', 'writing score']
input_data_scaled = input_data.copy()
input_data_scaled[numerical_columns] = scaler.transform(input_data[numerical_columns])

# Predict
if st.button("Predict"):
    predictions = model.predict(input_data_scaled[numerical_columns + list(encoders.keys())])
    st.write("Predicted Outcome:", predictions[0])