import streamlit as st
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeRegressor
from sklearn.preprocessing import StandardScaler

# Streamlit app
st.title("Math Score Prediction App")

# Use a fixed file path for StudentsPerformance.csv
file_path = "/mnt/data/StudentsPerformance.csv"  # Adjusted to the uploaded file path

# Load dataset
try:
    df = pd.read_csv(file_path)
except FileNotFoundError:
    st.error(f"File '{file_path}' not found. Please check the file path and try again.")
    st.stop()

# Add computed columns if necessary
if 'total score' not in df.columns or 'avg score' not in df.columns:
    df['total score'] = df[['math score', 'reading score', 'writing score']].sum(axis=1)
    df['avg score'] = df[['math score', 'reading score', 'writing score']].mean(axis=1)

# Preprocessing
X = df[['total score', 'avg score']]  # Use total and average score as features
Y = df['math score']

# Standardize numerical features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Train-test split
X_train, X_test, Y_train, Y_test = train_test_split(X_scaled, Y, test_size=0.2, random_state=42)

# Train model
model = DecisionTreeRegressor()
model.fit(X_train, Y_train)

# Input section for prediction
st.write("### Predict Your Own Math Score")

# Input fields for total and average score
total_score = st.number_input("Total Score", min_value=float(df['total score'].min()), max_value=float(df['total score'].max()), step=1.0)
avg_score = st.number_input("Average Score", min_value=float(df['avg score'].min()), max_value=float(df['avg score'].max()), step=0.1)

# Create user input DataFrame
user_input = pd.DataFrame([[total_score, avg_score]], columns=['total score', 'avg score'])

# When the user clicks the 'Submit' button, predict and display the math score
if st.button("Submit"):
    try:
        # Transform user input and predict
        user_input_scaled = scaler.transform(user_input)
        user_prediction = model.predict(user_input_scaled)

        # Display prediction
        st.write(f"Predicted Math Score: {user_prediction[0]:.2f}")
    except ValueError as e:
        st.error(f"Error during prediction: {e}")
