import streamlit as st
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# Streamlit app
st.title("Math Score Prediction App")

# Use a fixed file path for StudentsPerformance.csv
file_path = "StudentsPerformance.csv"

# Load dataset
try:
    df = pd.read_csv(file_path)
    st.write("### Dataset Preview:")
    st.dataframe(df.head(10))
except FileNotFoundError:
    st.error(f"File '{file_path}' not found. Please check the file path and try again.")
    st.stop()

# Add computed columns if necessary
if 'total score' not in df.columns or 'avg score' not in df.columns:
    df['total score'] = df[['math score', 'reading score', 'writing score']].sum(axis=1)
    df['avg score'] = df[['math score', 'reading score', 'writing score']].mean(axis=1)

# Using only 'math score' for prediction
Y = df['math score']
X = df[['reading score', 'writing score']]

# Train-test split
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=42)

# Train model
model = DecisionTreeRegressor()
model.fit(X_train, Y_train)

# Predict on the test data
Y_pred_test = model.predict(X_test)

# Metrics for test data
mae = mean_absolute_error(Y_test, Y_pred_test)
mse = mean_squared_error(Y_test, Y_pred_test)
r2 = r2_score(Y_test, Y_pred_test)

st.write("### Model Performance Metrics")
st.write(f"Mean Absolute Error (MAE): {mae:.2f}")
st.write(f"Mean Squared Error (MSE): {mse:.2f}")
st.write(f"R² Score: {r2:.2f}")

# Input box for prediction
st.write("### Predict Your Math Score")
reading_score = st.number_input("Enter Reading Score:", min_value=0, max_value=100, step=1)
writing_score = st.number_input("Enter Writing Score:", min_value=0, max_value=100, step=1)

# Perform prediction
if st.button("Predict Math Score"):
    user_input = pd.DataFrame([[reading_score, writing_score]], columns=['reading score', 'writing score'])
    predicted_score = model.predict(user_input)
    st.success(f"The Predicted Math Score is: {predicted_score[0]:.2f}")
