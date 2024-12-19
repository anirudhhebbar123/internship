import streamlit as st
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer

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

# Preprocessing
st.write("### Preprocessing and Feature Transformation")
columns_to_drop = ['total score', 'avg score', 'math score']

# Safely drop columns that exist in the dataset
X = df.drop(columns=[col for col in columns_to_drop if col in df.columns], axis=1)
Y = df['math score']

# Feature transformation
num_features = X.select_dtypes(exclude="object").columns
cat_features = X.select_dtypes(include="object").columns

numeric_transformer = StandardScaler()
oh_transformer = OneHotEncoder()

preprocessor = ColumnTransformer(
    [
        ("OneHotEncoder", oh_transformer, cat_features),
        ("StandardScaler", numeric_transformer, num_features),
    ]
)

# Handle transformation
try:
    X = preprocessor.fit_transform(X)
    st.write(f"Transformed Feature Shape: {X.shape}")
except ValueError as e:
    st.error(f"Error during feature transformation: {e}")
    st.stop()

# Train-test split
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=42)

# Train model
model = DecisionTreeRegressor()
model.fit(X_train, Y_train)

# Predict on the entire dataset
Y_pred_all = model.predict(X)

# Create a DataFrame with Actual and Predicted values for the entire dataset
results_df = pd.DataFrame({
    'Actual Math Score': Y,
    'Predicted Math Score': Y_pred_all
}).reset_index(drop=True)

# Display predictions inside an expander
with st.expander("See All Predictions (Actual vs Predicted Math Scores)", expanded=False):
    st.write("### Full Predictions Table")
    st.dataframe(results_df)

# Metrics for test data
Y_pred_test = model.predict(X_test)
mae = mean_absolute_error(Y_test, Y_pred_test)
mse = mean_squared_error(Y_test, Y_pred_test)
r2 = r2_score(Y_test, Y_pred_test)

st.write("### Model Performance Metrics")
st.write(f"Mean Absolute Error (MAE): {mae:.2f}")
st.write(f"Mean Squared Error (MSE): {mse:.2f}")
st.write(f"R² Score: {r2:.2f}")

# Input form for user to enter their own values
st.write("### Predict Math Score for New Data")
actual_math_score_input = st.number_input("Enter the actual math score:", min_value=0, max_value=100, value=50)
reading_score_input = st.number_input("Enter the reading score:", min_value=0, max_value=100, value=50)
writing_score_input = st.number_input("Enter the writing score:", min_value=0, max_value=100, value=50)

if st.button("Predict"):
    # Combine user input into a DataFrame for prediction
    user_input = pd.DataFrame({
        'math score': [actual_math_score_input],
        'reading score': [reading_score_input],
        'writing score': [writing_score_input]
    })
    
    # Apply preprocessing to user input
    user_input_transformed = preprocessor.transform(user_input)
    
    # Predict math score for user input
    predicted_math_score = model.predict(user_input_transformed)[0]
    
    # Display both actual and predicted math scores
    st.write(f"Actual Math Score: {actual_math_score_input:.2f}")
    st.write(f"Predicted Math Score: {predicted_math_score:.2f}")
