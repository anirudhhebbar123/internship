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
df = pd.read_csv(file_path)
st.write("### Dataset Preview:")
st.dataframe(df.head(10))

# Preprocessing
st.write("### Preprocessing and Feature Transformation")
X = df.drop(columns=['total score', 'avg score', 'math score'], axis=1)
Y = df['math score']

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

X = preprocessor.fit_transform(X)
st.write(f"Transformed Feature Shape: {X.shape}")

# Train-test split
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=42)

# Train model
model = DecisionTreeRegressor()
model.fit(X_train, Y_train)

# Predict on test set
Y_pred = model.predict(X_test)

# Display only the prediction of math scores
pred_df = pd.DataFrame({'Predicted Math Score': Y_pred}).reset_index(drop=True)

st.write("### Predictions of Math Scores")
st.dataframe(pred_df.head(10))




