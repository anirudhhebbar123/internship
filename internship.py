import streamlit as st
import pandas as pd
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
columns_to_drop = ['total score', 'avg score', 'math score']

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

# Transform features
try:
    X_transformed = preprocessor.fit_transform(X)
except ValueError as e:
    st.error(f"Error during feature transformation: {e}")
    st.stop()

# Train-test split
X_train, X_test, Y_train, Y_test = train_test_split(X_transformed, Y, test_size=0.2, random_state=42)

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

# Input section for prediction
st.write("### Predict Your Own Math Score")
user_input = {}
for col in cat_features:
    user_input[col] = st.selectbox(f"{col}", df[col].unique())
for col in num_features:
    user_input[col] = st.number_input(f"{col}", min_value=float(df[col].min()), max_value=float(df[col].max()))

# Create user input DataFrame
user_input_df = pd.DataFrame([user_input])

# Align user input DataFrame with the expected feature columns
missing_cols = set(X.columns) - set(user_input_df.columns)
for col in missing_cols:
    user_input_df[col] = 0  # Fill missing columns with default values

user_input_df = user_input_df[X.columns]  # Reorder columns to match training data

# Transform user input and predict
try:
    user_input_transformed = preprocessor.transform(user_input_df)
    user_prediction = model.predict(user_input_transformed)
    st.write(f"Predicted Math Score: {user_prediction[0]:.2f}")
except ValueError as e:
    st.error(f"Error during prediction: {e}")


