import streamlit as st
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeRegressor
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

# Input section for prediction
st.write("### Predict Your Own Math Score")
user_input = {}
for col in cat_features:
    user_input[col] = st.selectbox(f"{col}", df[col].unique())
for col in num_features:
    user_input[col] = st.number_input(f"{col}", min_value=float(df[col].min()), max_value=float(df[col].max()))

# Add a field for actual math score
actual_math_score = st.number_input("Actual Math Score (if available)", min_value=0.0, max_value=100.0, step=1.0)

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

    # Display prediction along with actual math score (if provided)
    if actual_math_score > 0:
        st.write(f"Predicted Math Score: {user_prediction[0]:.2f}")
        st.write(f"Actual Math Score: {actual_math_score:.2f}")
    else:
        st.write(f"Predicted Math Score: {user_prediction[0]:.2f}")

except ValueError as e:
    st.error(f"Error during prediction: {e}")

