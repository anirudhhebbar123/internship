import streamlit as st
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# Function for evaluating the model
def evaluate_model(true, predicted):
    mae = mean_absolute_error(true, predicted)
    mse = mean_squared_error(true, predicted)
    rmse = np.sqrt(mean_squared_error(true, predicted))
    r2_square = r2_score(true, predicted)
    return mae, rmse, r2_square

# Streamlit app
st.title("Math Score Prediction App")

# Upload dataset
uploaded_file = st.file_uploader("Upload your dataset (CSV format)", type="csv")

if uploaded_file is not None:
    # Load dataset
    df = pd.read_csv(uploaded_file)
    st.write("### Dataset Preview:")
    st.dataframe(df.head())

    # Select features and target
    st.write("### Select Features and Target")
    features = st.multiselect("Select feature columns:", options=df.columns)
    target = st.selectbox("Select target column:", options=df.columns)

    if features and target:
        X = df[features]
        Y = df[target]

        # Train-test split
        X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=42)

        # Train model
        lin_model = LinearRegression()
        lin_model.fit(X_train, Y_train)

        # Predict on test set
        Y_pred = lin_model.predict(X_test)

        # Evaluate the model
        mae, rmse, r2 = evaluate_model(Y_test, Y_pred)

        st.write("### Model Evaluation")
        st.write(f"- Mean Absolute Error (MAE): {mae:.2f}")
        st.write(f"- Root Mean Squared Error (RMSE): {rmse:.2f}")
        st.write(f"- R2 Score: {r2:.2f}")

        # Option to predict for new input
        st.write("### Predict for New Input")
        input_data = {}
        for feature in features:
            input_value = st.number_input(f"{feature}", value=0.0)
            input_data[feature] = input_value

        if st.button("Predict Math Score"):
            input_df = pd.DataFrame([input_data])
            prediction = lin_model.predict(input_df)
            st.write(f"### Predicted Math Score: {prediction[0]:.2f}")

