import streamlit as st
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# Streamlit app
st.title("Math Score Prediction App")

# Upload dataset
uploaded_file = st.file_uploader("Upload your dataset (CSV format)", type="csv")

if uploaded_file is not None:
    # Load dataset
    df = pd.read_csv(uploaded_file)
    st.write("### Dataset Preview:")
    st.dataframe(df.head())

    # Preprocessing (adjust as per the notebook logic)
    st.write("### Preprocessing and Feature Selection")
    features = ['reading_score', 'writing_score']  # Example features from the notebook
    target = 'math_score'  # Example target column from the notebook

    if all(col in df.columns for col in features + [target]):
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
        mae = mean_absolute_error(Y_test, Y_pred)
        rmse = np.sqrt(mean_squared_error(Y_test, Y_pred))
        r2 = r2_score(Y_test, Y_pred)

        st.write("### Model Evaluation")
        st.write(f"- Mean Absolute Error (MAE): {mae:.2f}")
        st.write(f"- Root Mean Squared Error (RMSE): {rmse:.2f}")
        st.write(f"- R2 Score: {r2:.2f}")

        # Option to predict for new input
        st.write("### Predict for New Input")
        input_data = {}
        for feature in features:
            input_value = st.number_input(f"Enter {feature}", value=0.0)
            input_data[feature] = input_value

        if st.button("Predict Math Score"):
            input_df = pd.DataFrame([input_data])
            prediction = lin_model.predict(input_df)
            st.write(f"### Predicted Math Score: {prediction[0]:.2f}")
    else:
        st.write("### Error: Required columns not found in the dataset.")
        st.write(f"Expected features: {features + [target]}")


