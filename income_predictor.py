import streamlit as st
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import LabelEncoder

# Title for the Streamlit App
st.title("Income Predictor")
st.write("This app predicts whether a person's income is greater than $50K based on their details. Disclaimer: This is a speculative tool and may not reflect real-world outcomes.")

# Load the dataset
st.header("Dataset and Preprocessing")
file_path = 'adult.xlsx'  # Replace with your actual file path
try:
    data = pd.read_excel(file_path, header=1)
    st.write("Dataset loaded successfully!")
except FileNotFoundError:
    st.error("The dataset file was not found. Please ensure 'adult.xlsx' is in the same directory as this script.")
    st.stop()

# Preprocessing the data
data['income'] = data['income'].apply(lambda x: 1 if x.strip() == '>50K' else 0)
data = data.dropna()  # Drop missing values

# Encode categorical features
categorical_columns = data.select_dtypes(include=['object']).columns
encoders = {}  # Dictionary to store the encoders for each categorical column
for col in categorical_columns:
    le = LabelEncoder()
    data[col] = le.fit_transform(data[col].astype(str))
    encoders[col] = le  # Save the encoder for later use

# Split data into features and target
X = data.drop(columns=['income'])
y = data['income']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train the model
model = LinearRegression()
model.fit(X_train, y_train)

# Dropdown for user input
st.header("Make a Prediction")
st.write("Provide your details below:")

# Display options for each feature
user_data = {}
for col in X.columns:
    if col in categorical_columns:
        # Get the original categories from the encoder
        options = encoders[col].classes_
        user_data[col] = st.selectbox(f"{col} (choose an option)", options)
    else:
        user_data[col] = st.number_input(f"{col} (enter a value)", min_value=float(data[col].min()), max_value=float(data[col].max()), step=1.0)

# Re-encode the user input before passing it to the model
encoded_user_data = {}
for col in user_data:
    if col in categorical_columns:
        # Encode the user input using the stored encoder
        encoded_user_data[col] = encoders[col].transform([user_data[col]])[0]
    else:
        encoded_user_data[col] = user_data[col]

# Convert the encoded user input into a DataFrame
user_input_df = pd.DataFrame([encoded_user_data])

# Show user input
st.subheader("Your Input Data:")
st.write(user_input_df)

# Make prediction
if st.button("Predict Income"):
    prediction = model.predict(user_input_df)
    predicted_income = ">50K" if prediction[0] >= 0.5 else "<=50K"
    st.subheader("Prediction Result:")
    st.write(f"Predicted Income: {predicted_income}")
