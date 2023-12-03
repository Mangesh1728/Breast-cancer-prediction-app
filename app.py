# app.py

import streamlit as st
import pandas as pd
from sklearn.impute import SimpleImputer
import joblib

# Load the dataset from a local CSV file
csv_file_path = "/content/Mangesh Shinde - breast-cancer-wisconsin (1).data"  # Adjust the file path accordingly
names = [
    "id",
    "clump_thickness",
    "uniformity_of_cell_size",
    "uniformity_of_cell_shape",
    "marginal_adhesion",
    "single_epithelial_cell_size",
    "bare_nuclei",
    "bland_chromatin",
    "normal_nucleoli",
    "mitoses",
    "class",
]
df = pd.read_csv(csv_file_path, names=names)

# Replace '?' with the mean of the column
for feature in df.columns[1:-1]:
    df[feature] = pd.to_numeric(df[feature], errors='coerce')  # Convert to numeric, replace non-numeric with NaN
    mean_value = df[feature].mean()
    df[feature].fillna(mean_value, inplace=True)

# Load the trained model
model_filename = '/content/final_model_logistic.joblib'  # Adjust the file path accordingly
final_model = joblib.load(model_filename)

# Define a SimpleImputer for handling missing values
imputer = SimpleImputer(strategy='mean')

# Fit the imputer on the training data
imputer.fit(df.iloc[:, 1:-1])  # Exclude the 'id' and 'class' columns

# Define a function to make predictions
def predict_malignancy(features):
    # Create a DataFrame with the input features
    input_data = pd.DataFrame([features], columns=df.columns[1:-1])

    # Impute missing values
    input_data_imputed = imputer.transform(input_data)

    # Make predictions
    prediction = final_model.predict(input_data_imputed)
    return prediction[0]

# Streamlit app
def main():
    st.title("Breast Cancer Prediction App")

    # Create a form for user input
    features = {}
    for feature in df.columns[1:-1]:
        features[feature] = st.slider(f"Enter {feature}", float(df[feature].min()), float(df[feature].max()), float(df[feature].mean()))

    # Make predictions
    if st.button("Predict"):
        result = predict_malignancy(list(features.values()))
        st.success(f"The tumor is predicted as {'Malignant' if result == 4 else 'Benign'}.")

if __name__ == "__main__":
    main()
