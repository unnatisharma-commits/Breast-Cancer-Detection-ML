import streamlit as st
import numpy as np
import json

# Page configuration
st.set_page_config(
    page_title="Breast Cancer Detection",
    page_icon="🎗️",
    layout="wide"
)

# Load model parameters from JSON
with open("model_parameters.json", "r") as file:
    model_info = json.load(file)

coef = np.array(model_info["coef"])
intercept = np.array(model_info["intercept"])
classes = np.array(model_info["classes"])
mean = np.array(model_info["mean"])
scale = np.array(model_info["scale"])

# Default sample values
sample_data = mean
# Title
st.title("🎗️ Breast Cancer Detection Using Machine Learning")

# Introduction
st.write(
    "This application uses a Machine Learning model "
    "to classify breast cancer cases as Malignant or Benign."
)

st.info(
    "This application is for educational purposes only "
    "and should not be used as a medical diagnosis."
)

# Input section
st.header("Enter Patient Measurements")

st.subheader("Mean Features")

col1, col2, col3 = st.columns(3)

with col1:
    mean_radius = st.number_input(
        "Mean Radius",
        min_value=0.0,
        value=float(sample_data[0])
    )

with col2:
    mean_texture = st.number_input(
        "Mean Texture",
        min_value=0.0,
        value=float(sample_data[1])
    )

with col3:
    mean_perimeter = st.number_input(
        "Mean Perimeter",
        min_value=0.0,
        value=float(sample_data[2])
    )


col1, col2, col3 = st.columns(3)

with col1:
    mean_area = st.number_input(
        "Mean Area",
        min_value=0.0,
        value=float(sample_data[3])
    )

with col2:
    mean_smoothness = st.number_input(
        "Mean Smoothness",
        min_value=0.0,
        value=float(sample_data[4])
    )

with col3:
    mean_compactness = st.number_input(
        "Mean Compactness",
        min_value=0.0,
        value=float(sample_data[5])
    )


col1, col2, col3 = st.columns(3)

with col1:
    mean_concavity = st.number_input(
        "Mean Concavity",
        min_value=0.0,
        value=float(sample_data[6])
    )

with col2:
    mean_concave_points = st.number_input(
        "Mean Concave Points",
        min_value=0.0,
        value=float(sample_data[7])
    )

with col3:
    mean_symmetry = st.number_input(
        "Mean Symmetry",
        min_value=0.0,
        value=float(sample_data[8])
    )


col1, col2, col3 = st.columns(3)

with col1:
    mean_fractal_dimension = st.number_input(
        "Mean Fractal Dimension",
        min_value=0.0,
        value=float(sample_data[9])
    )


# Standard Error Features
st.subheader("Standard Error Features")

col1, col2, col3 = st.columns(3)

with col1:
    radius_error = st.number_input(
        "Radius Error",
        min_value=0.0,
        value=float(sample_data[10])
    )

with col2:
    texture_error = st.number_input(
        "Texture Error",
        min_value=0.0,
        value=float(sample_data[11])
    )

with col3:
    perimeter_error = st.number_input(
        "Perimeter Error",
        min_value=0.0,
        value=float(sample_data[12])
    )


col1, col2, col3 = st.columns(3)

with col1:
    area_error = st.number_input(
        "Area Error",
        min_value=0.0,
        value=float(sample_data[13])
    )

with col2:
    smoothness_error = st.number_input(
        "Smoothness Error",
        min_value=0.0,
        value=float(sample_data[14])
    )

with col3:
    compactness_error = st.number_input(
        "Compactness Error",
        min_value=0.0,
        value=float(sample_data[15])
    )


col1, col2, col3 = st.columns(3)

with col1:
    concavity_error = st.number_input(
        "Concavity Error",
        min_value=0.0,
        value=float(sample_data[16])
    )

with col2:
    concave_points_error = st.number_input(
        "Concave Points Error",
        min_value=0.0,
        value=float(sample_data[17])
    )

with col3:
    symmetry_error = st.number_input(
        "Symmetry Error",
        min_value=0.0,
        value=float(sample_data[18])
    )


col1, col2, col3 = st.columns(3)

with col1:
    fractal_dimension_error = st.number_input(
        "Fractal Dimension Error",
        min_value=0.0,
        value=float(sample_data[19])
    )

# Worst Features
st.subheader("Worst Features")

col1, col2, col3 = st.columns(3)

with col1:
    worst_radius = st.number_input(
        "Worst Radius",
        min_value=0.0,
        value=float(sample_data[20])
    )

with col2:
    worst_texture = st.number_input(
        "Worst Texture",
        min_value=0.0,
        value=float(sample_data[21])
    )

with col3:
    worst_perimeter = st.number_input(
        "Worst Perimeter",
        min_value=0.0,
        value=float(sample_data[22])
    )


col1, col2, col3 = st.columns(3)

with col1:
    worst_area = st.number_input(
        "Worst Area",
        min_value=0.0,
        value=float(sample_data[23])
    )

with col2:
    worst_smoothness = st.number_input(
        "Worst Smoothness",
        min_value=0.0,
        value=float(sample_data[24])
    )

with col3:
    worst_compactness = st.number_input(
        "Worst Compactness",
        min_value=0.0,
        value=float(sample_data[25])
    )


col1, col2, col3 = st.columns(3)

with col1:
    worst_concavity = st.number_input(
        "Worst Concavity",
        min_value=0.0,
        value=float(sample_data[26])
    )

with col2:
    worst_concave_points = st.number_input(
        "Worst Concave Points",
        min_value=0.0,
        value=float(sample_data[27])
    )

with col3:
    worst_symmetry = st.number_input(
        "Worst Symmetry",
        min_value=0.0,
        value=float(sample_data[28])
    )


col1, col2, col3 = st.columns(3)

with col1:
    worst_fractal_dimension = st.number_input(
        "Worst Fractal Dimension",
        min_value=0.0,
        value=float(sample_data[29])
    )
# Sample case button
if st.button("📋 Load Sample Case"):
    st.session_state["sample_loaded"] = True
    st.session_state["sample_data"] = sample_data
    st.rerun()

# Prediction button
if st.button("🔍 Predict Diagnosis"):

    input_data = [
        mean_radius,
        mean_texture,
        mean_perimeter,
        mean_area,
        mean_smoothness,
        mean_compactness,
        mean_concavity,
        mean_concave_points,
        mean_symmetry,
        mean_fractal_dimension,
        radius_error,
        texture_error,
        perimeter_error,
        area_error,
        smoothness_error,
        compactness_error,
        concavity_error,
        concave_points_error,
        symmetry_error,
        fractal_dimension_error,
        worst_radius,
        worst_texture,
        worst_perimeter,
        worst_area,
        worst_smoothness,
        worst_compactness,
        worst_concavity,
        worst_concave_points,
        worst_symmetry,
        worst_fractal_dimension
    ]

       # Scale input using saved scaler values
    input_array = np.array(input_data, dtype=float)
    input_scaled = (input_array - mean) / scale

    # Calculate prediction using Logistic Regression
    z = float(np.dot(coef[0], input_scaled) + intercept[0])
    z = np.clip(z, -500, 500)

    prob_class_1 = 1 / (1 + np.exp(-z))

    probabilities = np.array([
        1 - prob_class_1,
        prob_class_1
    ])

    prediction = classes[np.argmax(probabilities)]

    if prediction == 0:
        st.error("🔴 Prediction: Malignant")
    else:
        st.success("🟢 Prediction: Benign")

    st.subheader("Prediction Probabilities")

    col1, col2 = st.columns(2)

    with col1:
        st.metric(
            "Malignant Probability",
            f"{probabilities[0] * 100:.2f}%"
        )

    with col2:
        st.metric(
            "Benign Probability",
            f"{probabilities[1] * 100:.2f}%"
        )