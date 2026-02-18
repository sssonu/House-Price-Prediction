### PROJECT OVERVIEW
This project focuses on building an end-to-end machine learning pipeline to predict house prices using structured tabular data. The project uses the California Housing dataset as a proxy dataset to simulate a real-world house price prediction problem, such as predicting property prices in a city like Gurgaon.

The objective is to demonstrate the complete machine learning workflow, including data preprocessing, exploratory data analysis, model training, evaluation, and model persistence using industry-standard tools.

### PROBLEM DESCRIPTION
House prices depend on multiple factors such as location, income levels, population density, number of rooms, and proximity to important areas. Predicting house prices manually is difficult due to the complex relationships between these variables.

This project formulates the task as a supervised regression problem, where the goal is to predict a continuous target variable (house price) based on multiple input features.

### DATASET DETAILS
The project uses the California Housing dataset, which contains historical housing data with the following characteristics:

Numerical features such as median income, number of rooms, population, and geographical coordinates

A categorical feature representing proximity to the ocean

Target variable: median house value

Although the dataset represents California housing data, it is treated as a simulated dataset for cities like Gurgaon to demonstrate real-world applicability.

### MACHINE LEARNING APPROACH:-

**1. Type of Problem**
- Supervised Learning
- Regression Problem

**2. Target Variable**
- Median house value

**3.Input Features**

- Median income
- Number of rooms
- Housing age
- Population
- Latitude and longitude
- Ocean proximity (categorical)

**3. DATA PREPROCESSING**
The following preprocessing steps are applied:
- Stratified train-test split based on income categories to avoid sampling bias
- Handling missing values using median imputation
- Feature scaling using standardization
- Encoding categorical features using one-hot encoding
- Combining preprocessing steps using Scikit-learn Pipelines and ColumnTransformer

This ensures reproducibility and consistent preprocessing during training and inference.

**4. MODELS IMPLEMENTED**
The following regression models are trained and evaluated:

- Linear Regression
- Decision Tree Regressor
- Random Forest Regressor

Among these, the Random Forest Regressor provides the best performance in terms of error metrics and generalization.

**5. MODEL EVALUATION**
Model performance is evaluated using the following metrics:

- Mean Absolute Error (MAE)
- Root Mean Squared Error (RMSE)
- Cross-validation RMSE
- Cross-validation is used to obtain a reliable estimate of model performance and avoid overfitting.

### PIPELINE AND MODEL PERSISTENCE
The project uses Scikit-learn Pipelines to combine preprocessing and modeling steps.
**Trained models and preprocessing pipelines are saved using Joblib, allowing:**
- Avoidance of repeated training
- Fast inference on new data
- Reproducible and production-ready workflows
- An if-else logic is implemented to train the model only if a saved model does not already exist.

### INFERENCE WORKFLOW

- Load saved preprocessing pipeline and trained model
- Read new input data from a CSV file
- Apply identical preprocessing
- Generate price predictions
- Save predictions to an output CSV file

### TECHNOLOGY STACK
**Programming Language**
- Python

**Libraries and Frameworks**
- Scikit-learn
- Pandas
- NumPy
- Matplotlib
- Model Persistence
- Joblib

### PROJECT OUTCOMES

Demonstrated a complete machine learning workflow from raw data to predictions
Built a reusable and modular preprocessing pipeline
Achieved reliable performance using ensemble learning
Created a production-ready inference setup

### CONCLUSION
This dataset-based project demonstrates how machine learning can be applied to real-world regression problems using structured data. By following best practices such as stratified sampling, pipelines, cross-validation, and model persistence, the project ensures accuracy, reliability, and scalability.

The same approach can be easily adapted to real housing datasets from cities like Gurgaon when available.
