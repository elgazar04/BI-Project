# Telecom Customer Churn Prediction

This project demonstrates a complete pipeline for **data preprocessing** and implementing a **Logistic Regression Classifier** to predict customer churn in the telecom industry. It includes data cleaning, feature engineering, model training, evaluation, and visualization of results.

---

## Overview

The goal of this project is to predict whether a customer will churn (i.e., leave the company) based on their attributes and behaviors. The workflow spans from preprocessing raw data to evaluating the performance of the classifier.

---

## Features

### Preprocessing
- **Data Cleaning**:
  - Handled missing values using **KNN Imputer** for numerical and categorical columns.
  - Dropped duplicate rows to ensure dataset consistency.
  - Transformed categorical variables into binary (0/1) representations.
- **Feature Engineering**:
  - Derived new features and standardized numerical ones for uniformity.
  - Converted `TotalCharges` from an object to a numeric type.
- **Class Imbalance Handling**:
  - Applied **Synthetic Minority Over-sampling Technique (SMOTE)** to balance the dataset.

### Logistic Regression Model
- **Model Training**: Logistic regression is implemented as the primary algorithm for classification tasks.
- **Evaluation Metrics**:
  - **Accuracy**: Overall correctness of the classifier.
  - **Precision**: Fraction of relevant instances among retrieved instances.
  - **Recall**: Ability of the model to identify all relevant instances.
  - **F1-score**: Harmonic mean of precision and recall.
- **Hyperparameter Tuning**: Adjusted parameters like regularization strength to optimize performance.

### Visualizations
- **Data Insights**: Plots for understanding feature distributions, correlations, and class imbalances.
- **Evaluation Insights**: Confusion matrix, precision-recall curve, and ROC curve to visualize model performance.

---

## Dataset Description

### Initial Data Overview
- **Number of Entries**: 7,043
- **Number of Features**: 19
- **Target Variable**: `Churn` (binary: Yes/No)
- **Challenges**:
  - Missing values in `SeniorCitizen` (20 missing values) and `TotalCharges` (10 missing values).
  - `TotalCharges` stored as an object instead of a numerical type.
  - Imbalanced class distribution in `Churn`.

### Data Types Before Transformation
```plaintext
gender                object
SeniorCitizen         float64
Partner               object
Dependents            object
tenure                int64
MultipleLines         object
InternetService       object
OnlineSecurity        object
OnlineBackup          object
DeviceProtection      object
TechSupport           object
StreamingTV           object
StreamingMovies       object
Contract              object
PaperlessBilling      object
PaymentMethod         object
MonthlyCharges        float64
TotalCharges          object
Churn                 object
```

### Data Types After Transformation
```plaintext
gender                int64
SeniorCitizen         int64
Partner               int64
Dependents            int64
tenure                int64
MultipleLines         object
InternetService       object
OnlineSecurity        int64
OnlineBackup          int64
DeviceProtection      int64
TechSupport           int64
StreamingTV           int64
StreamingMovies       int64
Contract              object
PaperlessBilling      int64
PaymentMethod         object
MonthlyCharges       float64
TotalCharges         float64
Churn                 int64
```

---

## Model Evaluation

### Confusion Matrix

| Actual \ Predicted |    0    |    1    |
|---------------------|---------|---------|
| **0**              |  1119   |   433   |
| **1**              |   101   |  1446   |

- **True Negatives (0,0): 1119**
- **False Positives (0,1): 433**
- **False Negatives (1,0): 101**
- **True Positives (1,1): 1446**

### Classification Report
```plaintext
              precision    recall  f1-score   support

           0       0.92      0.72      0.81      1552
           1       0.77      0.93      0.84      1547

    accuracy                           0.83      3099
   macro avg       0.84      0.83      0.83      3099
weighted avg       0.84      0.83      0.83      3099
```

- **Weighted F1-Score**: 0.83
- **Accuracy**: 0.83

### Key Insights
- The model achieves a good balance between precision and recall, with an overall F1-score of 0.83.
- The ROC curve and confusion matrix provide evidence of robust predictive performance.

---

## Project Workflow

### 1. Data Import
- Load the dataset from a CSV file or database into a pandas DataFrame.
- Display initial insights such as shape, column types, and missing values.

### 2. Data Preprocessing
- **Data Cleaning**:
  - Imputed missing values in `SeniorCitizen` and `TotalCharges` using KNN Imputer.
  - Converted categorical variables like `gender` and `Partner` to binary format.
  - Ensured all features were in the correct data type.
- **Class Balance**:
  - Analyzed the class distribution for `Churn`.
  - Applied SMOTE to generate synthetic samples for the minority class.

### 3. Exploratory Data Analysis (EDA)
- Visualized data distributions, feature correlations, and potential outliers.
- Explored relationships between predictors and the target variable.

### 4. Model Training
- Split the data into training and testing sets.
- Trained a logistic regression model on the training set.
- Tuned hyperparameters to improve predictive performance.

### 5. Model Evaluation
- Evaluated performance on the testing set using metrics:
  - **Accuracy**: 0.83
  - **Weighted F1-Score**: 0.83
---

## Acknowledgments

- Mohamed seif elnasr
- Libraries and frameworks used:
  - pandas
  - numpy
  - matplotlib
  - seaborn
  - scikit-learn
  - nltk
  - joblib
