# Cancer-Survival-Prediction

This repository contains machine learning models for predicting the **survival months** and **mortality status** of patients with cancer based on various features. The models utilize different algorithms, including classification models (Logistic Regression, K-Nearest Neighbors, Naive Bayes) for mortality status prediction and regression models (Decision Tree, Random Forest, Gradient Boosting) for predicting the survival months.

## Project Structure

The project is organized into three main notebooks:

### 1. Data Preprocessing and Feature Engineering (`1_data_preprocessing.ipynb`):
- This notebook covers data loading, feature selection, handling missing values, and encoding categorical variables.
- The output dataset is prepared for both classification and regression tasks.

### 2. Mortality Status Classification (`2_mortality_classification.ipynb`):
- This notebook uses various **classification models**:
  - **Logistic Regression** for predicting if the patient is alive or dead.
  - **K-Nearest Neighbors (KNN)** for classifying mortality status based on feature proximity.
  - **Naive Bayes** for mortality status prediction assuming feature independence.
- Performance metrics: Accuracy, Precision, Recall, F1-Score, ROC-AUC.

### 3. Survival Months Prediction (`3_survival_regression.ipynb`):
- This notebook focuses on **regression models** to predict the number of survival months:
  - **Decision Tree Regressor** for regression.
- Model evaluation using RMSE, MSE, MAE, and R² score.

## Datasets

- **`cancer-dataset.csv`**: The original dataset containing patient information, including demographics, tumor details, and survival status.
- **`classification_dataset.csv`**: A preprocessed version of the dataset used for the classification task of mortality status prediction.
- **`regression_dataset.csv`**: A preprocessed version of the dataset used for the regression task of survival month prediction.

## Models

- **Classification models**: These models predict whether a patient survives (Alive/Dead) based on the features in the dataset.
  - `knn_model.pkl` (Logistic Regression)
  - `mortality_knn_model.pkl` (K-Nearest Neighbors)
  - `naive_bayes_model.pkl` (Naive Bayes)
  - `ensemble_classifier_model.pkl` (Ensemble Classifier)

- **Regression models**: These models predict the number of months a patient will survive.
  - `best_dt_regression_model.pkl` (Decision Tree Regressor)
