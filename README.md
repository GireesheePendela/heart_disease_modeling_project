# Heart Disease Modeling Project

This project analyzes the UCI Heart Disease dataset (Cleveland subset) using Python and machine learning techniques. It includes steps like exploratory data analysis (EDA), preprocessing, classification, regression, PCA, and clustering.

---

##  Step 1: Introduction

The aim of this project is to explore, clean, and model the **Cleveland Heart Disease dataset**, a widely used real-world clinical dataset. The dataset is sourced from the UCI Machine Learning Repository and is used to predict the presence or absence of heart disease in patients based on 13 clinical features.

- **Dataset Source:** [UCI Heart Disease Dataset](https://archive.ics.uci.edu/dataset/45/heart+disease)
- **Subset Used:** Cleveland (the only one with complete and reliable data for ML tasks)
- **Target Variable:** `num` (0 = no disease, 1–4 = presence of disease)
- **Goal:** Build predictive and analytical models to understand the risk of heart disease and group patients by profile.

---

##  Step 2: Dataset Reference

We use the `processed.cleveland.data` file, which contains 303 instances and 14 columns (13 features + 1 target).

### Dataset Structure

| Column      | Description                       |
|-------------|-----------------------------------|
| age         | Age in years                      |
| sex         | Sex (1 = male; 0 = female)        |
| cp          | Chest pain type (0–3)             |
| trestbps    | Resting blood pressure (mm Hg)    |
| chol        | Serum cholesterol (mg/dl)         |
| fbs         | Fasting blood sugar > 120 mg/dl   |
| restecg     | Resting ECG results               |
| thalach     | Max heart rate achieved           |
| exang       | Exercise-induced angina (1 = yes) |
| oldpeak     | ST depression induced by exercise |
| slope       | Slope of peak ST segment          |
| ca          | Number of major vessels (0–3)     |
| thal        | Thalassemia (3 = normal; 6,7 = fix)|
| num         | Diagnosis (0 = no disease, 1–4 = disease) |

---

## 🔍 Step 3: EDA & Data Preprocessing

This step focuses on understanding and preparing the dataset for machine learning. It involves exploring the structure of the data, identifying missing values, and applying necessary transformations to clean and standardize the features.

### Exploratory Data Analysis (EDA)

In the EDA phase, we:

- Inspected the first few rows of the dataset.
- Checked the number of rows, columns, and data types.
- Reviewed summary statistics such as mean, median, standard deviation, and quartiles.
- Identified missing values, particularly in the `ca` and `thal` columns.
- Verified class distribution of the target variable `num`.

The purpose of EDA was to gain insights into the data, spot any inconsistencies, and decide how to handle them during preprocessing.


### Data Preprocessing

Based on the findings from EDA, the following actions were taken:

- Replaced missing value placeholders (`?`) with proper `NaN` entries.
- Converted categorical columns like `ca` and `thal` from string to numeric.
- Imputed missing values using the **median** of their respective columns.
- Converted the target column `num` to binary:
  - `0` remained as `0` (no heart disease)
  - Values `1`, `2`, `3`, and `4` were replaced with `1` (presence of heart disease)


### Outcome

- All missing values were successfully handled.
- The target variable was transformed into a binary classification label.
- The dataset is now clean and ready for machine learning modeling tasks in the next steps.

