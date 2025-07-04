# Heart Disease Modeling Project

This project analyzes the UCI Heart Disease dataset (Cleveland subset) using Python and machine learning techniques. It includes steps like exploratory data analysis (EDA), preprocessing, classification, regression, PCA, and clustering.

---

## Introduction

The aim of this project is to explore, clean, and model the **Cleveland Heart Disease dataset**, a widely used real-world clinical dataset. The dataset is sourced from the UCI Machine Learning Repository and is used to predict the presence or absence of heart disease in patients based on 13 clinical features.

- **Dataset Source:** [UCI Heart Disease Dataset](https://archive.ics.uci.edu/dataset/45/heart+disease)
- **Subset Used:** Cleveland (the only one with complete and reliable data for ML tasks)
- **Target Variable:** `num` (0 = no disease, 1–4 = presence of disease)
- **Goal:** Build predictive and analytical models to understand the risk of heart disease and group patients by profile.

---

##  Step 1: Dataset Reference

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

##  Step 2: EDA & Data Preprocessing

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
- **Normalized all feature columns** using `StandardScaler` to scale them to a standard range (mean = 0, std = 1), ensuring models aren't biased by differing feature scales.

###  Outcome

- All missing values were successfully handled.
- The target variable was transformed into a binary classification label.
- All features were standardized for optimal model performance.
- The dataset is now clean and ready for machine learning modeling tasks in the next steps.

---

##  Step 3.1: Heart Disease Prediction

In this step, we built two supervised machine learning models — **Logistic Regression** and **Random Forest Classifier** — to predict the presence of heart disease.

After splitting the dataset into training and test sets, both models were trained and evaluated using standard classification metrics including **Accuracy**, **Precision**, **Recall**, **F1-Score**, and **Confusion Matrix**.

**Logistic Regression** achieved slightly better balanced performance across all metrics, while **Random Forest** showed strong precision.

This step demonstrated the practical use of ML models in clinical risk prediction.

---

##  Step 3.2: Cholesterol Level Prediction

In this task, we built a **Multiple Linear Regression** model to predict the serum cholesterol level (`chol`) based on the remaining 12 clinical features (excluding the target label `num` and the `chol` column itself).

###  Methodology

- The `chol` column was used as the **regression target**.
- All other features were **normalized** using `StandardScaler` prior to model training.
- The dataset was split into **training and test sets** (80/20 split).
- A **Linear Regression model** was trained using scikit-learn.
- A **correlation matrix** was computed and visualized with a heatmap to identify key predictors of cholesterol levels.

### Results

- The model provided a basic estimation of cholesterol levels based on the available features.
- Evaluation metrics included **Mean Squared Error (MSE)** and **R² Score**.
- A correlation heatmap revealed the **most strongly correlated features** with cholesterol.

###  Key Findings

- The most positively or negatively correlated features with serum cholesterol (`chol`) were identified and can be used to understand **which health metrics most influence cholesterol levels**.
- These insights may aid in targeted health interventions or further clinical analysis.

---

## Step 3.3: Principal Component Analysis (PCA)

The goal of this step was to reduce the dataset’s dimensionality while retaining as much variance as possible, in preparation for unsupervised learning tasks like clustering.

### Methodology

- The target column (`num`) was excluded from the analysis.
- PCA was applied to the normalized feature set using `scikit-learn`.
- The number of components was chosen such that **95% of the variance** in the dataset was retained.
- A plot of the **cumulative explained variance** was generated to visualize how much information is preserved as dimensions are reduced.

### Results

- The original dataset had **13 features**.
- After PCA, the dimensionality was reduced to **X components** (shown in the output).
- The resulting dataset still retains **95% of the variance**, making it suitable for downstream tasks like clustering while reducing computational complexity.

---

##  Step 3.4: Grouping Patients Based on Health Profiles (Clustering)

In this step, we applied **K-Means Clustering** on the PCA-reduced dataset to group patients based on their health profiles.

### Methodology

- The PCA-reduced dataset was used as input for clustering (unsupervised).
- The **Elbow Method** was used to evaluate the optimal number of clusters by plotting inertia scores.
- The **Silhouette Score** was also calculated for `k = 2 to 10` to assess clustering quality.
- Once the optimal number of clusters was selected (e.g., `k = 3`), final clustering was performed.
- A 2D scatterplot was created using the first two principal components to visualize patient groups.

### Insights

- Patients were successfully grouped into **k clusters** based on shared patterns in their clinical data.
- These clusters may represent different risk profiles or patient subtypes.
- This grouping could support personalized treatment strategies or health monitoring systems.
