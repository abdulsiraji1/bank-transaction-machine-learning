# Bank Transaction Machine Learning Project

## Project Overview
This project implements a machine learning pipeline that combines unsupervised learning and supervised learning for bank transaction data analysis.

The first stage uses clustering to generate labels from an unlabeled dataset. The generated cluster labels are then used in the second stage to train a classification model that predicts the cluster class based on transaction features.

This project is part of the Machine Learning submission from Dicoding.

---

## Dataset
The dataset used in this project is a modified version of the Bank Transaction Dataset for Fraud Detection.

The dataset contains various transaction features such as:

- Transaction amount
- Customer information
- Device information
- Merchant category
- Transaction time

The dataset is preprocessed before applying machine learning models.

---

## Project Workflow

### 1. Exploratory Data Analysis (EDA)
Several exploratory steps were performed:

- Display dataset using `head()`
- Inspect dataset structure using `info()`
- Analyze statistical distribution using `describe()`
- Correlation analysis
- Histogram visualization

---

### 2. Data Preprocessing

Data preprocessing includes:

- Handling missing values using `dropna()`
- Removing duplicate records using `drop_duplicates()`
- Dropping ID related columns:
  - TransactionID
  - AccountID
  - DeviceID
  - IPAddress
  - MerchantID
  - TransactionDate
- Encoding categorical features using `LabelEncoder`
- Feature scaling using `StandardScaler`
- Outlier removal
- Feature binning for selected numerical features

---

### 3. Clustering Model

Unsupervised learning is applied using **K-Means Clustering**.

Steps performed:

- Determining optimal number of clusters using **Elbow Method**
- Training clustering model using `KMeans`
- Evaluating clustering performance using **Silhouette Score**
- Saving the model using `joblib`

Saved models:

- `model_clustering.h5`
- `PCA_model_clustering.h5`

---

### 4. Cluster Interpretation

Cluster results are analyzed by computing:

- Mean
- Minimum
- Maximum

Each cluster is interpreted based on transaction characteristics.

The dataset is then exported with a new column:
