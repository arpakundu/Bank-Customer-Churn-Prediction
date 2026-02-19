# 📊 Bank Customer Churn Prediction using Machine Learning

**Machine Learning • Python • Scikit-learn**

This project focuses on predicting whether a bank customer will churn using supervised machine learning techniques. It demonstrates a complete ML workflow including exploratory data analysis, preprocessing, model training, evaluation, and feature importance analysis.


# 🚀 Project Overview

The objective of this project is to build a churn prediction system using classical machine learning algorithms. The workflow includes:

* Loading and understanding the dataset
* Performing exploratory data analysis (EDA)
* Cleaning and preprocessing data
* Encoding categorical variables
* Feature scaling
* Stratified train-test split (80/20)
* Training classification models
* Evaluating model performance
* Interpreting feature importance

The final output helps identify high-risk customers and supports data-driven retention strategies.


# 📂 Dataset Details

The dataset contains **10,000 customer records** with banking attributes including demographics, financial activity, and account behavior.
Dataset Link: https://www.kaggle.com/datasets/shrutimechlearn/churn-modelling

Target Variable:

* `Exited`

  * 1 → Customer churned
  * 0 → Customer retained

Feature Types:

**Numerical Features**

* CreditScore
* Age
* Tenure
* Balance
* NumOfProducts
* EstimatedSalary

**Categorical Features**

* Geography
* Gender
* IsActiveMember
* HasCrCard

No missing values were found in the dataset.


# 🔎 Exploratory Data Analysis (EDA)

EDA included:

* Target distribution visualization
* Geography vs churn comparison
* Gender vs churn comparison
* Age and balance distribution analysis
* Correlation heatmap
* Campaign and activity behavior analysis

Key Insights:

* Older customers have higher churn probability.
* Customers with higher balances churn more frequently.
* Active members are significantly less likely to churn.
* Germany shows comparatively higher churn rates.


# 🔧 Data Preprocessing

* Removed irrelevant columns: RowNumber, CustomerId, Surname
* One-hot encoded categorical features
* Applied StandardScaler for numerical feature normalization
* Used stratified train-test split to preserve churn distribution
* Ensured clean and model-ready dataset


# 🤖 Machine Learning Models

Two models were implemented:

## ✔ Logistic Regression

Baseline linear classifier.

* Accuracy: **80.8%**

## ✔ Random Forest Classifier

Chosen as final model due to superior predictive performance.

* Accuracy: **85.9%**
* ROC-AUC Score: **0.861**


# 📈 Model Evaluation

Confusion Matrix:

```
[[1571   22]
 [ 260  147]]
```

* True Negatives: 1571
* False Positives: 22
* False Negatives: 260
* True Positives: 147

Evaluation Metrics Used:

* Accuracy
* Precision
* Recall
* F1-score
* ROC Curve
* ROC-AUC

The Random Forest model demonstrates strong predictive capability with balanced precision and recall.


# ⭐ Feature Importance Analysis

Random Forest feature importance identified the following top predictors:

* Age
* Balance
* IsActiveMember
* NumOfProducts
* Geography

These variables significantly influence customer churn behavior.


# 💻 How to Run the Project

### Clone the repository

```
git clone https://github.com/yourusername/bank-customer-churn-prediction-ml.git
cd bank-customer-churn-prediction-ml
```

### Create virtual environment

```
python -m venv venv
venv\Scripts\activate
```

### Install dependencies

```
pip install -r requirements.txt
```

### Run notebook

```
jupyter notebook
```

Or open directly in VS Code.


# 📁 Project Structure

```
bank-customer-churn-prediction-ml/
│
├── Churn_Modelling.csv
├── churn_prediction.ipynb
├── requirements.txt
├── README.md
└── venv/
```


# 📝 Results Summary

| Model               | Accuracy |
| ------------------- | -------- |
| Logistic Regression | 80.8%    |
| Random Forest       | 85.9%    |

Random Forest achieved the best balance between predictive performance and interpretability.


# 📌 Conclusion

This project demonstrates a complete machine learning pipeline for churn prediction, from data preprocessing to model evaluation and insight extraction.

It showcases:

* End-to-end ML workflow implementation
* Data cleaning and feature engineering
* Classification modeling using multiple algorithms
* Performance evaluation using ROC-AUC and confusion matrix
* Business-focused insight generation

The model can help financial institutions identify high-risk customers and implement targeted retention strategies.

## 👩‍💻 Author
Arpa Kundu

## 📜 License
MIT License
