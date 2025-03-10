# 📊 **Bank Customer Churn Prediction**

### 🔍 **Predicting Customer Retention Using Machine Learning**

## **Overview**
This project is a **Data Science portfolio** piece focused on **predicting customer churn** using **machine learning**. The objective is to develop a scalable and interpretable model to predict customer churn, providing actionable insights to help banks retain customers. The primary focus is on maximizing **recall** over **accuracy**, as correctly identifying customers at risk of leaving is more critical than overall prediction accuracy.

This project also compares various **machine learning models** including **Random Forest**, **XGBoost**, **Logistic Regression**, **Support Vector Machines (SVM)**, and **Multi-Layer Perceptron (MLP)** to evaluate different approaches in churn prediction.

---

## 📂 **Project Structure**
This project is divided into **two main parts**:

### **Part I: Data Exploration & Preprocessing**
1. **Project Overview** → Context and objectives of the churn prediction project.
2. **Dataset Overview** → How the data was acquired, the structure of the data, and initial insights.
3. **Exploratory Data Analysis (EDA)** → Visualizing and understanding the distribution of the features and the churn target variable.
4. **Data Preprocessing** → Handling missing values, encoding categorical variables, and normalizing numerical features.

### **Part II: Modeling & Evaluation**
5. **Feature Engineering** → Creating new features to capture customer behavior and financial information.
6. **Modeling** → Implementation of machine learning models (Random Forest, XGBoost, SVM, etc.).
7. **Model Evaluation** → Evaluating model performance with accuracy, recall, precision, and F1-score.
8. **Conclusion** → Comparing models and insights for further improvements.

---

## ⚖ **Model Comparison: Research vs. Other Approaches**

**Research Model**  
**Stratified Sampling + Feature Selection + Cross-Validation**  
**Recall**: ~78% (Best Model for Detecting Churn)  
**Accuracy**: ~80%  
**Lower False Negatives (~15%)**

1️⃣ **Best Accuracy Across Models**
The table below highlights the best-performing model in terms of accuracy for each dataset:

| **Model**                     | **Best Algorithm**  | **Accuracy**  |
|--------------------------------|---------------------|---------------|
| **Model 1 (df_balanced)**      | Random Forests      | 0.769764      |
| **Model 2 (df_all_features_balanced)** | Random Forests      | 0.768542      |
| **Model 3 (df_all_features_SMOTE)**  | XGBoost            | **0.853222**  |
| **Model 4 (df_original)**     | Random Forests      | **0.863125**  |

 **Key Insights:**
- **Models 3 & 4 (SMOTE and Original) show the highest accuracy (~85%+).**
- **However, accuracy alone is misleading** in churn prediction, as it does not reflect the model’s ability to detect actual churned customers.

2️⃣ **Best Recall Across Models**
Since our goal is **customer churn prediction**, recall is the most critical metric. The table below highlights the best-performing model in terms of recall for each dataset:

| **Model**                     | **Best Algorithm**  | **Recall**  |
|--------------------------------|---------------------|------------|
| **Model 1 (df_balanced)**     | XGBoost             | **0.782095** |
| **Model 2 (df_all_features_balanced)** | XGBoost             | **0.782221** |
| **Model 3 (df_all_features_SMOTE)**  | SVM                 | 0.618600    |
| **Model 4 (df_original)**     | Random Forests      | 0.460736    |

**Key Insights:**
- **Models 1 & 2 (Balanced Datasets) perform best in recall (~78%).**
- **Model 3 (SMOTE) performs worse (only 61.8%).**
- **Model 4 (Original, Imbalanced) performs the worst, detecting only ~46% of churned customers.**

**Why Recall Matters More Than Accuracy**
Since our goal is **to identify customers who are most likely to churn**, recall is prioritized over accuracy. A model with high accuracy may still miss a large number of churned customers, which is not ideal for proactive retention strategies.

---

## 🤖 **Machine Learning Models Used**
The following **classification algorithms** were implemented and evaluated:

- **Random Forest**
- **XGBoost**
- **Logistic Regression**
- **Support Vector Machines (SVM)**
- **Naive Bayes**
- **K-Nearest Neighbors (KNN)**
- **Decision Tree (CART)**
- **Multilayer Perceptron (MLP)**

---

## 📉 **Evaluation Metrics**
The models were evaluated using the following key **performance metrics**:

- **Accuracy**: Measures the overall correctness of the model.
- **Precision**: Proportion of correctly predicted churn cases.
- **Recall (Sensitivity)**: Proportion of actual churn cases correctly identified.
- **F1-Score**: Harmonic mean of Precision and Recall.
- **AUC-ROC**: Measures the model's ability to distinguish between churned and non-churned customers.
- **Type I Error (False Negative Rate)**: Percentage of churned customers misclassified as non-churned.
- **Type II Error (False Positive Rate)**: Percentage of non-churned customers misclassified as churned.

---

## 🛠 **Technologies & Tools Used**
- **Programming Language**: Python (Pandas, NumPy, Scikit-learn, XGBoost)
- **Visualization**: Matplotlib, Seaborn
- **Machine Learning**: RandomizedSearchCV, GridSearchCV, K-Fold Cross-Validation
- **Data Processing**: SMOTE, Standardization, One-Hot Encoding, Feature Engineering

---

## 📊 **Dataset Information**
- **Source**: Bank Customer Churn Dataset - Kaggle
- **Number of Samples**: 10,000
- **Features**: 10 demographic and financial features (age, gender, credit score, etc.)
- **Target Variable**: Churn (1 = Churned, 0 = Non-Churned)

---

## 📅 **Conclusion**
In **customer churn prediction**, **minimizing False Negatives (FN)** is more critical than maximizing accuracy. This study demonstrates how a **data-driven approach** can improve retention strategies, balancing **predictive power and interpretability**.  

**The research-based model (Liang et al., 2016) performs better at identifying churned customers**, making it more suitable for **real-world financial risk analysis**.

---

## **📚 References**

- Lemos, R. A. de L., Silva, T. C., & Tabak, B. M. (2022). Propension to customer churn in a financial institution: A machine learning approach. *Neural Computing and Applications, 34*(11751–11768). [https://doi.org/10.1007/s00521-022-07067-x](https://doi.org/10.1007/s00521-022-07067-x)

- Kaggle - Bank Customer Churn Dataset by Gaurav Topre. Retrieved from [Kaggle](https://www.kaggle.com/datasets/gauravtopre/bank-customer-churn-dataset)

---

## ⚠️ **Important Note**
This project was developed as part of a **Data Science portfolio** to demonstrate the application of **machine learning techniques** in churn prediction. The results are **not intended to serve as conclusive business strategies or a scientific study**.



