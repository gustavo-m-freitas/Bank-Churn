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

## ⚖ **Model Comparison: Research vs. New Approaches**
This project evaluates two different **customer churn prediction strategies**:

### **Research Model**  
**Stratified Sampling + Feature Selection**  
**Recall: ~78%** (Best Model for Detecting Churn)  
**Accuracy: ~80%**  
**Lower False Negatives (~15%)**

### **New Models (SMOTE + Cross-Validation)**  
**Higher Accuracy (~95%) but Lower Recall (~60%)**  
**High False Negatives (~48%) → Misses Nearly Half of Churned Customers**  
**Low Precision (~0.39 for XGBoost, 0.36 for RF)**

🔎 **Key Finding**: The **Research Model** prioritizing recall over accuracy performs better in identifying customers at risk of churn, which is the primary goal of this project.

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

## 💡 **Why Recall Matters More Than Accuracy**

Since our goal is **to identify customers who are most likely to churn**, **recall** is prioritized over **accuracy**. A model with high accuracy may still miss a large number of churned customers, which is not ideal for proactive retention strategies.

### **Recall vs Accuracy**
- **Recall** is crucial for identifying churned customers, as this allows the bank to target these customers with retention strategies. Missing churned customers would lead to missed opportunities to prevent churn.
- **Accuracy**, while important, does not necessarily tell us how well the model identifies churn. For example, a model that classifies most customers as non-churned will have high accuracy but will fail to catch the churn cases.

---

## 📊 **Graphical Representation**

| Model                          | Accuracy | Recall  |
|---------------------------------|----------|---------|
| **Model 1 (Balanced Data)**     | 0.769764 | 0.782095|
| **Model 2 (SMOTE + Cross-Validation)** | 0.853222 | 0.618600|

The **above table** compares **accuracy** and **recall** for two different models. While **Model 2** achieves a higher accuracy (~85%), it fails to capture as many churned customers compared to **Model 1**, where recall is prioritized (~78%).

---

## 🛠 **Technologies & Tools Used**
- **Programming Language**: Python (Pandas, NumPy, Scikit-learn, XGBoost)
- **Visualization**: Matplotlib, Seaborn
- **Machine Learning**: RandomizedSearchCV, GridSearchCV, K-Fold Cross-Validation
- **Data Processing**: SMOTE, Standardization, One-Hot Encoding, Feature Engineering

---

## 📊 **Dataset Information**
- **Source**: Bank Customer Churn Dataset
- **Number of Samples**: 10,000
- **Features**: 10 demographic and financial features (age, gender, credit score, etc.)
- **Target Variable**: Churn (1 = Churned, 0 = Non-Churned)

---

## 📅 **Conclusion**
In **customer churn prediction**, **minimizing False Negatives (FN)** is more critical than maximizing accuracy. This study demonstrates how a **data-driven approach** can improve retention strategies, balancing **predictive power and interpretability**.  

**The research-based model (Liang et al., 2016) performs better at identifying churned customers**, making it more suitable for **real-world financial risk analysis**.

---

## ⚠️ **Important Note**
This project was developed as part of a **Data Science portfolio** to demonstrate the application of **machine learning techniques** in churn prediction. The results are **not intended to serve as conclusive business strategies or a scientific study**.


