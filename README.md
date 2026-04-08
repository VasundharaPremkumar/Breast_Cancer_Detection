# 🩺 Breast Cancer Prediction using Machine Learning

## 📌 Project Overview
This project uses **Machine Learning (Logistic Regression)** to predict whether a tumor is **malignant (cancerous)** or **benign (non-cancerous)** based on medical features.

The goal is to build a simple, efficient, and accurate classification model that can assist in early cancer detection.

---

## 🚀 Features
- Data preprocessing and cleaning  
- Handling missing/unnecessary columns  
- Feature scaling using StandardScaler  
- Logistic Regression model training  
- Model evaluation using accuracy score  
- Custom input prediction  

---

## 📂 Dataset
The dataset used for this project is the **Breast Cancer Wisconsin Dataset**.

🔗 Dataset Link:  
https://www.kaggle.com/datasets/uciml/breast-cancer-wisconsin-data?resource=download

---

## 🛠️ Tech Stack
- Python 🐍  
- NumPy  
- Pandas  
- Scikit-learn  

---

## ⚙️ Workflow
1. Load dataset  
2. Data cleaning (remove unnecessary columns like `id`, `Unnamed: 32`)  
3. Encode target variable (`M → 1`, `B → 0`)  
4. Split dataset into training and testing sets  
5. Scale features using StandardScaler  
6. Train Logistic Regression model  
7. Evaluate model performance  
8. Predict results using custom input  

---

## 📊 Model Performance
- Algorithm: Logistic Regression  
- Evaluation Metric: Accuracy Score  
- Achieved Accuracy: ~95% (may vary slightly)

---

## 🧪 Example Prediction
```python
if predict[0] == 1:
    print("Cancerous")
else:
    print("Not Cancerous")
