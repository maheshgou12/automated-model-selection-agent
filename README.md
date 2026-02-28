# Automated Model Selection Agent

## 📌 Project Overview
The Automated Model Selection Agent is a Machine Learning automation system that automatically trains multiple machine learning algorithms on a dataset, evaluates their performance, and selects the best-performing model based on accuracy.  
The project reduces manual effort in model experimentation and improves efficient model selection.

---

## 🎯 Problem Statement
Choosing the best machine learning model manually requires testing several algorithms and comparing results.  
This project automates the entire process of training, evaluation, and selection of the optimal model.

---

## 🚀 Features
- Automatic dataset loading
- Data preprocessing and cleaning
- Train-Test data splitting
- Training multiple ML algorithms
- Model performance comparison
- Automatic best model selection
- Saving best model as `.pkl` file
- Performance visualization

---

## 🧠 Algorithms Used
- Logistic Regression
- Decision Tree Classifier
- Random Forest Classifier
- Support Vector Machine (SVM)

---

## ⚙️ Technologies Used
- Python
- Scikit-learn
- Pandas
- NumPy
- Matplotlib
- Joblib / Pickle

---

## 🔄 Workflow
1. Load dataset  
2. Perform preprocessing  
3. Split dataset into training and testing sets  
4. Train multiple machine learning models  
5. Evaluate models using accuracy score  
6. Compare model performances  
7. Select and save the best-performing model  

---

## ▶️ How to Run

```bash
pip install -r requirements.txt
python main.py

