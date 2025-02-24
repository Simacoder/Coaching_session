# **Loan Default Risk Analysis and Prediction**

## **Project Overview**
Loan default risk prediction is critical for banks, lenders, and credit agencies. Instead of passively reacting to insolvency losses, financial institutions can proactively identify customers who are likely to default and take preventive measures.

This project analyzes customer profiling based on financial and demographic attributes (such as income, occupation, and marriage status) and applies machine learning models to predict loan default risk.

## **Dataset**
The dataset consists of **252,000** customer records with attributes such as:
- **Income**
- **Age**
- **Experience**
- **Marital Status**
- **House Ownership**
- **Car Ownership**
- **Profession**
- **Years at Current Job**
- **Years at Current Residence**
- **Risk_Flag** (Target variable: **0 = No Default, 1 = Default**)

## **Objective**
The goal is to **classify customers into two categories: default vs. non-default**, using machine learning models.

## **Authors**
- **Simang Mchunu** – Machine Learning Engineer  
- **Nkosinathi Nhlapo** – Data Scientist  
- **Kagiso Leboka** – Data Analyst  
- **Bongani Baloyi** – Software Engineer  

---

## **Exploratory Data Analysis (EDA)**
We first perform **data visualization** to understand key patterns:
- **Income Distribution vs. Default Rate**
- **Age vs. Default Risk**
- **Marital Status Impact on Default**
- **Profession & Geographic Trends**

## **Model Selection**
### **Step 1: Determine the Learning Approach**
- ✅ **Supervised Learning** (Labeled dataset with `Risk_Flag`)
- ✅ **Classification Problem** (Binary outcome: default or non-default)
- ✅ **Large Dataset (>100K samples)** → Requires **complex models**

### **Step 2: Model Choices**
We evaluate two models:
1. **Logistic Regression** – Simple, interpretable, but less accurate.
2. **Random Forest** – More powerful, handles large datasets better but harder to interpret.

| Model | Accuracy | Precision | Recall | F1-Score |
|--------|----------|----------|--------|---------|
| Logistic Regression | ~51% | Low | Low | Low |
| Random Forest | ~75% | Higher | Higher | Higher |

🛠 **Final Choice:** **Random Forest**, as it provides a better trade-off between performance and interpretability.

---

## **Implementation Steps**
1. **Data Preprocessing**
   - Handle missing values
   - Convert categorical variables to numerical using one-hot encoding
   - Address class imbalance using **SMOTE (Synthetic Minority Over-sampling Technique)**

2. **Model Training**
   - Split data into **training (70%)** and **testing (30%)** sets
   - Train models using **Logistic Regression** and **Random Forest**
   - Optimize hyperparameters using **Grid Search**

3. **Model Evaluation**
   - Calculate **Accuracy, Precision, Recall, F1-score**
   - Display **Confusion Matrix**

4. **Predictions on New Data**
   - Load **test dataset**
   - Predict and classify **potentially risky customers**

---

## **Technologies Used**
- **Python**
- **Pandas, NumPy** – Data Manipulation
- **Matplotlib, Seaborn** – Data Visualization
- **Scikit-Learn** – Machine Learning Models
- **Imbalanced-learn (SMOTE)** – Handling Imbalanced Data

---

## **Results & Insights**
- **Younger customers and those with low experience tend to default more**
- **Renters have a higher risk of default compared to homeowners**
- **Certain professions and geographic locations have higher default risks**
- **Random Forest outperforms Logistic Regression with ~75% accuracy**

📌 **Business Implication**:  
Using this model, financial institutions can identify high-risk customers and take **preventive actions** (e.g., adjusting credit limits, offering financial education, or requiring additional collateral).

---

## **How to Run the Project**
### **1. Install Dependencies**
```bash
pip install pandas numpy matplotlib seaborn scikit-learn imbalanced-learn
```
### **2. Run the Notebook**
Open **Jupyter Notebook** or **Google Colab** and execute the step-by-step code.

---

## **Future Enhancements**
- ✅ **Try Advanced Models:** XGBoost, LightGBM for better accuracy  
- ✅ **Explain Model Predictions:** Use **SHAP (SHapley Additive Explanations)**  
- ✅ **Deploy as a Web App** using **Streamlit**  

---

## **Conclusion**
This project successfully builds a predictive model for **loan default risk**, helping financial institutions **minimize losses** by identifying **high-risk customers** before defaults happen.

🔹 **Best Model:** Random Forest  
🔹 **Accuracy:** ~75%  
🔹 **Business Impact:** Improved loan risk management 🚀  

📢 **Have questions or suggestions?** Feel free to reach out to the authors!  

