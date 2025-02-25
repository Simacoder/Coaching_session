import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import streamlit as st
from imblearn.over_sampling import SMOTE
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix, mean_absolute_error, accuracy_score

# Streamlit App Title
st.set_page_config(page_title="Loan Default Prediction", layout="wide")

# Sidebar Navigation
st.sidebar.title("Navigation")
selection = st.sidebar.radio("Go to:", ["Data Visualization", "Model Comparison", "Authors"])

# Problem Statement
st.title("Loan Default Prediction")
st.write("""
### Problem Statement
Predict whether a customer will default on a loan based on their financial and demographic data.
""")

# Authors Section
if selection == "Authors":
    st.header("Authors")
    st.write("""
    - **Simang Mchunu** (Machine Learning Engineer)  
    - **Nkosinathi Nhlapo** (Data Scientist)  
    - **Kagiso Leboka** (Data Analyst)  
    - **Bongani Baloyi** (Software Engineer)  
    """)

# Load Data
@st.cache_data
def load_data():
    return pd.read_csv('data/Training Data.csv')

train = load_data()

if selection == "Data Visualization":
    st.header("Data Overview")
    st.write("### Sample Data")
    st.dataframe(train.head())

    default_counts = train['Risk_Flag'].value_counts()
    default_ratios = train['Risk_Flag'].value_counts(normalize=True)
    st.write("### Default Count:")
    st.write(default_counts)
    st.write("### Default Ratios:")
    st.write(default_ratios)

    # Data Visualization
    st.header("Data Visualization")
    numeric_cols = ['Income', 'Age', 'Experience', 'CURRENT_JOB_YRS', 'CURRENT_HOUSE_YRS']

    for col in numeric_cols:
        fig, ax = plt.subplots()
        sns.histplot(x=col, data=train, hue='Risk_Flag', common_norm=False, stat='percent', multiple='dodge', ax=ax)
        st.pyplot(fig)

    # Categorical Variables
    fig, ax = plt.subplots()
    sns.countplot(x='Risk_Flag', data=train, hue='Married/Single', ax=ax)
    st.pyplot(fig)

    fig, ax = plt.subplots()
    sns.countplot(x='Risk_Flag', data=train, hue='House_Ownership', ax=ax)
    st.pyplot(fig)

# Data Preprocessing
train_encoded = pd.get_dummies(train, drop_first=True)
y = train_encoded['Risk_Flag'].values
X = train_encoded.drop(['Risk_Flag'], axis=1).values

# Handle Class Imbalance
smote = SMOTE()
X, y = smote.fit_resample(X, y)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=0, stratify=y)

if selection == "Model Comparison":
    # Logistic Regression Model
    st.header("Model Comparison")
    st.subheader("Logistic Regression")
    logreg = LogisticRegression()
    logreg.fit(X_train, y_train)
    pred_log = logreg.predict(X_test)
    acc_log = accuracy_score(y_test, pred_log)
    mae_log = mean_absolute_error(y_test, pred_log)
    st.write("Accuracy:", acc_log)
    st.write("Mean Absolute Error:", mae_log)
    st.text("Classification Report:")
    st.text(classification_report(y_test, pred_log))
    st.write("Confusion Matrix:")
    st.dataframe(confusion_matrix(y_test, pred_log))

    # Random Forest Model
    st.subheader("Random Forest")
    rfc = RandomForestClassifier(n_estimators=100, random_state=0)
    rfc.fit(X_train, y_train)
    pred_rf = rfc.predict(X_test)
    acc_rf = accuracy_score(y_test, pred_rf)
    mae_rf = mean_absolute_error(y_test, pred_rf)
    st.write("Accuracy:", acc_rf)
    st.write("Mean Absolute Error:", mae_rf)
    st.text("Classification Report:")
    st.text(classification_report(y_test, pred_rf))
    st.write("Confusion Matrix:")
    st.dataframe(confusion_matrix(y_test, pred_rf))

    # Model Comparison
    st.subheader("Model Performance Comparison")
    comparison_df = pd.DataFrame({
        'Model': ['Logistic Regression', 'Random Forest'],
        'Accuracy': [acc_log, acc_rf],
        'MAE': [mae_log, mae_rf]
    })
    st.dataframe(comparison_df)

    st.write("### Conclusion:")
    st.write("Random Forest generally provides better accuracy, but Logistic Regression is computationally efficient and interpretable.")
