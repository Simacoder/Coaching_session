# Data Science & Analytics Coaching Session

Welcome to the **Data Science & Analytics Coaching Session**! This repository contains all the materials, code examples, and instructions used during the session. Our goal is to empower you with practical data science techniques and analytics strategies that you can apply to real-world problems.

---

## Table of Contents

- [Data Science \& Analytics Coaching Session](#data-science--analytics-coaching-session)
  - [Table of Contents](#table-of-contents)
  - [Overview](#overview)
  - [Session Objectives](#session-objectives)
  - [Prerequisites](#prerequisites)
  - [Session Agenda](#session-agenda)
  - [Project Example: Loan default analysis](#project-example-loan-default-analysis)
    - [Data Generation](#data-generation)
    - [Data Analysis \& Transformation](#data-analysis--transformation)
    - [Model Training \& Evaluation](#model-training--evaluation)
    - [API Deployment with FastAPI \& Docker](#api-deployment-with-fastapi--docker)
    - [Live Dashboard with Streamlit](#live-dashboard-with-streamlit)
  - [Live Coding \& Quizzes](#live-coding--quizzes)
  - [Next Steps \& Additional Resources](#next-steps--additional-resources)
    - [Recommended Reading \& Resources](#recommended-reading--resources)
  - [Contact \& Acknowledgements](#contact--acknowledgements)

---

## Overview

In this coaching session, we explore the practical side of data science and analytics. We'll work through a complete machine learning pipeline using a real-world dataset (a simulated sales dataset for risky Loan default analysis). We’ll cover data generation, preprocessing, exploratory data analysis (EDA), model training, and deployment.

---

## Session Objectives

- **Understand the Value of Data Science:** Learn how data science and analytics drive decision-making in business.
- **Hands-On Data Preparation:** Generate and preprocess real-world sales data.
- **Build & Evaluate Models:** Train models to forecast sales using techniques such as Random Forest Regression.
- **Deploy Models:** Deploy your models using FastAPI and Docker.
- **Visualize Data & Predictions:** Create interactive dashboards using Streamlit.
- **Interactive Learning:** Participate in live coding and quizzes to reinforce key concepts.

---

## Prerequisites

Before attending the session, please ensure you have the following installed:

- **Python 3.8+**
- **Jupyter Notebook or Google Colab**
- **Docker** ([Installation Instructions](https://www.docker.com/get-started))
- **FastAPI** and **Uvicorn** (`pip install fastapi uvicorn`)
- **Streamlit** (`pip install streamlit`)
- Additional Python packages: `pandas`, `numpy`, `scikit-learn`, `plotly`, `seaborn`, `pickle`, `mlflow`

---

## Session Agenda

1. **Introduction to Data Science & Analytics**
   - Overview of key concepts
   - Real-world applications and case studies

2. **Data Generation & Preparation**
   - Creating a simulated sales dataset for a spaza shop
   - Data cleaning, feature engineering, and transformation

3. **Exploratory Data Analysis (EDA)**
   - Visualizing sales trends and correlations
   - Statistical analysis of sales data

4. **Machine Learning Model Training**
   - Building and evaluating a Random Forest Regressor for sales forecasting
   - Discussion on overfitting, data leakage, and model performance

5. **Model Deployment**
   - Deploying the model as an API with FastAPI
   - Containerizing the API using Docker

6. **Building a Live Dashboard**
   - Creating an interactive dashboard with Streamlit
   - Integrating real-time predictions and visualizations

7. **Live Coding & Quizzes**
   - Interactive coding challenges
   - Small quizzes to reinforce learning

8. **Next Steps & Q&A**
   - Discussing advanced topics (MLOps, CI/CD, cloud deployment)
   - Open floor for questions and further discussions

---

## Project Example: Loan default analysis 

### Data Generation

A Python script is provided to generate a simulated sales dataset including:
- Transaction date
- Store location
- Product, category, and pricing details
- Customer demographics and payment method
- Seasonal discounts and trends

### Data Analysis & Transformation

Key steps include:
- Exploratory Data Analysis (EDA) using Pandas, Seaborn, and Plotly.
- Statistical analysis and correlation heatmaps.
- Data transformation techniques like one-hot encoding and normalization.

### Model Training & Evaluation

The session covers:
- Splitting the data into training and testing sets.
- Training a Random Forest model.
- Evaluating model performance using metrics like MAE (Mean Absolute Error).
- Strategies to prevent overfitting and data leakage.

### API Deployment with FastAPI & Docker

Learn to:
- Build a FastAPI endpoint to serve model predictions.
- Use Pydantic for request validation.
- Containerize the application with Docker for deployment.

### Live Dashboard with Streamlit

Create an interactive dashboard that:
- Visualizes sales trends and statistical insights.
- Provides real-time predictions via an integrated API.
- Offers user-friendly filters and dynamic charts.

---

## Live Coding & Quizzes

During the session, you'll participate in:
- **Live Coding:** Step-by-step coding exercises to implement data processing, modeling, and deployment.
- **Interactive Quizzes:** Quick questions to test your understanding of concepts such as overfitting, feature engineering, and API development.

---

## Next Steps & Additional Resources

After the session, you can further explore:
- **MLOps and CI/CD:** Automating model retraining and deployment.
- **Cloud Deployment:** Deploying your applications on AWS, Google Cloud, or Azure.
- **Advanced Analytics:** Time-series forecasting with LSTM, reinforcement learning, and more.

### Recommended Reading & Resources

- [Data Science Handbook](https://www.datasciencehandbook.org/)
- [FastAPI Documentation](https://fastapi.tiangolo.com/)
- [Streamlit Documentation](https://docs.streamlit.io/)
- [Docker Documentation](https://docs.docker.com/)

---

## Contact & Acknowledgements

**Produced by:**  
- *Simanga Mchunu* – Machine Learning Engineer/ ALX_SE Alumin
- *Nkosinathi Nhlapo* - ALX_DS Alumin
- *Kagiso Leboka* - ALX_DA Alumin
- *Bongani Baloyi* - Software Engineer/ALX_SE Alumin

For questions, suggestions, or feedback, please reach out via [simacoder@hotmail.com](mailto:simacoder@hotmail.com).

Special thanks to all participants and contributors who made this coaching session a success!

---

Happy Coding & Data Exploring! 🚀
