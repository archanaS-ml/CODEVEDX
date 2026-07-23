# Student Performance Prediction System

A machine learning application that predicts a student's final academic performance (Pass/Fail) based on attendance, marks, and study hours.

This was built as **Project 2** of my AI/ML Engineering Internship at **CodeVedX**, focused on data preprocessing, exploratory data analysis, and ML workflow practices.

## Features
- Data cleaning and handling of missing values
- Exploratory Data Analysis (EDA) — summary statistics, distributions, correlations
- Feature selection (attendance, marks, study hours)
- ML model (Logistic Regression) to predict final performance
- Data visualization — Pass/Fail count, marks vs. study hours, attendance distribution charts
- Model accuracy evaluation — accuracy score, confusion matrix, classification report

## Tech Stack
- Python 3
- Pandas, NumPy
- Matplotlib
- Scikit-learn

## How It Works
The tool loads (or auto-generates) a student dataset containing attendance, marks, study hours, and final result. It cleans missing values, explores the data, trains a Logistic Regression model on the selected features, evaluates its accuracy, generates visual charts, and predicts the outcome for a new student based on user input.

## Installation
```bash
pip install pandas numpy matplotlib scikit-learn
