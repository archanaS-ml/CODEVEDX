# AI Based Fake News Detection Tool

A console-based machine learning application that classifies news headlines or articles as **Fake** or **Real**, along with a confidence score for each prediction. It combines text preprocessing (tokenization, stopword removal), TF-IDF vectorization, and a Logistic Regression classifier to turn raw, unstructured news text into a reliable classification system — with the trained model saved to disk so it doesn't need to retrain every time it runs.

This was built as **Project 3** of my AI/ML Engineering Internship at **CodeVedX**, focused on Natural Language Processing fundamentals and text handling — including text cleaning, feature extraction, text classification concepts, and debugging ML pipelines.

## Features
- Menu-driven console interface
- News text input for real-time classification
- Text preprocessing — lowercasing, punctuation/number removal, tokenization, stopword removal
- TF-IDF vectorization to convert text into meaningful numeric features
- Logistic Regression model for Fake/Real classification
- Confidence score output for every prediction
- Trained model and vectorizer saved to disk (pickle) for reuse without retraining
- Exception handling throughout

## Tech Stack
- Python 3
- Pandas
- Scikit-learn (TF-IDF, Logistic Regression, model evaluation)

## How It Works
The tool loads (or auto-generates) a labeled news dataset, cleans and tokenizes the text, removes stopwords, and converts it into TF-IDF vectors. A Logistic Regression model is trained on this data and evaluated for accuracy. Once trained, the model and vectorizer are saved to disk. On future runs, the saved model loads instantly, letting the user input any news text and instantly get a Fake/Real prediction along with a confidence percentage.

## Installation
```bash
pip install pandas scikit-learn
