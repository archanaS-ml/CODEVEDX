# Utility Usage Prediction Tool

A console-based machine learning application that predicts a household's next month utility usage (electricity, water, etc.) based on household size and previous consumption data.

This was built as **Project 1** of my AI/ML Engineering Internship at **CodeVedX**, focused on Python fundamentals, ML foundations, and real-world workflow practices.

## Features
- Menu-driven console interface
- Add and update utility usage records
- CSV-based data storage (persists between runs)
- Simple ML prediction model (Linear Regression) trained on stored data
- Input validation and exception handling throughout

## Tech Stack
- Python 3
- Pandas
- Scikit-learn

## How It Works
The tool stores usage records (household size, previous month's units, month, current month's units) in a CSV file. Once enough data is available, it trains a Linear Regression model to predict the next month's utility usage based on household size and prior consumption.

## Installation
```bash
pip install pandas scikit-learn
