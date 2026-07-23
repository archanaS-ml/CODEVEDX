import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

DATA_FILE = "student_data.csv"


def create_sample_data():
    if os.path.exists(DATA_FILE):
        return

    np.random.seed(42)
    n = 60
    attendance = np.random.randint(50, 100, n).astype(float)
    marks = np.random.randint(35, 100, n).astype(float)
    study_hours = np.random.randint(1, 8, n).astype(float)

    final_result = np.where((attendance >= 70) & (marks >= 50), "Pass", "Fail")

    df = pd.DataFrame({
        "student_id": range(1, n + 1),
        "attendance": attendance,
        "marks": marks,
        "study_hours": study_hours,
        "final_result": final_result,
    })

    for col in ["attendance", "marks", "study_hours"]:
        missing_idx = np.random.choice(df.index, size=3, replace=False)
        df.loc[missing_idx, col] = np.nan

    df.to_csv(DATA_FILE, index=False)
    print(f"Sample dataset created: {DATA_FILE}\n")


def load_and_clean_data():
    try:
        df = pd.read_csv(DATA_FILE)
    except FileNotFoundError:
        print("Data file not found.")
        return pd.DataFrame()
    except Exception as e:
        print(f"Error loading data: {e}")
        return pd.DataFrame()

    print("--- Missing values before cleaning ---")
    print(df.isnull().sum(), "\n")

    numeric_cols = ["attendance", "marks", "study_hours"]
    for col in numeric_cols:
        df[col] = df[col].fillna(df[col].mean())

    df = df.dropna(subset=["final_result"])

    print("--- Missing values after cleaning ---")
    print(df.isnull().sum(), "\n")

    return df


def explore_data(df):
    print("--- Dataset Summary ---")
    print(df.describe(), "\n")

    print("--- Final Result Distribution ---")
    print(df["final_result"].value_counts(), "\n")

    print("--- Correlation Between Numeric Features ---")
    print(df[["attendance", "marks", "study_hours"]].corr(), "\n")


def select_features(df):
    X = df[["attendance", "marks", "study_hours"]]
    y = df["final_result"].map({"Pass": 1, "Fail": 0})
    return X, y


def train_model(X, y):
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    model = LogisticRegression()
    model.fit(X_train, y_train)
    return model, X_test, y_test


def evaluate_model(model, X_test, y_test):
    y_pred = model.predict(X_test)

    acc = accuracy_score(y_test, y_pred)
    print(f"--- Model Accuracy: {acc * 100:.2f}% ---\n")

    print("--- Confusion Matrix ---")
    print(confusion_matrix(y_test, y_pred), "\n")

    print("--- Classification Report ---")
    print(classification_report(y_test, y_pred, target_names=["Fail", "Pass"]))

    return y_pred


def create_visualizations(df):
    plt.figure(figsize=(5, 4))
    df["final_result"].value_counts().plot(kind="bar", color=["green", "red"])
    plt.title("Pass vs Fail Count")
    plt.xlabel("Result")
    plt.ylabel("Number of Students")
    plt.tight_layout()
    plt.savefig("chart_pass_fail_count.png")
    plt.close()

    plt.figure(figsize=(5, 4))
    colors = df["final_result"].map({"Pass": "green", "Fail": "red"})
    plt.scatter(df["study_hours"], df["marks"], c=colors)
    plt.title("Marks vs Study Hours")
    plt.xlabel("Study Hours")
    plt.ylabel("Marks")
    plt.tight_layout()
    plt.savefig("chart_marks_vs_study_hours.png")
    plt.close()

    plt.figure(figsize=(5, 4))
    df["attendance"].plot(kind="hist", bins=10, color="skyblue", edgecolor="black")
    plt.title("Attendance Distribution")
    plt.xlabel("Attendance (%)")
    plt.tight_layout()
    plt.savefig("chart_attendance_distribution.png")
    plt.close()

    print("Charts saved: chart_pass_fail_count.png, "
            "chart_marks_vs_study_hours.png, chart_attendance_distribution.png\n")


def predict_new_student(model):
    print("--- Predict Performance for a New Student ---")
    try:
        attendance = float(input("Attendance (%): "))
        marks = float(input("Marks (out of 100): "))
        study_hours = float(input("Study hours per day: "))

        input_df = pd.DataFrame(
            [[attendance, marks, study_hours]],
            columns=["attendance", "marks", "study_hours"],
        )
        result = model.predict(input_df)[0]
        print("Predicted Result:", "Pass" if result == 1 else "Fail", "\n")
    except ValueError:
        print("Invalid input. Please enter numeric values only.\n")
    except Exception as e:
        print(f"Error during prediction: {e}\n")


def main():
    create_sample_data()
    df = load_and_clean_data()

    if df.empty:
        print("No data available. Exiting.")
        return

    explore_data(df)

    X, y = select_features(df)
    model, X_test, y_test = train_model(X, y)
    evaluate_model(model, X_test, y_test)
    create_visualizations(df)
    predict_new_student(model)


if __name__ == "__main__":
    main()