"""
AI Based Fake News Detection Tool
-----------------------------------
Internship Project 3 - CodeVedX

Requirements:
  pip install pandas scikit-learn

Data model (news_data.csv):
  id, text, label   (label is "Fake" or "Real")
"""

import os
import re
import pickle
import pandas as pd

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_extraction import text as sk_text
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report

DATA_FILE = "news_data.csv"
MODEL_FILE = "fake_news_model.pkl"
VECTORIZER_FILE = "tfidf_vectorizer.pkl"

STOPWORDS = sk_text.ENGLISH_STOP_WORDS


def create_sample_data():
    if os.path.exists(DATA_FILE):
        return

    real_news = [
        "The government announced a new policy to improve public healthcare access.",
        "Scientists discovered a new species of frog in the Amazon rainforest.",
        "The central bank raised interest rates to control inflation this quarter.",
        "Local schools will reopen next week after the holiday break.",
        "The company reported steady growth in its quarterly earnings report.",
        "Researchers published a study on renewable energy efficiency improvements.",
        "The city council approved funding for new public transportation routes.",
        "A new vaccine trial showed promising results in early clinical stages.",
        "The stock market closed slightly higher after a stable trading session.",
        "Local farmers reported a good harvest season due to favorable weather.",
    ]

    fake_news = [
        "Shocking! Scientists confirm the moon is actually a hologram made by aliens.",
        "You won't believe this one weird trick that cures all diseases overnight.",
        "Secret government files reveal the earth is flat and NASA is hiding it.",
        "Celebrity secretly replaced by a robot clone, insiders claim.",
        "Miracle fruit found to reverse aging completely within 24 hours.",
        "Government plans to ban all sunlight starting next month, sources say.",
        "This ancient text predicts the exact date the world will end next year.",
        "Doctors hate this simple trick that makes you lose 20kg in a week.",
        "Breaking: time travel machine invented in a random man's garage.",
        "Aliens spotted controlling world leaders, anonymous whistleblower claims.",
    ]

    texts = real_news + fake_news
    labels = ["Real"] * len(real_news) + ["Fake"] * len(fake_news)

    df = pd.DataFrame({
        "id": range(1, len(texts) + 1),
        "text": texts,
        "label": labels,
    })

    df.to_csv(DATA_FILE, index=False)
    print(f"Sample dataset created: {DATA_FILE}\n")


def clean_text(raw_text):
    try:
        text = str(raw_text).lower()
        text = re.sub(r"[^a-z\s]", " ", text)
        tokens = text.split()
        tokens = [word for word in tokens if word not in STOPWORDS]
        return " ".join(tokens)
    except Exception as e:
        print(f"Error cleaning text: {e}")
        return ""


def load_and_preprocess_data():
    try:
        df = pd.read_csv(DATA_FILE)
    except FileNotFoundError:
        print("Data file not found.")
        return pd.DataFrame()
    except Exception as e:
        print(f"Error loading data: {e}")
        return pd.DataFrame()

    df = df.dropna(subset=["text", "label"])
    df["clean_text"] = df["text"].apply(clean_text)
    return df


def vectorize_and_split(df):
    vectorizer = TfidfVectorizer()
    X = vectorizer.fit_transform(df["clean_text"])
    y = df["label"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    return vectorizer, X_train, X_test, y_train, y_test


def train_model(X_train, y_train):
    model = LogisticRegression()
    model.fit(X_train, y_train)
    return model


def evaluate_model(model, X_test, y_test):
    y_pred = model.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    print(f"--- Model Accuracy: {acc * 100:.2f}% ---\n")
    print("--- Classification Report ---")
    print(classification_report(y_test, y_pred))


def save_model(model, vectorizer):
    try:
        with open(MODEL_FILE, "wb") as f:
            pickle.dump(model, f)
        with open(VECTORIZER_FILE, "wb") as f:
            pickle.dump(vectorizer, f)
        print(f"Model saved to {MODEL_FILE}")
        print(f"Vectorizer saved to {VECTORIZER_FILE}\n")
    except Exception as e:
        print(f"Error saving model: {e}")


def load_model():
    try:
        with open(MODEL_FILE, "rb") as f:
            model = pickle.load(f)
        with open(VECTORIZER_FILE, "rb") as f:
            vectorizer = pickle.load(f)
        return model, vectorizer
    except Exception as e:
        print(f"Error loading saved model: {e}")
        return None, None


def predict_news(model, vectorizer, news_text):
    try:
        cleaned = clean_text(news_text)
        vectorized = vectorizer.transform([cleaned])

        prediction = model.predict(vectorized)[0]
        probabilities = model.predict_proba(vectorized)[0]
        confidence = max(probabilities) * 100

        print(f"\nPrediction: {prediction}")
        print(f"Confidence: {confidence:.2f}%\n")
    except Exception as e:
        print(f"Error during prediction: {e}\n")


def display_menu():
    print("=" * 45)
    print(" AI BASED FAKE NEWS DETECTION TOOL")
    print("=" * 45)
    print("1. Check a news headline/article")
    print("2. Retrain model on current dataset")
    print("3. Exit")
    print("=" * 45)


def main():
    create_sample_data()

    if os.path.exists(MODEL_FILE) and os.path.exists(VECTORIZER_FILE):
        model, vectorizer = load_model()
    else:
        model, vectorizer = None, None

    if model is None or vectorizer is None:
        df = load_and_preprocess_data()
        if df.empty:
            print("No data available. Exiting.")
            return
        vectorizer, X_train, X_test, y_train, y_test = vectorize_and_split(df)
        model = train_model(X_train, y_train)
        evaluate_model(model, X_test, y_test)
        save_model(model, vectorizer)

    while True:
        display_menu()
        try:
            choice = input("Enter your choice (1-3): ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nExiting program. Goodbye!")
            break

        if choice == "1":
            news_text = input("\nEnter a news headline or article text: ").strip()
            if news_text:
                predict_news(model, vectorizer, news_text)
            else:
                print("Input cannot be empty.\n")
        elif choice == "2":
            df = load_and_preprocess_data()
            if df.empty:
                print("No data available to retrain.\n")
                continue
            vectorizer, X_train, X_test, y_train, y_test = vectorize_and_split(df)
            model = train_model(X_train, y_train)
            evaluate_model(model, X_test, y_test)
            save_model(model, vectorizer)
        elif choice == "3":
            print("Exiting program. Goodbye!")
            break
        else:
            print("Invalid choice. Please enter a number between 1 and 3.\n")


if __name__ == "__main__":
    main()