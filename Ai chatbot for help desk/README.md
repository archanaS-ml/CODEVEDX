# AI Chatbot for Internal Helpdesk

A Flask-based conversational AI application that answers employee questions on IT, HR, and admin topics in real time, using intent matching to find the closest known answer from a growing FAQ knowledge base. It combines TF-IDF vectorization, cosine similarity, and a lightweight web interface to turn a static FAQ list into an interactive helpdesk assistant — complete with an admin panel for adding new questions without touching any code.

This was built as **Project 4** of my AI/ML Engineering Internship at **CodeVedX**, focused on NLP fundamentals, intent detection, and backend integration — including dataset preparation, Flask basics, and collaborative development practices.

## Features
- Web-based chat interface for real-time Q&A
- Intent recognition using TF-IDF vectorization and cosine similarity matching
- FAQ dataset stored and trained from a CSV file
- Confidence-based fallback response for unrecognized questions
- Admin dashboard to view and add new FAQ entries live, without restarting the app
- Exception handling throughout

## Tech Stack
- Python 3
- Flask
- Pandas
- Scikit-learn (TF-IDF, cosine similarity)

## How It Works
The tool loads (or auto-generates) a set of FAQ question-answer pairs and converts the questions into TF-IDF vectors. When a user asks something in the chat window, their question is vectorized the same way and compared against the stored FAQs using cosine similarity. The closest matching answer is returned if the similarity score crosses a set threshold; otherwise, the bot responds with a fallback message. New FAQs can be added anytime through the admin page, instantly refreshing the chatbot's knowledge base.

## Installation
```bash
pip install flask pandas scikit-learn
```

## Usage
```bash
python3 helpdesk_chatbot.py
```

Then open in a browser:
- Chat interface: `http://127.0.0.1:5000/`
- Admin panel: `http://127.0.0.1:5000/admin`

## Author
Archana — AI/ML Engineering Student
GitHub: [archanaS-ml](https://github.com/archanaS-ml)
