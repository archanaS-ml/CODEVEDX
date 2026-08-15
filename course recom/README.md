# Smart Course Recommendation System

A content-based machine learning application that recommends online courses based on a course you've already liked or a topic you want to learn, using TF-IDF vectorization and cosine similarity to match learner intent with the right courses.

This was built as **Project 5** of my AI/ML Engineering Internship at **CodeVedX**, bringing together everything from the internship — data handling, ML similarity logic, backend integration, and frontend design — into one full web application.

## Features
- Two recommendation modes: search by a course you liked, or describe what you want to learn
- Content-based filtering using course category, level, and description
- ML similarity scoring (TF-IDF + cosine similarity) with a match percentage shown per result
- Browsable, filterable course catalog by category
- Clean web interface built with Flask and a responsive frontend
- Modular project structure separating recommendation logic, backend, and frontend

## Tech Stack
- Python 3
- Flask
- Pandas
- Scikit-learn (TF-IDF, cosine similarity)
- HTML, CSS, JavaScript (Jinja templating)

## Project Structure
## How It Works
The system loads a catalog of courses and builds a content profile for each one by combining its category, level, and description. These profiles are converted into TF-IDF vectors. When a user searches by a course title or describes what they want to learn, their input is vectorized the same way and compared against the catalog using cosine similarity, returning the closest matches ranked by similarity score.

## Installation
```bash
pip install flask pandas scikit-learn
```

## Usage
```bash
python3 app.py
```
Then open in a browser:
