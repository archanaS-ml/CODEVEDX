import os
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

DATA_FILE = "courses_data.csv"

def create_sample_data():
    """Generates a ~45-course demo catalog spanning many categories, so the
    content-based filtering has real variety to work with. Replace
    courses_data.csv with your own dataset any time (same 'title',
    'category', 'level', 'duration_hours', 'rating', 'description'
    columns) and this function will skip."""
    if os.path.exists(DATA_FILE):
        return

    courses = [
        ("Python Programming Fundamentals", "Programming", "Beginner", 12, 4.6,
         "Learn Python syntax, data types, loops, and functions through hands-on coding exercises."),
        ("Advanced Python for Data Science", "Data Science", "Advanced", 20, 4.7,
         "Master NumPy, Pandas, and data manipulation techniques for real-world data science projects."),
        ("Machine Learning A-Z", "AI & Machine Learning", "Intermediate", 30, 4.8,
         "Build and evaluate machine learning models covering regression, classification, and clustering."),
        ("Deep Learning with Neural Networks", "AI & Machine Learning", "Advanced", 25, 4.7,
         "Understand neural network architectures and build deep learning models using modern frameworks."),
        ("Natural Language Processing Basics", "AI & Machine Learning", "Intermediate", 18, 4.5,
         "Learn text preprocessing, tokenization, and how to build simple text classification models."),
        ("Full-Stack Web Development", "Web Development", "Beginner", 40, 4.6,
         "Build complete web applications using HTML, CSS, JavaScript, and a backend framework."),
        ("React for Frontend Developers", "Web Development", "Intermediate", 16, 4.5,
         "Build interactive user interfaces with React components, hooks, and state management."),
        ("Backend Development with Flask", "Web Development", "Intermediate", 14, 4.4,
         "Learn to build REST APIs and backend services using the Flask framework."),
        ("Introduction to Cloud Computing", "Cloud Computing", "Beginner", 10, 4.3,
         "Understand core cloud concepts, deployment models, and popular cloud service providers."),
        ("AWS Certified Solutions Architect Prep", "Cloud Computing", "Advanced", 28, 4.7,
         "Prepare for the AWS certification exam by mastering core architecture and deployment principles."),
        ("DevOps and CI/CD Pipelines", "Cloud Computing", "Intermediate", 16, 4.5,
         "Learn to automate testing and deployment using modern CI/CD tools and practices."),
        ("Cybersecurity Fundamentals", "Cybersecurity", "Beginner", 14, 4.5,
         "Understand core security principles, common threats, and best practices for safe systems."),
        ("Ethical Hacking and Penetration Testing", "Cybersecurity", "Advanced", 22, 4.6,
         "Learn how attackers exploit systems and how to defend against real-world security threats."),
        ("Network Security Essentials", "Cybersecurity", "Intermediate", 15, 4.4,
         "Explore firewalls, VPNs, and network monitoring techniques to secure infrastructure."),
        ("UI/UX Design Principles", "Design", "Beginner", 12, 4.6,
         "Learn the fundamentals of user-centered design, wireframing, and prototyping."),
        ("Advanced Figma for Product Design", "Design", "Intermediate", 10, 4.5,
         "Master advanced Figma workflows for designing polished, production-ready interfaces."),
        ("Graphic Design Masterclass", "Design", "Beginner", 18, 4.4,
         "Learn typography, color theory, and layout design using industry-standard tools."),
        ("Digital Marketing Essentials", "Marketing", "Beginner", 10, 4.4,
         "Understand SEO, social media marketing, and content strategy for growing an audience."),
        ("Data-Driven Marketing Analytics", "Marketing", "Intermediate", 14, 4.5,
         "Learn to analyze marketing campaigns and make data-informed decisions."),
        ("Social Media Advertising Strategy", "Marketing", "Beginner", 8, 4.3,
         "Learn to plan and run effective ad campaigns across major social media platforms."),
        ("Business Fundamentals for Startups", "Business", "Beginner", 12, 4.4,
         "Learn the essentials of building and running a startup, from idea to execution."),
        ("Financial Accounting Basics", "Finance", "Beginner", 14, 4.3,
         "Understand core accounting principles, financial statements, and business reporting."),
        ("Personal Finance and Investing", "Finance", "Beginner", 9, 4.6,
         "Learn budgeting, saving, and the fundamentals of investing for long-term growth."),
        ("Project Management Professional Prep", "Business", "Advanced", 24, 4.7,
         "Prepare for project management certification by mastering planning and leadership frameworks."),
        ("Public Speaking and Communication Skills", "Personal Development", "Beginner", 8, 4.6,
         "Build confidence and clarity in public speaking through practical exercises and feedback."),
        ("Time Management and Productivity", "Personal Development", "Beginner", 6, 4.5,
         "Learn practical techniques to prioritize tasks and manage time more effectively."),
        ("Leadership and Team Management", "Personal Development", "Intermediate", 12, 4.5,
         "Develop leadership skills to manage teams and drive results in the workplace."),
        ("Spanish for Beginners", "Language Learning", "Beginner", 20, 4.6,
         "Learn conversational Spanish through vocabulary, grammar, and speaking practice."),
        ("French Language Foundations", "Language Learning", "Beginner", 18, 4.5,
         "Build a strong foundation in French grammar, pronunciation, and everyday conversation."),
        ("Business English Communication", "Language Learning", "Intermediate", 10, 4.4,
         "Improve professional English communication skills for meetings, emails, and presentations."),
        ("Photography Basics", "Photography", "Beginner", 8, 4.5,
         "Learn camera settings, composition, and lighting to take better photographs."),
        ("Portrait and Studio Photography", "Photography", "Intermediate", 12, 4.6,
         "Master studio lighting setups and techniques for professional portrait photography."),
        ("Mobile Photography and Editing", "Photography", "Beginner", 6, 4.3,
         "Learn to shoot and edit stunning photos using just a smartphone."),
        ("Music Production Fundamentals", "Music", "Beginner", 14, 4.5,
         "Learn the basics of music production, mixing, and using a digital audio workstation."),
        ("Guitar for Beginners", "Music", "Beginner", 10, 4.6,
         "Learn basic chords, strumming patterns, and your first songs on guitar."),
        ("Advanced Music Theory", "Music", "Advanced", 16, 4.4,
         "Deepen your understanding of harmony, scales, and composition techniques."),
        ("Yoga and Mindfulness Basics", "Health & Fitness", "Beginner", 6, 4.6,
         "Learn foundational yoga poses and mindfulness practices for stress relief and flexibility."),
        ("Strength Training Fundamentals", "Health & Fitness", "Beginner", 8, 4.5,
         "Understand proper form and programming basics for building strength safely."),
        ("Nutrition Science Essentials", "Health & Fitness", "Beginner", 10, 4.4,
         "Learn the fundamentals of nutrition science to build healthier eating habits."),
        ("SQL for Data Analysis", "Data Science", "Beginner", 10, 4.6,
         "Learn to write SQL queries to extract, filter, and analyze data from relational databases."),
        ("Data Visualization with Tableau", "Data Science", "Intermediate", 12, 4.5,
         "Learn to build interactive dashboards and visualizations to communicate data insights."),
        ("Statistics for Data Science", "Data Science", "Intermediate", 16, 4.6,
         "Build a solid statistical foundation for hypothesis testing and data-driven decision making."),
        ("Introduction to Blockchain Technology", "Programming", "Beginner", 9, 4.2,
         "Understand how blockchain works and explore its applications beyond cryptocurrency."),
        ("Mobile App Development with Flutter", "Programming", "Intermediate", 20, 4.5,
         "Build cross-platform mobile apps for iOS and Android using a single codebase."),
        ("Game Development with Unity", "Programming", "Intermediate", 22, 4.6,
         "Learn to design and build 2D and 3D games using the Unity game engine."),
    ]

    df = pd.DataFrame({
        "id": range(1, len(courses) + 1),
        "title": [c[0] for c in courses],
        "category": [c[1] for c in courses],
        "level": [c[2] for c in courses],
        "duration_hours": [c[3] for c in courses],
        "rating": [c[4] for c in courses],
        "description": [c[5] for c in courses],
    })
    df.to_csv(DATA_FILE, index=False)


def load_data():
    try:
        df = pd.read_csv(DATA_FILE)
        df = df.dropna(subset=["title", "category", "level", "description"])
        return df.reset_index(drop=True)
    except FileNotFoundError:
        return pd.DataFrame(columns=["id", "title", "category", "level", "duration_hours", "rating", "description"])
    except Exception as e:
        print(f"Error loading data: {e}")
        return pd.DataFrame(columns=["id", "title", "category", "level", "duration_hours", "rating", "description"])


def build_content_profiles(df):
  
    df["content_profile"] = (
        (df["category"].astype(str) + " ") * 2
        + df["level"].astype(str) + " "
        + df["description"].astype(str)
    )

    vectorizer = TfidfVectorizer(stop_words="english")
    item_vectors = vectorizer.fit_transform(df["content_profile"])
    return vectorizer, item_vectors


def recommend_by_title(title, df, item_vectors, top_n=6):
    matches = df[df["title"].str.lower() == title.strip().lower()]
    if matches.empty:
        return None

    index = matches.index[0]
    similarities = cosine_similarity(item_vectors[index], item_vectors)[0]

    ranked = sorted(
        [(i, score) for i, score in enumerate(similarities) if i != index],
        key=lambda x: x[1],
        reverse=True,
    )[:top_n]

    results = []
    for i, score in ranked:
        row = df.loc[i]
        results.append({
            "title": row["title"],
            "category": row["category"],
            "level": row["level"],
            "duration_hours": int(row["duration_hours"]),
            "rating": float(row["rating"]),
            "description": row["description"],
            "score": round(float(score) * 100, 1),
        })
    return results
def recommend_by_preferences(preference_text, df, vectorizer, item_vectors, top_n=6):
    preference_vector = vectorizer.transform([preference_text])
    similarities = cosine_similarity(preference_vector, item_vectors)[0]

    ranked = sorted(enumerate(similarities), key=lambda x: x[1], reverse=True)[:top_n]

    results = []
    for i, score in ranked:
        if score <= 0:
            continue
        row = df.loc[i]
        results.append({
            "title": row["title"],
            "category": row["category"],
            "level": row["level"],
            "duration_hours": int(row["duration_hours"]),
            "rating": float(row["rating"]),
            "description": row["description"],
            "score": round(float(score) * 100, 1),
        })
    return results
