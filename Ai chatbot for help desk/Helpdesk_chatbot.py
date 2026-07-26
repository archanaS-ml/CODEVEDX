"""
AI Chatbot for Internal Helpdesk
-----------------------------------
Internship Project 4 - CodeVedX

Requirements:
  pip install flask pandas scikit-learn

Data model (faq_data.csv):
  id, question, answer
"""

import os
import pandas as pd
from flask import Flask, request, render_template_string, redirect, url_for
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

FAQ_FILE = "faq_data.csv"
SIMILARITY_THRESHOLD = 0.3  # below this, the bot admits it doesn't know

app = Flask(__name__)


def create_sample_faq_data():
    if os.path.exists(FAQ_FILE):
        return

    faqs = [
        ("How do I reset my password?",
         "Go to the login page and click 'Forgot Password' to reset it via email."),
        ("How do I request leave?",
         "Submit a leave request through the HR portal under the 'Leave' tab."),
        ("Who do I contact for IT support?",
         "Email it-support@codevedx.in or raise a ticket on the internal helpdesk portal."),
        ("What are the office working hours?",
         "Office hours are 9:30 AM to 6:30 PM, Monday to Friday."),
        ("How do I connect to the company VPN?",
         "Install the VPN client from the IT portal and log in with your company credentials."),
        ("How do I raise a reimbursement request?",
         "Submit your bills through the Finance portal under 'Reimbursements'."),
        ("How can I update my personal details?",
         "Go to the HR portal, open 'My Profile', and edit your details there."),
        ("Who is my reporting manager?",
         "You can check your reporting manager under 'My Profile' in the HR portal."),
    ]

    df = pd.DataFrame({
        "id": range(1, len(faqs) + 1),
        "question": [q for q, a in faqs],
        "answer": [a for q, a in faqs],
    })
    df.to_csv(FAQ_FILE, index=False)


def load_faq_data():
    try:
        df = pd.read_csv(FAQ_FILE)
        df = df.dropna(subset=["question", "answer"])
        return df
    except FileNotFoundError:
        return pd.DataFrame(columns=["id", "question", "answer"])
    except Exception as e:
        print(f"Error loading FAQ data: {e}")
        return pd.DataFrame(columns=["id", "question", "answer"])


def build_vectorizer(df):
    vectorizer = TfidfVectorizer()
    question_vectors = vectorizer.fit_transform(df["question"])
    return vectorizer, question_vectors


def get_bot_reply(user_message, df, vectorizer, question_vectors):
    try:
        if df.empty:
            return "The FAQ knowledge base is empty. Please ask an admin to add some questions."

        user_vector = vectorizer.transform([user_message])
        similarities = cosine_similarity(user_vector, question_vectors)[0]

        best_index = similarities.argmax()
        best_score = similarities[best_index]

        if best_score < SIMILARITY_THRESHOLD:
            return "Sorry, I don't have an answer for that yet. Please contact the helpdesk directly."

        return df.iloc[best_index]["answer"]
    except Exception as e:
        return f"Sorry, something went wrong while processing your question. ({e})"


create_sample_faq_data()
faq_df = load_faq_data()
vectorizer, question_vectors = build_vectorizer(faq_df)


def refresh_knowledge_base():
    global faq_df, vectorizer, question_vectors
    faq_df = load_faq_data()
    if not faq_df.empty:
        vectorizer, question_vectors = build_vectorizer(faq_df)


CHAT_PAGE = """
<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>Helpdesk Console — CodeVedX</title>
<link rel="preconnect" href="https://fonts.googleapis.com">
<link href="https://fonts.googleapis.com/css2?family=Space+Grotesk:wght@500;600;700&family=Inter:wght@400;500;600&family=JetBrains+Mono:wght@400;500&display=swap" rel="stylesheet">
<style>
    :root {
        --teal-900: #08302c;
        --teal-700: #0f5c56;
        --teal-500: #16847c;
        --teal-100: #e4f3f1;
        --amber: #f2a93b;
        --ink: #1c2624;
        --paper: #f6f8f7;
        --line: #d9e3e1;
    }
    * { box-sizing: border-box; }
    body {
        margin: 0;
        background: var(--paper);
        font-family: 'Inter', sans-serif;
        color: var(--ink);
        min-height: 100vh;
        display: flex;
        align-items: center;
        justify-content: center;
        padding: 24px;
    }
    .console {
        width: 100%;
        max-width: 620px;
        background: #fff;
        border: 1px solid var(--line);
        border-radius: 14px;
        overflow: hidden;
        box-shadow: 0 20px 50px -25px rgba(8, 48, 44, 0.35);
    }
    .console-header {
        background: linear-gradient(135deg, var(--teal-900), var(--teal-700));
        color: #fff;
        padding: 20px 24px;
        display: flex;
        align-items: center;
        justify-content: space-between;
    }
    .brand {
        display: flex;
        align-items: center;
        gap: 12px;
    }
    .brand-mark {
        width: 34px;
        height: 34px;
        border-radius: 8px;
        background: var(--amber);
        display: flex;
        align-items: center;
        justify-content: center;
        font-family: 'Space Grotesk', sans-serif;
        font-weight: 700;
        color: var(--teal-900);
        font-size: 15px;
    }
    .brand-text h1 {
        font-family: 'Space Grotesk', sans-serif;
        font-size: 16px;
        font-weight: 600;
        margin: 0;
        letter-spacing: 0.01em;
    }
    .brand-text p {
        margin: 2px 0 0;
        font-size: 12px;
        color: rgba(255,255,255,0.65);
        font-family: 'JetBrains Mono', monospace;
    }
    .status {
        display: flex;
        align-items: center;
        gap: 6px;
        font-size: 11px;
        font-family: 'JetBrains Mono', monospace;
        color: rgba(255,255,255,0.85);
        text-transform: uppercase;
        letter-spacing: 0.05em;
    }
    .status-dot {
        width: 7px;
        height: 7px;
        border-radius: 50%;
        background: #4ade80;
        box-shadow: 0 0 0 0 rgba(74, 222, 128, 0.6);
        animation: pulse 2s infinite;
    }
    @keyframes pulse {
        0% { box-shadow: 0 0 0 0 rgba(74, 222, 128, 0.55); }
        70% { box-shadow: 0 0 0 7px rgba(74, 222, 128, 0); }
        100% { box-shadow: 0 0 0 0 rgba(74, 222, 128, 0); }
    }
    #chatbox {
        height: 420px;
        overflow-y: auto;
        padding: 22px 24px;
        display: flex;
        flex-direction: column;
        gap: 16px;
        background:
            radial-gradient(circle at 1px 1px, var(--line) 1px, transparent 0) 0 0/22px 22px;
        background-color: var(--paper);
    }
    .msg-row { display: flex; flex-direction: column; max-width: 82%; animation: rise 0.25s ease; }
    @keyframes rise { from { opacity: 0; transform: translateY(6px);} to { opacity: 1; transform: translateY(0);} }
    .msg-row.user { align-self: flex-end; align-items: flex-end; }
    .msg-row.bot { align-self: flex-start; align-items: flex-start; }
    .ticket-id {
        font-family: 'JetBrains Mono', monospace;
        font-size: 10px;
        color: #8a9a97;
        margin-bottom: 4px;
        letter-spacing: 0.03em;
    }
    .bubble {
        padding: 10px 14px;
        border-radius: 12px;
        font-size: 14px;
        line-height: 1.5;
    }
    .msg-row.user .bubble {
        background: var(--teal-700);
        color: #fff;
        border-bottom-right-radius: 3px;
    }
    .msg-row.bot .bubble {
        background: #fff;
        border: 1px solid var(--line);
        color: var(--ink);
        border-bottom-left-radius: 3px;
    }
    .composer {
        display: flex;
        gap: 10px;
        padding: 16px 20px;
        border-top: 1px solid var(--line);
        background: #fff;
    }
    #userInput {
        flex: 1;
        padding: 11px 14px;
        border: 1px solid var(--line);
        border-radius: 9px;
        font-family: 'Inter', sans-serif;
        font-size: 14px;
        outline: none;
        transition: border-color 0.15s;
    }
    #userInput:focus { border-color: var(--teal-500); }
    button.send {
        background: var(--amber);
        color: var(--teal-900);
        border: none;
        border-radius: 9px;
        padding: 0 20px;
        font-family: 'Space Grotesk', sans-serif;
        font-weight: 600;
        font-size: 14px;
        cursor: pointer;
        transition: filter 0.15s;
    }
    button.send:hover { filter: brightness(1.06); }
    .footer-link {
        text-align: center;
        padding: 12px;
        font-size: 12px;
        font-family: 'JetBrains Mono', monospace;
        background: var(--teal-100);
        border-top: 1px solid var(--line);
    }
    .footer-link a { color: var(--teal-700); text-decoration: none; }
    .footer-link a:hover { text-decoration: underline; }
</style>
</head>
<body>
    <div class="console">
        <div class="console-header">
            <div class="brand">
                <div class="brand-mark">CV</div>
                <div class="brand-text">
                    <h1>Helpdesk Console</h1>
                    <p>internal-support / codevedx</p>
                </div>
            </div>
            <div class="status"><span class="status-dot"></span>online</div>
        </div>

        <div id="chatbox"></div>

        <div class="composer">
            <input type="text" id="userInput" placeholder="Describe your issue or question..." autocomplete="off">
            <button class="send" onclick="sendMessage()">Send</button>
        </div>

        <div class="footer-link"><a href="/admin">→ Admin: manage knowledge base</a></div>
    </div>

    <script>
        let ticketCount = 0;

        function nextTicketId() {
            ticketCount += 1;
            return "HD-" + String(ticketCount).padStart(4, "0");
        }

        function appendMessage(role, text) {
            const chatbox = document.getElementById("chatbox");
            const row = document.createElement("div");
            row.className = "msg-row " + role;

            const idTag = document.createElement("div");
            idTag.className = "ticket-id";
            idTag.textContent = "#" + nextTicketId() + (role === "user" ? " · you" : " · bot");

            const bubble = document.createElement("div");
            bubble.className = "bubble";
            bubble.textContent = text;

            row.appendChild(idTag);
            row.appendChild(bubble);
            chatbox.appendChild(row);
            chatbox.scrollTop = chatbox.scrollHeight;
        }

        async function sendMessage() {
            const input = document.getElementById("userInput");
            const message = input.value.trim();
            if (!message) return;

            appendMessage("user", message);
            input.value = "";

            try {
                const response = await fetch("/chat", {
                    method: "POST",
                    headers: { "Content-Type": "application/json" },
                    body: JSON.stringify({ message: message })
                });
                const data = await response.json();
                appendMessage("bot", data.reply);
            } catch (err) {
                appendMessage("bot", "Connection error. Please try again.");
            }
        }

        document.getElementById("userInput").addEventListener("keydown", function(e) {
            if (e.key === "Enter") sendMessage();
        });

        appendMessage("bot", "Hi, I'm the CodeVedX internal helpdesk assistant. Ask me about IT, HR, or admin questions.");
    </script>
</body>
</html>
"""

ADMIN_PAGE = """
<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>Knowledge Base — Admin</title>
<link rel="preconnect" href="https://fonts.googleapis.com">
<link href="https://fonts.googleapis.com/css2?family=Space+Grotesk:wght@500;600;700&family=Inter:wght@400;500;600&family=JetBrains+Mono:wght@400;500&display=swap" rel="stylesheet">
<style>
    :root {
        --teal-900: #08302c;
        --teal-700: #0f5c56;
        --teal-500: #16847c;
        --teal-100: #e4f3f1;
        --amber: #f2a93b;
        --ink: #1c2624;
        --paper: #f6f8f7;
        --line: #d9e3e1;
    }
    * { box-sizing: border-box; }
    body {
        margin: 0;
        background: var(--paper);
        font-family: 'Inter', sans-serif;
        color: var(--ink);
        padding: 40px 24px;
    }
    .wrap { max-width: 760px; margin: 0 auto; }
    .top {
        display: flex;
        align-items: center;
        justify-content: space-between;
        margin-bottom: 22px;
    }
    .eyebrow {
        font-family: 'JetBrains Mono', monospace;
        font-size: 11px;
        color: var(--teal-500);
        text-transform: uppercase;
        letter-spacing: 0.08em;
        margin: 0 0 4px;
    }
    h1 {
        font-family: 'Space Grotesk', sans-serif;
        font-size: 24px;
        margin: 0;
    }
    a.back {
        font-family: 'JetBrains Mono', monospace;
        font-size: 12px;
        color: var(--teal-700);
        text-decoration: none;
        border: 1px solid var(--line);
        padding: 8px 12px;
        border-radius: 8px;
        background: #fff;
    }
    a.back:hover { border-color: var(--teal-500); }
    .card {
        background: #fff;
        border: 1px solid var(--line);
        border-radius: 12px;
        overflow: hidden;
        margin-bottom: 26px;
    }
    table { width: 100%; border-collapse: collapse; }
    th {
        text-align: left;
        font-family: 'JetBrains Mono', monospace;
        font-size: 11px;
        text-transform: uppercase;
        letter-spacing: 0.05em;
        color: #6b7d79;
        background: var(--teal-100);
        padding: 10px 16px;
    }
    td {
        padding: 12px 16px;
        border-top: 1px solid var(--line);
        font-size: 13.5px;
        vertical-align: top;
    }
    td.q { font-weight: 500; width: 42%; }
    td.a { color: #40514d; }
    .empty-row td { color: #8a9a97; font-style: italic; }
    .form-card {
        background: #fff;
        border: 1px solid var(--line);
        border-radius: 12px;
        padding: 20px;
    }
    .form-card h2 {
        font-family: 'Space Grotesk', sans-serif;
        font-size: 15px;
        margin: 0 0 14px;
    }
    label {
        display: block;
        font-family: 'JetBrains Mono', monospace;
        font-size: 11px;
        text-transform: uppercase;
        letter-spacing: 0.04em;
        color: #6b7d79;
        margin-bottom: 6px;
    }
    input[type=text] {
        width: 100%;
        padding: 10px 12px;
        border: 1px solid var(--line);
        border-radius: 8px;
        font-family: 'Inter', sans-serif;
        font-size: 14px;
        margin-bottom: 14px;
        outline: none;
    }
    input[type=text]:focus { border-color: var(--teal-500); }
    button.add {
        background: var(--teal-900);
        color: #fff;
        border: none;
        border-radius: 8px;
        padding: 11px 18px;
        font-family: 'Space Grotesk', sans-serif;
        font-weight: 600;
        font-size: 14px;
        cursor: pointer;
    }
    button.add:hover { background: var(--teal-700); }
</style>
</head>
<body>
    <div class="wrap">
        <div class="top">
            <div>
                <p class="eyebrow">codevedx / internal-support</p>
                <h1>Knowledge Base</h1>
            </div>
            <a class="back" href="/">← back to console</a>
        </div>

        <div class="card">
            <table>
                <tr><th>Question</th><th>Answer</th></tr>
                {% for row in faqs %}
                <tr><td class="q">{{ row.question }}</td><td class="a">{{ row.answer }}</td></tr>
                {% else %}
                <tr class="empty-row"><td colspan="2">No FAQs yet — add the first one below.</td></tr>
                {% endfor %}
            </table>
        </div>

        <div class="form-card">
            <h2>Add a new entry</h2>
            <form action="/admin/add" method="post">
                <label for="question">Question</label>
                <input type="text" id="question" name="question" placeholder="e.g. How do I book a meeting room?" required>
                <label for="answer">Answer</label>
                <input type="text" id="answer" name="answer" placeholder="e.g. Use the Outlook room booking system." required>
                <button class="add" type="submit">Add to knowledge base</button>
            </form>
        </div>
    </div>
</body>
</html>
"""


@app.route("/")
def home():
    return render_template_string(CHAT_PAGE)


@app.route("/chat", methods=["POST"])
def chat():
    try:
        data = request.get_json(force=True)
        user_message = data.get("message", "").strip()
        if not user_message:
            return {"reply": "Please type a question."}
        reply = get_bot_reply(user_message, faq_df, vectorizer, question_vectors)
        return {"reply": reply}
    except Exception as e:
        return {"reply": f"Error processing request: {e}"}


@app.route("/admin")
def admin():
    return render_template_string(ADMIN_PAGE, faqs=faq_df.to_dict(orient="records"))


@app.route("/admin/add", methods=["POST"])
def admin_add():
    try:
        question = request.form.get("question", "").strip()
        answer = request.form.get("answer", "").strip()

        if not question or not answer:
            return "Both question and answer are required.", 400

        df = load_faq_data()
        new_id = (df["id"].max() + 1) if not df.empty else 1
        new_row = pd.DataFrame([{"id": new_id, "question": question, "answer": answer}])
        df = pd.concat([df, new_row], ignore_index=True)
        df.to_csv(FAQ_FILE, index=False)

        refresh_knowledge_base()
    except Exception as e:
        return f"Error adding FAQ: {e}", 500

    return redirect(url_for("admin"))


if __name__ == "__main__":
    app.run(debug=True)