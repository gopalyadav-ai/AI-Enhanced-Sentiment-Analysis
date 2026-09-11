import streamlit as st
import pickle
import numpy as np
import sqlite3

from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences

# ---------------- DATABASE ----------------

conn = sqlite3.connect("sentiment_reviews.db", check_same_thread=False)
cursor = conn.cursor()

cursor.execute("""
CREATE TABLE IF NOT EXISTS reviews (
id INTEGER PRIMARY KEY AUTOINCREMENT,
review TEXT,
model TEXT,
sentiment TEXT
)
""")

conn.commit()

# ---------------- LOAD MODELS ----------------

# Load ML models
lr = pickle.load(open("lr_model.pkl","rb"))
nb = pickle.load(open("nb_model.pkl","rb"))
tfidf = pickle.load(open("tfidf.pkl","rb"))

# Load DL models
rnn = load_model("rnn_model.h5")
birnn = load_model("birnn_model.h5")
gru = load_model("gru_model.h5")

# Load tokenizer
tokenizer = pickle.load(open("tokenizer.pkl","rb"))

max_length = 100

# ---------------- UI ----------------

st.title("AI Enhanced Sentiment Analysis System")

st.write("Enter a review and choose model")

# Text input
review = st.text_area("Enter Review")

# Model selection
model_choice = st.selectbox(
    "Select Model",
    ("Logistic Regression",
     "Naive Bayes",
     "Simple RNN",
     "Bidirectional RNN",
     "GRU")
)

# ---------------- PREDICTION ----------------

if st.button("Predict Sentiment"):

    if model_choice in ["Logistic Regression","Naive Bayes"]:

        vec = tfidf.transform([review])

        if model_choice == "Logistic Regression":
            pred = lr.predict(vec)

        else:
            pred = nb.predict(vec)

    else:

        seq = tokenizer.texts_to_sequences([review])
        padded = pad_sequences(seq, maxlen=max_length)

        if model_choice == "Simple RNN":
            pred = rnn.predict(padded)

        elif model_choice == "Bidirectional RNN":
            pred = birnn.predict(padded)

        else:
            pred = gru.predict(padded)

        pred = (pred > 0.5)

    if pred[0] == 1:
        sentiment = "Positive"
        st.success("Positive Review 😊")
    else:
        sentiment = "Negative"
        st.error("Negative Review 😠")

    # ---------------- SAVE TO DATABASE ----------------

    cursor.execute(
        "INSERT INTO reviews (review, model, sentiment) VALUES (?, ?, ?)",
        (review, model_choice, sentiment)
    )

    conn.commit()
    st.subheader("Stored Review History")

if st.button("Show Stored Reviews"):

    cursor.execute("SELECT * FROM reviews")

    rows = cursor.fetchall()

    for row in rows:
        st.write(row)