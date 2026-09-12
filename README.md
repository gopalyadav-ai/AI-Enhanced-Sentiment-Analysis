# AI Enhanced Sentiment Analysis of Text Data

## Project Overview

This project focuses on sentiment analysis of text reviews using Natural Language Processing (NLP), Machine Learning, and Deep Learning techniques.

The main goal is to classify a given review as either **Positive** or **Negative** and compare the performance of different Machine Learning and Deep Learning approaches.

I independently implemented the data preprocessing, NLP pipeline, model training, evaluation, model comparison, model serialization, and Streamlit deployment, with guidance from my faculty mentor.

---

## Project Objective

The objectives of this project are to:

* Perform text preprocessing using NLP techniques.
* Convert textual data into numerical representations.
* Build and evaluate Machine Learning models.
* Build and evaluate Deep Learning models.
* Compare different approaches using accuracy and other evaluation metrics.
* Deploy the trained models through a Streamlit web application.
* Store prediction history using SQLite.

---

## Key Features

* Text preprocessing using NLP techniques.
* TF-IDF based Machine Learning models.
* Tokenizer and sequence-based Deep Learning models.
* Comparison of five different models.
* Interactive Streamlit application.
* Positive/Negative sentiment prediction.
* SQLite database for storing prediction history.
* Multiple trained models available for comparison.

---

## Dataset

The project uses a sentiment review dataset containing **999 reviews**.

### Dataset Details

| Property         |       Value |
| ---------------- | ----------: |
| Total records    |         999 |
| Total columns    |           3 |
| Positive reviews |         500 |
| Negative reviews |         499 |
| Target variable  | `sentiment` |
| Positive label   |           1 |
| Negative label   |           0 |

The dataset is nearly balanced between positive and negative reviews.

The dataset contains the review text and its corresponding sentiment label.

---

## NLP Preprocessing

Text data cannot be directly provided to most Machine Learning models, so several preprocessing steps were performed.

### Preprocessing Pipeline

1. Convert text to lowercase.
2. Remove punctuation.
3. Remove numerical characters.
4. Remove extra whitespace.
5. Tokenize the text.
6. Remove English stopwords.
7. Perform lemmatization.

### Feature Representation

Two different approaches were used.

**For Machine Learning models:**

* TF-IDF Vectorization
* Maximum features: 5000

**For Deep Learning models:**

* Keras Tokenizer
* Sequence conversion
* Padding
* Maximum sequence length: 100

---

## Machine Learning Models

Two classical Machine Learning algorithms were implemented.

### 1. Logistic Regression

Logistic Regression was used as a classification model to predict whether a review belongs to the positive or negative sentiment class.

### 2. Multinomial Naive Bayes

Multinomial Naive Bayes is commonly used for text classification because it works well with text features such as TF-IDF representations.

---

## Deep Learning Models

Three Deep Learning architectures were implemented.

### 1. Simple RNN

A Simple Recurrent Neural Network was used to process sequential text data.

### 2. Bidirectional RNN

A Bidirectional RNN processes the sequence in both forward and backward directions, allowing the model to use information from both directions of the sequence.

### 3. GRU

A Gated Recurrent Unit (GRU) was implemented as another recurrent architecture for sequence-based sentiment classification.

### Common Architecture

The Deep Learning models use:

* Embedding layer
* 64-dimensional embedding representation
* Recurrent layer with 64 units
* Dense output layer
* Sigmoid activation for binary classification
* 5 training epochs
* Batch size of 32

---

## Model Performance & Comparison

The models were evaluated using test data.

| Model               |  Accuracy |
| ------------------- | --------: |
| Logistic Regression |     77.5% |
| **Naive Bayes**     | **78.0%** |
| Simple RNN          |     63.0% |
| Bidirectional RNN   |     59.5% |
| GRU                 |     76.5% |

### Best Performing Model

**Multinomial Naive Bayes achieved the highest test accuracy of 78.0%.**

Although the Deep Learning models were more complex, the classical Naive Bayes model performed slightly better on this particular dataset.

This demonstrates that a more complex model does not always perform better than a simpler model, especially when the dataset is relatively small.

---

## Evaluation

For the Machine Learning models, the following evaluation techniques were used:

* Accuracy
* Precision
* Recall
* F1-score
* Confusion Matrix
* Classification Report

For the Deep Learning models, training behaviour was also analyzed using:

* Binary cross-entropy loss
* Training accuracy
* Validation accuracy

---

## Technologies Used

### Programming Language

* Python

### Data Processing

* Pandas
* NumPy

### Natural Language Processing

* NLTK
* TF-IDF
* Tokenization
* Stopword Removal
* Lemmatization

### Machine Learning

* Scikit-learn
* Logistic Regression
* Multinomial Naive Bayes

### Deep Learning

* TensorFlow
* Keras
* Simple RNN
* Bidirectional RNN
* GRU
* Embedding

### Visualization

* Matplotlib
* Seaborn
* WordCloud

### Deployment

* Streamlit

### Database

* SQLite

### Model Serialization

* Pickle
* Keras HDF5 (`.h5`)

---

## Project Workflow

```text
                    Input Review
                         |
                         v
                Text Preprocessing
                         |
          +--------------+--------------+
          |                             |
          v                             v
     Machine Learning             Deep Learning
          |                             |
       TF-IDF                    Tokenization
          |                             |
    +-----+------+              Padding to 100
    |            |                     |
    v            v              +------+------+------+
Logistic      Naive Bayes       |      |             |
Regression                    RNN   BiRNN          GRU
    |            |              |      |             |
    +------------+--------------+------+-------------+
                         |
                         v
                  Sentiment Result
                  Positive / Negative
                         |
                         v
                   SQLite Database
```

---

## Project Structure

```text
AI-Enhanced-Sentiment-Analysis/
│
├── app.py
├── requirements.txt
│
├── AI- Enhanced Sentiment Analysis of the Text data Using NLP, Deep Learning and Machine Learning- A Comparative Study (1).ipynb
│
├── lr_model.pkl
├── nb_model.pkl
├── tfidf.pkl
├── tokenizer.pkl
│
├── rnn_model.h5
├── birnn_model.h5
├── gru_model.h5
│
└── sentiment_reviews.db
```

### File Description

| File                   | Purpose                                       |
| ---------------------- | --------------------------------------------- |
| `app.py`               | Streamlit application                         |
| `.ipynb`               | Complete project notebook                     |
| `lr_model.pkl`         | Trained Logistic Regression model             |
| `nb_model.pkl`         | Trained Naive Bayes model                     |
| `tfidf.pkl`            | Saved TF-IDF vectorizer                       |
| `tokenizer.pkl`        | Saved Keras tokenizer                         |
| `rnn_model.h5`         | Trained Simple RNN                            |
| `birnn_model.h5`       | Trained Bidirectional RNN                     |
| `gru_model.h5`         | Trained GRU model                             |
| `sentiment_reviews.db` | SQLite database containing prediction history |
| `requirements.txt`     | Python dependencies                           |

---

## Installation

Clone the repository:

```bash
git clone https://github.com/gopalyadav-ai/AI-Enhanced-Sentiment-Analysis.git
```

Move into the project directory:

```bash
cd AI-Enhanced-Sentiment-Analysis
```

Install the required Python packages:

```bash
pip install -r requirements.txt
```

---

## How to Run the Application

Run the Streamlit application using:

```bash
python -m streamlit run app.py
```

After running the command, Streamlit will provide a local URL where the application can be opened in a web browser.

---

## Streamlit Application

The application provides an interactive interface where the user can:

1. Enter a text review.
2. Select one of the five available models.
3. Click **Predict Sentiment**.
4. View the predicted sentiment.
5. Store the prediction in the SQLite database.
6. View previously stored reviews.

### Available Models

* Logistic Regression
* Naive Bayes
* Simple RNN
* Bidirectional RNN
* GRU

The application uses the appropriate preprocessing pipeline depending on the selected model.

For Machine Learning models, the review is transformed using the saved TF-IDF vectorizer.

For Deep Learning models, the review is converted into sequences using the saved tokenizer and padded to a maximum length of 100.

---

## SQLite Database

The application uses SQLite to store prediction history.

The database contains a `reviews` table with the following fields:

| Column      | Description                       |
| ----------- | --------------------------------- |
| `id`        | Automatically generated record ID |
| `review`    | User-entered review               |
| `model`     | Model used for prediction         |
| `sentiment` | Predicted sentiment               |

The application also provides a **Show Stored Reviews** option to retrieve and display the stored prediction history.

---

## Results & Conclusion

Five different approaches were implemented and compared.

The results showed that **Multinomial Naive Bayes achieved the highest accuracy of 78.0%**, followed by GRU with 76.5%.

The experiment shows that traditional Machine Learning methods can perform competitively with Deep Learning models on a relatively small text dataset.

The project demonstrates the complete workflow of a text classification system:

```text
Data
  ↓
NLP Preprocessing
  ↓
Feature Representation
  ↓
Model Training
  ↓
Model Evaluation
  ↓
Model Comparison
  ↓
Model Serialization
  ↓
Streamlit Deployment
  ↓
SQLite Storage
```

---

## Future Improvements

The project can be further improved by:

* Using a larger and more diverse dataset.
* Performing hyperparameter tuning.
* Experimenting with Transformer-based models such as BERT.
* Adding prediction confidence scores.
* Adding timestamps to prediction history.
* Improving the review-history interface.
* Deploying the application to a cloud platform.
* Comparing the current models with modern Transformer-based approaches.

---

## Author

**Gopal Yadav**
