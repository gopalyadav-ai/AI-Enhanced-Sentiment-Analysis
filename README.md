# AI Enhanced Sentiment Analysis of Text Data

## Project Overview

This project is about sentiment analysis of text reviews using **NLP, Machine Learning, and Deep Learning**.

The main purpose of the project is to take a review as input and predict whether it is **Positive** or **Negative**.

I worked on the complete process starting from text preprocessing and model training to model evaluation and deployment using Streamlit. I also compared different Machine Learning and Deep Learning models to see how they perform on the same dataset.

---

## Project Objective

The main objectives of this project are:

* Preprocess text data using NLP techniques.
* Convert text into a format that Machine Learning and Deep Learning models can use.
* Train different Machine Learning models for sentiment classification.
* Train different Deep Learning models for the same task.
* Compare the performance of all the models.
* Save the trained models for later use.
* Build a Streamlit application for making predictions.
* Store prediction history using SQLite.

---

## Key Features

* Text preprocessing using NLP.
* TF-IDF based Machine Learning models.
* Tokenizer and padded sequences for Deep Learning models.
* Five different models for comparison.
* Positive/Negative sentiment prediction.
* Interactive Streamlit application.
* SQLite database for storing prediction history.
* Saved trained models for use in the application.

---

## Dataset

The project uses a sentiment review dataset containing **999 reviews** and **3 columns**.

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

The dataset is almost balanced, with 500 positive reviews and 499 negative reviews.

Each review has a corresponding sentiment label that is used for training and testing the models.

---

## NLP Preprocessing

Before training the models, the review text was cleaned and preprocessed.

The following steps were used:

1. Convert text to lowercase.
2. Remove punctuation.
3. Remove numerical characters.
4. Remove extra spaces.
5. Tokenize the text.
6. Remove English stopwords.
7. Perform lemmatization.

### Text Representation

Different representations were used for the Machine Learning and Deep Learning models.

**Machine Learning models:**

* TF-IDF Vectorization
* Maximum features: 5000

**Deep Learning models:**

* Keras Tokenizer
* Convert text into sequences
* Padding
* Maximum sequence length: 100

---

## Machine Learning Models

I used two traditional Machine Learning models for sentiment classification.

### Logistic Regression

Logistic Regression was used as one of the baseline classification models to predict whether a review is positive or negative.

### Multinomial Naive Bayes

Multinomial Naive Bayes is commonly used for text classification. It was used with the TF-IDF features generated from the review text.

---

## Deep Learning Models

I also tested three recurrent neural network models.

### Simple RNN

A Simple Recurrent Neural Network was used to process the review as a sequence of words.

### Bidirectional RNN

The Bidirectional RNN processes the sequence in both directions. This allows the model to use information from both the previous and following parts of the sequence.

### GRU

A Gated Recurrent Unit (GRU) was also used for sentiment classification. GRU is another type of recurrent neural network that can handle sequential data.

### Common Deep Learning Architecture

The Deep Learning models used the following basic setup:

* Embedding layer
* 64-dimensional embedding representation
* Recurrent layer with 64 units
* Dense output layer
* Sigmoid activation for binary classification
* 5 training epochs
* Batch size of 32

---

## Model Performance

The five models were tested and their accuracy was compared.

| Model               |  Accuracy |
| ------------------- | --------: |
| Logistic Regression |     77.5% |
| **Naive Bayes**     | **78.0%** |
| Simple RNN          |     63.0% |
| Bidirectional RNN   |     59.5% |
| GRU                 |     76.5% |

### Best Model

**Multinomial Naive Bayes achieved the highest test accuracy of 78.0%.**

GRU was the second-best model with an accuracy of 76.5%.

One interesting result from this project was that the Deep Learning models did not automatically perform better than the traditional Machine Learning models. On this dataset, Naive Bayes gave the best result.

---

## Model Evaluation

For the Machine Learning models, I used:

* Accuracy
* Precision
* Recall
* F1-score
* Confusion Matrix
* Classification Report

For the Deep Learning models, I also looked at:

* Training accuracy
* Validation accuracy
* Binary cross-entropy loss

These metrics helped me compare the performance of the different models.

---

## Technologies Used

### Programming

* Python

### Data Processing

* Pandas
* NumPy

### NLP

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

### Model Saving

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
     +-----------------------+
     |                       |
     v                       v
Machine Learning       Deep Learning
     |                       |
   TF-IDF              Tokenization
     |                       |
     |                   Padding
     |                       |
  +--+------+          +-----+-----+-----+
  |         |          |           |     |
  v         v          v           v     v
Logistic  Naive       RNN        BiRNN   GRU
Regression Bayes
  |         |          |           |     |
  +---------+----------+-----------+-----+
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

| File                   | Purpose                                        |
| ---------------------- | ---------------------------------------------- |
| `app.py`               | Streamlit application                          |
| `.ipynb`               | Project notebook containing the implementation |
| `lr_model.pkl`         | Saved Logistic Regression model                |
| `nb_model.pkl`         | Saved Naive Bayes model                        |
| `tfidf.pkl`            | Saved TF-IDF vectorizer                        |
| `tokenizer.pkl`        | Saved Keras tokenizer                          |
| `rnn_model.h5`         | Saved Simple RNN model                         |
| `birnn_model.h5`       | Saved Bidirectional RNN model                  |
| `gru_model.h5`         | Saved GRU model                                |
| `sentiment_reviews.db` | SQLite database used for prediction history    |
| `requirements.txt`     | Python dependencies                            |

---

## Installation

Clone the repository:

```bash
git clone https://github.com/gopalyadav-ai/AI-Enhanced-Sentiment-Analysis.git
```

Go to the project folder:

```bash
cd AI-Enhanced-Sentiment-Analysis
```

Install the required libraries:

```bash
pip install -r requirements.txt
```

---

## How to Run

To start the Streamlit application, run:

```bash
python -m streamlit run app.py
```

After running the command, Streamlit will provide a local URL. Open that URL in a browser to use the application.

---

## Streamlit Application

The application allows the user to:

1. Enter a review.
2. Select a model.
3. Click **Predict Sentiment**.
4. Get the predicted sentiment.
5. Save the prediction in the database.
6. View previously stored predictions.

### Available Models

* Logistic Regression
* Naive Bayes
* Simple RNN
* Bidirectional RNN
* GRU

The application uses different preprocessing depending on the selected model.

For the Machine Learning models, the review is converted into TF-IDF features using the saved TF-IDF vectorizer.

For the Deep Learning models, the review is converted into a sequence using the saved Keras tokenizer and then padded to a maximum length of 100.

---

## SQLite Database

The application uses **SQLite** to store the prediction history.

The database contains a `reviews` table.

| Column      | Description                       |
| ----------- | --------------------------------- |
| `id`        | Automatically generated record ID |
| `review`    | Review entered by the user        |
| `model`     | Model selected for prediction     |
| `sentiment` | Predicted sentiment               |

The application also has a **Show Stored Reviews** option to display the previous predictions stored in the database.

---

## Results & Conclusion

In this project, I compared **five different models** for sentiment classification.

The results were:

* Logistic Regression: **77.5%**
* Naive Bayes: **78.0%**
* Simple RNN: **63.0%**
* Bidirectional RNN: **59.5%**
* GRU: **76.5%**

Naive Bayes gave the best test accuracy of **78.0%** on this dataset.

The main thing I learned from the comparison was that a more complex Deep Learning model does not always give better results. For this particular dataset, the traditional Naive Bayes model performed better than the Deep Learning models.

The project also helped me understand the complete process of building a text classification application:

```text
Data
  ↓
Text Preprocessing
  ↓
Feature Representation
  ↓
Model Training
  ↓
Model Evaluation
  ↓
Model Comparison
  ↓
Saving Trained Models
  ↓
Streamlit Application
  ↓
SQLite Database
```

---

## Future Improvements

Some improvements I would like to make in the future are:

* Use a larger and more diverse dataset.
* Perform more hyperparameter tuning.
* Try Transformer-based models such as BERT.
* Add prediction confidence scores.
* Add timestamps to the prediction history.
* Improve the prediction-history interface.
* Deploy the application on a cloud platform.
* Compare the existing models with modern Transformer-based models.

---

## Author

**Gopal Yadav**
