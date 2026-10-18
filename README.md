# AI Sentiment Analyzer
A beginner-friendly Python project that uses Artificial Intelligence to understand the emotion behind a piece of text.

## 📝 What is this project?
This tool acts like an automated reader. It takes raw text (like a product review or a comment) and categorizes its emotional tone into one of three buckets:
* **Positive:** Expressing happiness or praise
* **Negative:** Expressing disappointment or anger
* **Neutral:** Factual statements without strong emotion

## 🧠 How does it work?
This project uses Natural Language Processing (NLP). Here is the 3-step process:
1. **Cleaning up:** We remove extra noise like punctuation and stop words.
2. **Math Magic (TF-IDF):** We use a formula called TF-IDF to convert words into numerical scores.
3. **The Judge (Classification):** A machine learning model looks at these numbers and makes a final decision.

## 🛠️ Tools Used
* **Python**, **Scikit-learn**, **NLTK**, **Pandas**

## 🚀 How to Run It
1. git clone url?id=2.git
2. pip install scikit-learn nltk pandas
3. python sentiment_analyzer.py
