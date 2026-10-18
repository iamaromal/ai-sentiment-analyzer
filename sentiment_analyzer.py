import re
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline

class SentimentAnalyzer:
    def __init__(self):
        self.pipeline = Pipeline([
            ('tfidf', TfidfVectorizer(ngram_range=(1, 2))),
            ('clf', MultinomialNB())
        ])
        self.is_trained = False

    def preprocess(self, text):
        text = text.lower()
        text = re.sub(r'[^a-zA-Z\s]', '', text)
        return text

    def train(self, texts, labels):
        cleaned_texts = [self.preprocess(t) for t in texts]
        self.pipeline.fit(cleaned_texts, labels)
        self.is_trained = True

    def predict(self, texts):
        if not self.is_trained:
            raise ValueError("Model is not trained yet.")
        cleaned_texts = [self.preprocess(t) for t in texts]
        return self.pipeline.predict(cleaned_texts)

if __name__ == "__main__":
    train_data = [
        ("I love this AI tool, it is amazing and helpful!", "Positive"),
        ("This is wonderful, accurate and fast sentiment analysis.", "Positive"),
        ("Great performance and awesome results.", "Positive"),
        ("I hate bugs and errors in code.", "Negative"),
        ("This product is terrible, slow, and useless.", "Negative"),
        ("Very bad experience and terrible customer support.", "Negative")
    ]
    
    texts, labels = zip(*train_data)
    analyzer = SentimentAnalyzer()
    analyzer.train(texts, labels)
    
    test_samples = [
        "The AI sentiment analyzer works great and is super helpful!",
        "This code has errors and performs very badly."
    ]
    
    predictions = analyzer.predict(test_samples)
    for sample, pred in zip(test_samples, predictions):
        print(f"Text: '{sample}' -> Sentiment: {pred}")
