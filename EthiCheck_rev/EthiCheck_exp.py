import re
import spacy
from textblob import TextBlob

# Load the English spaCy model
nlp = spacy.load("en_core_web_sm")

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB

# Sample dataset of text and labels (0 for Ethical, 1 for Harmful)
data = []

file_path = r"C:\Users\User\Desktop\CSE 332\Patent\data.txt"
# Open the file for reading
with open(file_path, "r") as file:
    for line in file:
        parts = line.strip().split(",")
        if len(parts) == 2:
            text, label = parts[0], int(parts[1])
            data.append((text, label))

# Data preprocessing
def preprocess_text(text):
    text = text.lower()
    text = re.sub(r'[^a-zA-Z]', ' ', text)
    words = [token.text for token in nlp(text) if not token.is_stop]
    return " ".join(words)

corpus = [preprocess_text(text) for text, _ in data]
labels = [label for _, label in data]

# Vectorize the text data
vectorizer = TfidfVectorizer()
X = vectorizer.fit_transform(corpus)

# Train a simple text classifier
classifier = MultinomialNB()
classifier.fit(X, labels)

# Function to check text content
def check_content(text):
    preprocessed_text = preprocess_text(text)
    vectorized_text = vectorizer.transform([preprocessed_text])
    prediction = classifier.predict(vectorized_text)
    if prediction[0] == 0:
        return "Ethical"
    else:
        return "Potentially Harmful"

# Analyze text sentiment using TextBlob
def analyze_sentiment(text):
    blob = TextBlob(text)
    polarity = blob.sentiment.polarity
    subjectivity = blob.sentiment.subjectivity
    if polarity > 0:
        return "Positive"
    elif polarity < 0:
        return "Negative"
    else:
        return "Neutral"

# Test the content checker and sentiment analysis
text_to_check = input("Enter Text: ")
content_result = check_content(text_to_check)
sentiment_result = analyze_sentiment(text_to_check)
print(f"Content is classified as: {content_result}")
print(f"Sentiment is: {sentiment_result}")
