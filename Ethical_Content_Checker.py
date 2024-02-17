import re
import nltk

# Specify the data directory where NLTK should look for resources
nltk.data.path.append("C:/Users/User/nltk_data")

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB

# Update the path to the "english" subfolder within the "stopwords" dataset
stopwords_path = nltk.data.find("corpora/stopwords/english")

# Sample dataset of text and labels (0 for Ethical, 1 for Harmful)
data = [
    ("This is a helpful and positive message.", 0),
    ("I hate you! You're a terrible person.", 1),
    ("Let's spread love and positivity.", 0),
    ("This is a test message.", 0),
    ("You should die!", 1),
]

# Data preprocessing
from nltk.corpus import stopwords

def preprocess_text(text):
    text = text.lower()
    text = re.sub(r'[^a-zA-Z]', ' ', text)
    words = text.split()
    words = [word for word in words if word not in stopwords.words("english")]
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

# Test the content checker
text_to_check = input("Enter Text: ")
result = check_content(text_to_check)
print(f"Input text is: {text_to_check}")
print(f"Content is classified as: {result}")
print(f"NLTK is using the following path: {stopwords_path}")