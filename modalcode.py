# Import necessary libraries
import pandas as pd
import numpy as np
import re
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from imblearn.over_sampling import SMOTE
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

# Download NLTK stopwords and wordnet if not already installed
nltk.download('stopwords')
nltk.download('wordnet')

# Load the dataset
df = pd.read_csv('spam.csv', encoding='ISO-8859-1')

# Drop unnecessary columns and rename for easier access
df = df[['v1', 'v2']]  # Only keep label and text
df.columns = ['label', 'text']  # Rename columns
df['label'] = df['label'].map({'ham': 0, 'spam': 1})  # Convert labels to binary (ham=0, spam=1)

# Initialize Lemmatizer and stopwords
lemmatizer = WordNetLemmatizer()
stop_words = set(stopwords.words('english'))

# Function to preprocess text: remove special characters, lowercase, remove stopwords, and lemmatize
def preprocess_text(text):
    text = re.sub(r'\W', ' ', text)  # Remove special characters
    text = text.lower()  # Convert to lowercase
    text = text.split()  # Tokenize
    text = [lemmatizer.lemmatize(word) for word in text if word not in stop_words]  # Lemmatize and remove stopwords
    text = ' '.join(text)  # Join tokens back into a string
    return text

# Apply text preprocessing
df['text'] = df['text'].apply(preprocess_text)

# Apply TF-IDF Vectorizer to convert text data into numerical features
vectorizer = TfidfVectorizer(max_features=5000)
X = vectorizer.fit_transform(df['text']).toarray()  # Convert text data to feature matrix
y = df['label']  # Target labels

# Apply SMOTE to handle class imbalance
smote = SMOTE(random_state=42)
X_sm, y_sm = smote.fit_resample(X, y)

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X_sm, y_sm, test_size=0.2, random_state=42)

# Train the Naive Bayes model
model = MultinomialNB()
model.fit(X_train, y_train)

# Make predictions on the test set
y_pred = model.predict(X_test)

# Evaluate the model
accuracy = accuracy_score(y_test, y_pred)
print(f"Accuracy: {accuracy * 100:.2f}%")

# Display confusion matrix and classification report
print("Confusion Matrix:")
print(confusion_matrix(y_test, y_pred))

print("\nClassification Report:")
print(classification_report(y_test, y_pred))

# Function to predict if a user-entered message is spam or ham
def predict_spam_or_ham(user_input):
    user_input_cleaned = preprocess_text(user_input)  # Preprocess the input
    user_input_vectorized = vectorizer.transform([user_input_cleaned]).toarray()  # Vectorize the input
    prediction = model.predict(user_input_vectorized)  # Predict using the trained model
    return "Spam" if prediction[0] == 1 else "Ham"

# Get user input and predict
user_input = input("Enter a message to check if it's spam or ham: ")
result = predict_spam_or_ham(user_input)
print(f"The entered message is: {result}")
