import pandas as pd
import zipfile
import nltk
import re
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

# Download NLTK resources
nltk.download('punkt')
nltk.download('stopwords')

# Load the zipped dataset
zip_path = r"C:\Users\sreej\Downloads\Tweets.csv.zip"

with zipfile.ZipFile(zip_path, 'r') as z:
    csv_filename = [name for name in z.namelist() if name.endswith('.csv')][0]
    with z.open(csv_filename) as f:
        df = pd.read_csv(f)

# Check column names
print("Columns:", df.columns.tolist())
print(df[['airline_sentiment', 'text']].head())

# Keep only sentiment and text
df = df[['airline_sentiment', 'text']]
df = df[df['airline_sentiment'].isin(['positive', 'neutral', 'negative'])]  # optional: filter

# Rename columns
df.columns = ['sentiment', 'text']

# Text preprocessing
def preprocess_text(text):
    text = text.lower()
    text = re.sub(r"http\S+|@\S+|[^a-zA-Z\s]", "", text)  # remove URLs, mentions, punctuation
    tokens = word_tokenize(text)
    tokens = [word for word in tokens if word not in stopwords.words('english') and word.isalpha()]
    return " ".join(tokens)

df['processed_text'] = df['text'].apply(preprocess_text)

# Train/test split
X = df['processed_text']
y = df['sentiment']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# TF-IDF Vectorization
vectorizer = TfidfVectorizer(max_features=5000)
X_train_vec = vectorizer.fit_transform(X_train)
X_test_vec = vectorizer.transform(X_test)

# Naive Bayes model
model = MultinomialNB()
model.fit(X_train_vec, y_train)

# Prediction
y_pred = model.predict(X_test_vec)

# Evaluation
print(f"\nAccuracy: {accuracy_score(y_test, y_pred)*100:.2f}%")
print("\nClassification Report:")
print(classification_report(y_test, y_pred))

# Confusion matrix
cm = confusion_matrix(y_test, y_pred, labels=['negative', 'neutral', 'positive'])
plt.figure(figsize=(6, 5))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
            xticklabels=['negative', 'neutral', 'positive'],
            yticklabels=['negative', 'neutral', 'positive'])
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.title("Confusion Matrix")
plt.show()
