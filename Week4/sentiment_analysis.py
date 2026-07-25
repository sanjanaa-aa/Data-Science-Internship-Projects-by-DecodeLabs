import pandas as pd
import nltk
from nltk.corpus import stopwords
from sklearn.model_selection import train_test_split
from nltk.stem import WordNetLemmatizer
from sklearn.feature_extraction.text import TfidfVectorizer
from nltk.tokenize import TweetTokenizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report

nltk.download("stopwords")
nltk.download("wordnet")

# Amazon Reviews
amazon = pd.read_csv(
    "sentiment+labelled+sentences/sentiment labelled sentences/amazon_cells_labelled.txt",
    sep="\t",
    header=None
)

# IMDb Reviews
imdb = pd.read_csv(
    "sentiment+labelled+sentences/sentiment labelled sentences/imdb_labelled.txt",
    sep="\t",
    header=None
)

# Yelp Reviews
yelp = pd.read_csv(
    "sentiment+labelled+sentences/sentiment labelled sentences/yelp_labelled.txt",
    sep="\t",
    header=None
)

# Column Names
amazon.columns = ["Review", "Sentiment"]
imdb.columns = ["Review", "Sentiment"]
yelp.columns = ["Review", "Sentiment"]

# Combine Datasets
df = pd.concat(
    [amazon, imdb, yelp],
    ignore_index=True
)

print(df.head())

print("\nShape:")
print(df.shape)

print("\nColumns:")
print(df.columns)

print("\nClass Distribution:")
print(df["Sentiment"].value_counts())

print("\nMissing Values:")
print(df.isnull().sum())

# Sample Review
sample_review = df["Review"][0]

print("\nSample Review:")
print(sample_review)

# Tokenization
tokenizer = TweetTokenizer()
tokens = tokenizer.tokenize(sample_review)

print("\nTokens:")
print(tokens)
stop_words = set(stopwords.words("english"))

filtered_tokens = [
    word for word in tokens
    if word.lower() not in stop_words
]

print("\nTokens After Stop-word Removal:")
print(filtered_tokens)
lemmatizer = WordNetLemmatizer()

lemmatized_tokens = [
    lemmatizer.lemmatize(word)
    for word in filtered_tokens
]

print("\nLemmatized Tokens:")
print(lemmatized_tokens)
vectorizer = TfidfVectorizer()

X = vectorizer.fit_transform(df["Review"])

print("\nTF-IDF Shape:")
print(X.shape)
y = df["Sentiment"]

print("\nTarget Shape:")
print(y.shape)
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)

print("\nTraining Data Shape:")
print(X_train.shape)

print("\nTesting Data Shape:")
print(X_test.shape)
# Train Logistic Regression Model
model = LogisticRegression(max_iter=1000)

model.fit(X_train, y_train)

# Predictions
y_pred = model.predict(X_test)

# Accuracy
accuracy = accuracy_score(y_test, y_pred)

print("\nAccuracy:")
print(accuracy)
print("\nConfusion Matrix:")
print(confusion_matrix(y_test, y_pred))

print("\nClassification Report:")
print(classification_report(y_test, y_pred))