import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, classification_report

# Load the IMDb dataset
# Assuming the dataset is a CSV file with 'review' and 'sentiment' columns
# Replace 'imdb_dataset.csv' with the actual path to your dataset
data = pd.read_csv('/Users/jackmorin/Desktop/ML_Final/data/IMDB Dataset.csv')

# Preprocess the data
X = data['review']
y = data['sentiment']

# Convert text data to numerical data using CountVectorizer
vectorizer = CountVectorizer(stop_words='english', max_features=5000)
X_vectorized = vectorizer.fit_transform(X)

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X_vectorized, y, test_size=0.2, random_state=42)

# Create and train the decision tree classifier
clf = DecisionTreeClassifier(random_state=42)
clf.fit(X_train, y_train)

# Make predictions
y_pred = clf.predict(X_test)

# Evaluate the model
accuracy = accuracy_score(y_test, y_pred)
print(f"predicted: {y_pred}")
print(f"Accuracy: {accuracy:.2f}")
print("\nClassification Report:")
print(classification_report(y_test, y_pred))


# Create sarcastic movie reviews to test the model
sarcastic_reviews = [
    "Oh, what a masterpiece! I totally didn't fall asleep halfway through.",
    "The acting was so good, I almost believed they were real people. Almost.",
    "This movie changed my life. Now I know what not to watch.",
    "The plot twists were so predictable, I guessed them all in the first five minutes.",
    "Wow, the special effects were so realistic, I thought I was watching a cartoon.",
    "this was a movie",
    "that sure was an hour and a half long form of video content"
]

# Transform the sarcastic reviews using the same vectorizer
sarcastic_reviews_vectorized = vectorizer.transform(sarcastic_reviews)

# Predict sentiments for the sarcastic reviews
sarcastic_predictions = clf.predict(sarcastic_reviews_vectorized)

# Print the sarcastic reviews with their predicted sentiments
for review, sentiment in zip(sarcastic_reviews, sarcastic_predictions):
    print(f"Review: {review}\nPredicted Sentiment: {sentiment}\n")