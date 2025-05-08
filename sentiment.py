import pandas as pd
from transformers import pipeline
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

#model
sentiment_pipeline = pipeline("sentiment-analysis", model="textattack/bert-base-uncased-imdb", truncation=True)
#import data
df = pd.read_csv('data/IMDB Dataset.csv')

# Split the dataset into features and labels
X = df['review']
y = df['sentiment']

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


x = df['review']
x = x[:100]
# Prepare data for plotting
scores = []
lengths = []


dummy_reviews = [
    "This movie was absolutely fantastic! The acting was superb and the story was captivating.",
    "I didn't enjoy this film at all. The plot was predictable and the characters were dull.",
    "An average movie with some good moments, but overall it lacked depth.",
    "The visuals were stunning, but the storyline was confusing and hard to follow.",
    "One of the best movies I've seen this year! Highly recommend it to everyone.",
    "Terrible movie. I couldn't even finish watching it. Complete waste of time.",
    "A decent film with a few standout performances, but nothing extraordinary.",
    "The humor in this movie was spot on! I laughed from start to finish.",
    "The pacing was too slow, and the ending was disappointing. Not worth watching.",
    "An emotional rollercoaster with brilliant acting and a powerful message."
]

for review in X_test:

    result = sentiment_pipeline(review)
    #scores.append(result[0]['score'])
    #lengths.append(len(review))
    print(f"Review: {review}\n")
    print(f"Label: {result[0]['label']}, Score: {result[0]['score']}")
    print('----------------\n')
