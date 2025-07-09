from transformers import pipeline
import pandas as pd
import numpy as np
from tqdm import tqdm

books = pd.read_csv('books_with_categories.csv')

classifier = pipeline("text-classification",
                      model="j-hartmann/emotion-english-distilroberta-base", return_all_scores=True)
classifier("I love this!")


print(classifier(books["description"][0]))

sentence = books['description'][0].split('.')
predictions = classifier(sentence)

print(predictions[0])

sorted(predictions[0], key=lambda x: x["label"])

emotion_labels = ['anger', 'disgust', 'fear',
                  'joy', 'sadness', 'surprise', 'neutral']

isbn = []

emotion_scores = {label: [] for label in emotion_labels}


def calculate_max_emotion_scores(predictions):
    per_emotion_scores = {label: [] for label in emotion_labels}
    for predictions in predictions:
        sorted_predictions = sorted(predictions, key=lambda x: x['label'])
        for index, label in enumerate(emotion_labels):
            per_emotion_scores[label].append(
                sorted_predictions[index]["score"])
        return {label: np.max(scores) for label, scores in per_emotion_scores.items()}


emotion_labels = ["anger", "disgust", "fear",
                  "joy", "sadness", "surprise", "neutral"]
isbn = []
emotion_scores = {label: [] for label in emotion_labels}

for i in tqdm(range(len(books))):
    isbn.append(books["isbn13"][i])
    sentences = books["description"][i].split(".")
    predictions = classifier(sentences)
    max_scores = calculate_max_emotion_scores(predictions)
    for label in emotion_labels:
        emotion_scores[label].append(max_scores[label])


emotions_df = pd.DataFrame(emotion_scores)
emotions_df["isbn13"] = isbn


books = pd.merge(books, emotions_df, on="isbn13")
print(books)
books.to_csv("books_with_emotions.csv", index=False)
