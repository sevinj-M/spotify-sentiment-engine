import pandas as pd
import numpy as np
from utils_nb import process_tweet
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report

'''
1. load data: take artist(s), text, positivenes
2. balanced labeling: create column 'label' -> 0/1
3. preprocessing: tokenize lyrics
4. split Dataset and train the model
5. lyrics -> vectors
6. predict function
7. train + test datasets
8. example usage
'''

#1 load data
all_data = pd.read_csv("/Users/sevinjmahmudova/Desktop/archive/spotify_dataset.csv")
data = all_data[['Artist(s)', 'song', 'text', 'Positiveness']].copy()

#2 balanced labeling
data['Labels'] = (data['Positiveness'] >= 50).astype(int)

positives = data[data['Labels'] == 1].head(225000)
negatives = data[data['Labels'] == 0].head(225000)

data = pd.concat([positives, negatives], ignore_index=True)

#3 preprocessing
data['processed_text'] = data['text'].apply(lambda x: " ".join(process_tweet(x)))

#4 split dataset
X_train, X_test, y_train, y_test = train_test_split(
    data['processed_text'], data['Labels'], test_size=0.2, random_state=42
)

#5 vectorize
tfidf_vectorizer = TfidfVectorizer(max_features=1500)
X_train_tfidf = tfidf_vectorizer.fit_transform(X_train)
X_test_tfidf = tfidf_vectorizer.transform(X_test)

#6 train & evaluate
model = RandomForestClassifier(random_state=42, n_jobs=-1) # n_jobs=-1 speeds up RF
model.fit(X_train_tfidf, y_train)
y_pred = model.predict(X_test_tfidf)
print(classification_report(y_test, y_pred))

#7 predict function
def predict_lyrics(lyrics_list):
    processed = [" ".join(process_tweet(text)) for text in lyrics_list]
    vec = tfidf_vectorizer.transform(processed)
    return model.predict(vec)

#8 Ending: To predict whether your lyrics are positive or negative, use this function:
test_lyrics = ["I'll swim even when the water's cold That's the one thing that I know"]
print(f"Prediction: {predict_lyrics(test_lyrics)}")


