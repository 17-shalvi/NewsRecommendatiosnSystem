import numpy as np
import pandas as pd
from sklearn.feature_extraction import text
from sklearn.metrics.pairwise import cosine_similarity
import plotly.express as px
import plotly.graph_objects as go

#use 'r' to produce a raw string
data = pd.read_csv(r"C:\Users\17sha\Downloads\News1.csv")
print(data.head())

# Types of News Categories

#calculates the count of each unique value in the "News Category"
# column of dataset (data) and stores it in the categories.
categories = data["News Category"].value_counts()
label = categories.index
counts = categories.values
figure = px.bar(data, x=label, y = counts, title="Types of News Categories")
figure.show()

#Recommendation systme based on the Title of the News Article
#similarity analysis using TF-IDF (Term Frequency-Inverse Document Frequency) and cosine similarity.
feature = data["Title"].tolist()

tfidf = text.TfidfVectorizer(input=feature, stop_words="english")
tfidf_matrix = tfidf.fit_transform(feature)
similarity = cosine_similarity(tfidf_matrix)

indices = pd.Series(data.index, index=data['Title']).drop_duplicates()

def news_recommendation(Title, similarity = similarity):
    index = indices[Title]
    similarity_scores = list(enumerate(similarity[index]))
    similarity_scores = sorted(similarity_scores,
    key=lambda x: x[1], reverse=True)
    similarity_scores = similarity_scores[0:10]
    newsindices = [i[0] for i in similarity_scores]
    return data['Title'].iloc[newsindices]

print(news_recommendation("Walmart Slashes Prices on Last-Generation iPads"))
