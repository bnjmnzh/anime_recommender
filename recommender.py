import os
import numpy as np
import pandas as pd

from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity

def load_synop_data():
    anime_synop_df = pd.read_csv('../input/anime-recommendation-database-2020/anime_with_synopsis.csv')
    return anime_synop_df

def load_complete_data():
    anime_complete_df = pd.read_csv('../input/anime-recommendation-database-2020/anime.csv')
    return anime_complete_df

def preprocessing(anime_synop_df, anime_complete_df):
    anime_synop_df = anime_synop_df[['MAL_ID', 'Name', 'Genres', 'sypnopsis']]

    anime_df = anime_synop_df.join(anime_complete_df, on='MAL_ID', rsuffix='r')
    anime_df = anime_df[['Name', 'Genres', 'sypnopsis', 'Type']]
    anime_df.columns = ['Name', 'Genres', 'Synopsis', 'Type']

    anime_df['Synopsis'] = anime_df['Synopsis'].fillna(
        anime_df['Synopsis'].dropna().mode().values[0]
    )
    anime_df['Type'] = anime_df['Type'].fillna(
        anime_df['Type'].dropna().mode().values[0]
    )

    anime_df = anime_df[anime_df['Type']=='TV']
    anime_df.drop('Type', axis=1, inplace=True)

    return anime_df

anime_synop_df = load_synop_data()
anime_complete_df = load_complete_data()

anime_df = preprocessing(anime_synop_df, anime_complete_df)

indices = pd.Series(anime_df.index, index = anime_df['Name'])

tfidf = TfidfVectorizer(stop_words='english')
tfidf2 = TfidfVectorizer(stop_words='english')
count = CountVectorizer(stop_words='english')

tfidf_matrix = tfidf.fit_transform(anime_df['Name'])
count_matrix = count.fit_transform(anime_df['Genres'])
tfidf2_matrix = tfidf2.fit_transform(anime_df['Synopsis'])

name_similarity = cosine_similarity(tfidf_matrix)
genre_similarity = cosine_similarity(count_matrix)
synopsis_similarity = cosine_similarity(tfidf2_matrix)

def get_recommendations(anime):
    i = indices[anime]
    
    name_score = list(enumerate(name_similarity[i]))
    genre_score = list(enumerate(genre_similarity[i]))
    synopsis_score = list(enumerate(synopsis_similarity[i]))
    
    name_score = sorted(name_score, key = lambda x: x[0])
    genre_score = sorted(genre_score, key = lambda x: x[0])
    synopsis_score = sorted(synopsis_score, key = lambda x: x[0])
    
    combined_score = [(i, (sc_1 + sc_2 + sc_3) / 3) for (i, sc_1), (_, sc_2), (_, sc_3) in zip(name_score, genre_score, synopsis_score)]
    
    combined_score = sorted(combined_score, key = lambda x: x[1], reverse = True)
    
    anime_ids = [i[0] for i in combined_score[1:11]]
    
    anime_recs = []
    
    index = 0
    while len(anime_recs) != 10:
        anime_id = combined_score[1:][index][0]
        index += 1
        if anime in indices.iloc[[anime_id]].index[0]:
            continue
        else:
            anime_recs.append(indices.iloc[[anime_id]].index[0])
    
    
    print(f'If you liked {anime}, you should try:')
    for i, v in list(enumerate(anime_recs)):
        print(f'{i + 1}. {v}')
