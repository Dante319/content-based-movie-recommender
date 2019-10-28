import pandas as pd
import random
from ast import literal_eval
import csv
import os.path


metadata = pd.read_csv('../MR/Dataset/movies_metadata.csv', low_memory=False)
watch_list_name = '../MR/Dataset/watch_list.csv'
favorites_name = '../MR/Dataset/favorites.csv'

if not os.path.isfile(watch_list_name):
    with open(watch_list_name, 'w') as csvfile:
        filewriter = csv.writer(csvfile, delimiter=',',
                                quotechar='|', quoting=csv.QUOTE_MINIMAL)
        filewriter.writerow(['title', 'tagline', 'production_companies', 'overview', 'genres', 'runtime', 'vote_average'])

if not os.path.isfile(favorites_name):
    with open(favorites_name, 'w') as csvfile:
        filewriter = csv.writer(csvfile, delimiter=',',
                                quotechar='|', quoting=csv.QUOTE_MINIMAL)
        filewriter.writerow(['title', 'tagline', 'production_companies', 'overview', 'genres', 'runtime', 'vote_average'])

metadata.head(3)

C = metadata['vote_average'].mean()

m = metadata['vote_count'].quantile(0.90)

metadata = metadata.copy().loc[metadata['vote_count'] >= m]

# Function that computes the weighted rating of each movie
def weighted_rating(x, m=m, C=C):
    v = x['vote_count']
    R = x['vote_average']
    # Calculation based on the IMDB formula
    return (v/(v+m) * R) + (m/(m+v) * C)

# Define a new feature 'score' and calculate its value with `weighted_rating()`
metadata['score'] = metadata.apply(weighted_rating, axis=1)

#Sort movies based on score calculated above
metadata = metadata.sort_values('score', ascending=False)
metadata = metadata.reset_index()

#Content Based Recommender

from sklearn.feature_extraction.text import TfidfVectorizer

tfidf = TfidfVectorizer(stop_words='english')

metadata['overview'] = metadata['overview'].fillna('')

tfidf_matrix = tfidf.fit_transform(metadata['overview'])

from sklearn.metrics.pairwise import linear_kernel

cosine_sim = linear_kernel(tfidf_matrix, tfidf_matrix)

indices = pd.Series(metadata.index, index=metadata['title']).drop_duplicates()


# Function that takes in movie title as input and outputs most similar movies
def get_recommendations(title, cosine_sim=cosine_sim):
    idx = indices[title]
    sim_scores = list(enumerate(cosine_sim[idx]))
    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)
    sim_scores = sim_scores[1:11]
    movie_indices = [i[0] for i in sim_scores]

    # Return the top 10 most similar movies
    return metadata['title'].iloc[movie_indices]

# Load keywords and credits
credits = pd.read_csv('../MR/Dataset/credits.csv')
keywords = pd.read_csv('../MR/Dataset/keywords.csv')

# Convert IDs to int. Required for merging
keywords['id'] = keywords['id'].astype('int')
credits['id'] = credits['id'].astype('int')
metadata['id'] = metadata['id'].astype('int')

metadata = metadata.merge(credits, on='id')
metadata = metadata.merge(keywords, on='id')

from ast import literal_eval

features = ['cast', 'crew', 'keywords', 'genres']
for feature in features:
    metadata[feature] = metadata[feature].apply(literal_eval)

# Import Numpy
import numpy as np


def get_director(x):
    for i in x:
        if i['job'] == 'Director':
            return i['name']
    return np.nan


def get_list(x):
    if isinstance(x, list):
        names = [i['name'] for i in x]
        if len(names) > 3:
            names = names[:3]
        return names
    return []


metadata['director'] = metadata['crew'].apply(get_director)

features = ['cast', 'keywords', 'genres']
for feature in features:
    metadata[feature] = metadata[feature].apply(get_list)

def clean_data(x):
    if isinstance(x, list):
        return [str.lower(i.replace(" ", "")) for i in x]
    else:
        if isinstance(x, str):
            return str.lower(x.replace(" ", ""))
        else:
            return ''

features = ['cast', 'keywords', 'director', 'genres']

for feature in features:
    metadata[feature] = metadata[feature].apply(clean_data)


def create_soup(x):
    return ' '.join(x['keywords']) + ' ' + ' '.join(x['cast']) + ' ' + x['director'] + ' ' + ' '.join(x['genres'])


metadata['soup'] = metadata.apply(create_soup, axis=1)


from sklearn.feature_extraction.text import CountVectorizer

count = CountVectorizer(stop_words='english')
count_matrix = count.fit_transform(metadata['soup'])

from sklearn.metrics.pairwise import cosine_similarity

cosine_sim2 = cosine_similarity(count_matrix, count_matrix)

metadata = metadata.reset_index()
indices = pd.Series(metadata.index, index=metadata['title'])


def print_details(idx):
    production_companies = {}
    genres = {}
    print('Title: ' + metadata['title'].iloc[idx])
    print('Tagline: ' + str(metadata['tagline'].iloc[idx]))
    production_companies = literal_eval(metadata['production_companies'].iloc[idx])
    print('Production Companies: ')
    for item in production_companies:
        print(item['name'])
    print('Overview: ' + metadata['overview'].iloc[idx])
    #genres = literal_eval(metadata['genres'].iloc[idx])
    print('Genres: ')
    for item in metadata['genres'].iloc[idx]:
        print(item)
    print('Runtime: ' + str(metadata['runtime'].iloc[idx]))
    print('Average vote: ' + str(metadata['vote_average'].iloc[idx]))
    return


def add_movie_to_watch_list(name):
    idx = indices[name]
    title = metadata['title'].iloc[idx]
    tagline = metadata['title'].iloc[idx]
    production_companies = metadata['production_companies'].iloc[idx]
    overview = metadata['overview'].iloc[idx]
    genres = metadata['genres'].iloc[idx]
    runtime = metadata['runtime'].iloc[idx]
    vote_average = metadata['vote_average'].iloc[idx]

    row = [title,tagline,production_companies,overview,genres,runtime,vote_average]
    with open('../MR/Dataset/watch_list.csv', 'a') as csvFile:
        writer = csv.writer(csvFile)
        writer.writerow(row)
    return


def add_movie_to_favorites(name):
    idx = indices[name]
    title = metadata['title'].iloc[idx]
    tagline = metadata['title'].iloc[idx]
    production_companies = metadata['production_companies'].iloc[idx]
    overview = metadata['overview'].iloc[idx]
    genres = metadata['genres'].iloc[idx]
    runtime = metadata['runtime'].iloc[idx]
    vote_average = metadata['vote_average'].iloc[idx]

    row = [title, tagline, production_companies, overview, genres, runtime, vote_average]
    with open(favorites_name, 'a') as csvFile:
        writer = csv.writer(csvFile)
        writer.writerow(row)
    return


def roulette():
    rand = random.randint(1,4554)
    title = metadata['title'].iloc[rand]
    print_details(rand)
    choice = int(input("Do you want to add this movie to the watch list (1) or the favorites (2) or neither(0)?"))
    if choice == 1:
        add_movie_to_watch_list(title)
    elif choice == 2:
        add_movie_to_favorites(title)
    elif choice == 0:
        return
    return


def search():
    title = str(input("Enter movie: "))
    idx = indices[title]
    print_details(idx)
    choice = int(input("Do you want to add this movie to the watch list (1) or the favorites (2) or neither(0)?"))
    if choice == 1:
        add_movie_to_watch_list(title)
    elif choice == 2:
        add_movie_to_favorites(title)
    elif choice == 0:
        return
    return


def request_recommendations():
    while True:
        name = str(input("Enter movie title (or End): "))
        if name == "End":
            break
        print(get_recommendations(name, cosine_sim2))
    return


def watch_list():
    with open('../MR/Dataset/watch_list.csv', 'r') as csvFile:
        reader = csv.reader(csvFile)
        for row in reader:
            print(row)
    confirm = str(input("Do you want to add a movie to the watch list?"))
    if confirm == "yes":
        search()
    return


def favorites():
    with open('../MR/Dataset/watch_list.csv', 'r') as csvFile:
        reader = csv.reader(csvFile)
        for row in reader:
            print(row)
    confirm = str(input("Do you want to add a movie to the watch list?"))
    if confirm == "yes":
        search()
    return


while True:
    print("1.Roulette\n2.Search\n3.Request Recommendations\n4.Watch list\n5.Favorites\n0.Exit")
    choice = int(input("Enter your choice: "))
    if choice == 0:
        break
    elif choice == 1:
        roulette()
    elif choice == 2:
        search()
    elif choice == 3:
        request_recommendations()
    elif choice == 4:
        watch_list()
    elif choice == 5:
        favorites()

