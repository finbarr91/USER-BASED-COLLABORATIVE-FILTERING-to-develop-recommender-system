# PROJECT OVERVIEW
# Recommender systems are algorithms designed to help users discover movies, products, and song
# by predicting the user's rating of each item and displaying similar items that they might
# rate high as well.
# The objective is to show customers content that they would like best based on their historical activity.

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

movie_title_df = pd.read_csv('Movie_Id_Titles')
print('the head of the dataset:\n', movie_title_df.head())
print('\n The tail of the dataset:\n', movie_title_df.tail())

movies_rating_df = pd.read_csv('u.data', sep='\t', names=['user_id', 'item_id', 'rating', 'timestamp'])
print('\n Movie rating: \n', movies_rating_df)
movies_rating_df.drop(['timestamp'], axis= 1,inplace=True)
print(movies_rating_df)

print('\nThe description:\n',movies_rating_df.describe())
print('\n The info of the dataframe:\n', movies_rating_df.info)
movies_rating_df = pd.merge(movies_rating_df, movie_title_df, on= 'item_id')
print('\n The merged dataframe:\n', movies_rating_df)
print('\nThe shape of the dataframe:', movies_rating_df.shape)

# VISUALIZING THE DATASET
print('\nGrouping the elements based on their titles:\n', movies_rating_df.groupby('title').describe())
print('\nGrouping the rating elements based on their titles:\n', movies_rating_df.groupby('title')['rating'].describe())

ratings_df_mean = movies_rating_df.groupby('title')['rating'].describe()['mean']
print(ratings_df_mean)
ratings_df_count = movies_rating_df.groupby('title')['rating'].describe()['count']
print(ratings_df_count)

# Concatenating dataframes
ratings_mean_count_df = pd.concat([ratings_df_count,ratings_df_mean], axis =1)
print(ratings_mean_count_df)

print(ratings_mean_count_df.reset_index()) # reset the index

ratings_mean_count_df['mean'].plot(bins =100, kind = 'hist', color = 'r')
plt.show()

ratings_mean_count_df['count'].plot(bins=100, kind ='hist', color = 'b')
plt.show()

print(ratings_mean_count_df[ratings_mean_count_df['mean']==5])

# To show the movies that are most rated
print(ratings_mean_count_df.sort_values('count', ascending = False).head(100))
print(ratings_mean_count_df.sort_values('count', ascending = True).head(100))

# PERFORMING THE ITEM BASED COLLABORATIVE FILTERING ON ONE MOVIE SAMPLE
userid_movietitle_matrix = movies_rating_df.pivot_table(index = 'user_id', columns ='title', values = 'rating')
print(userid_movietitle_matrix)
titanic = userid_movietitle_matrix['Titanic (1997)'] # For the movie Titanic
print(titanic)

starwars = userid_movietitle_matrix['Star Wars (1977)']
print(starwars)

# To create a correlation between a movie Titanic with the whole dataframe
titanic_correlations =pd.DataFrame(userid_movietitle_matrix.corrwith(titanic), columns =['Correlation'])
titanic_correlations = titanic_correlations.join(ratings_mean_count_df['count'])
print(titanic_correlations)
print(titanic_correlations.dropna(inplace=True))
titanic_correlations.sort_values('Correlation', ascending = False)
# To find movies that have been reviewed several times in other to find a better correlations eith Titanic

print(titanic_correlations[titanic_correlations['count']>80].sort_values('Correlation', ascending = False ).head())

# To create a correlation between a movie Star wars with the whole movie dataframe
starwars_correlations =pd.DataFrame(userid_movietitle_matrix.corrwith(starwars), columns =['Correlation'])
starwars_correlations = starwars_correlations.join(ratings_mean_count_df['count'])
print(starwars_correlations)
print(starwars_correlations.dropna(inplace=True))
starwars_correlations.sort_values('Correlation', ascending = False)
# To find movies that have been reviewed several times in other to find a better correlations with Starwars

print(starwars_correlations[starwars_correlations['count']>80].sort_values('Correlation', ascending = False ).head())

# CREATE AN ITEM BASED COLLABORATIVE FILTER ON THE ENTIRE DATASET
movie_correlations = userid_movietitle_matrix.corr(method = 'pearson', min_periods=80)
print(movie_correlations)

myRatings = pd.read_csv('My_Ratings.csv')
print(myRatings)

print(myRatings['Movie Name'][0])

similar_movies_list= pd.Series()
for i in range(0,2):
    similar_movie = movie_correlations[myRatings['Movie Name'][i]].dropna()
    similar_movie = similar_movie.map(lambda x: x* myRatings['Ratings'][i])
    similar_movies_list = similar_movies_list.append(similar_movie)

similar_movies_list.sort_values(inplace= True, ascending=False)
print(similar_movies_list.head(10))









