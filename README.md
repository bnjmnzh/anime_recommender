
# Anime Recommender

  

Uses Tf-idf on anime title, synopsis and count frequency on anime genres then averaging cosine similarity scores to recommend animes from titles. 

## Data
The data on anime is taken from https://www.kaggle.com/hernan4444/anime-recommendation-database-2020.

## Usage
You can invoke the recommendation function using `get_recommendations()` with the name of an anime. One caveat is that you must enter the name exactly like it is as seen on [MyAnimeList](https://myanimelist.net/).  

## Output
The recommendation function will output 10 animes 'similar' to what you enter. Note that it removes any animes from the output that contain the name of the anime input. This is to avoid suggestions like Pokemon recommending Pokemon Diamond and Pearl, Pokemon Battle Frontier, etc. 

Sample output:
```
get_recommendations('Dragon Ball')

If you liked Dragon Ball, you should try:
1. Tetsujin 28-gou: Tanjou-hen
2. Bubblegum Crisis
3. Get Ride! AMDriver
4. Bubblegum Crisis Tokyo 2040
5. Gad Guard
6. Dennou Boukenki Webdiver
7. Chou Denji Robo Combattler V
8. Transformers: Choujin Master Force
9. Zoids Fuzors
10. Chogattai Majutsu Robot Ginguiser
```

## Results
This model is not very good.