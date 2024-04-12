import pandas as pd
import os
import gensim.downloader as api
import argparse

# Load the song lyric data
df = pd.read_csv("in/Spotify_Million_Song_Dataset_exported.csv")

# Load the word embedding model via gensim
model = api.load("glove-wiki-gigaword-50")


def expand_query(model, target_word, topn = 10):

    """
    First, lets define the expand_query function. The function will take a given string input (taget_word), for 
    which it will find  closely related words using a pre-trained word embedding model. 
    
    For example, if the target word were to be "love", the script will retrive 10 similar words (topn) such as
    "affection", "romance", etc. These words will expand the original query and allow to include a wider range of
    related words in the song lyric dataset.
    """

    similar_words = [word for word, _ in model.most_similar(target_word, topn=topn)]
    similar_words = similar_words.append(target_word) # append the target word as well
    return similar_words # returns a list of 10 similar words to the target word and the target word itself


def calculate_percentage(artist_name, similar_words):

    """
    Now, we can define a function, that calculates the percentage of songs by the given artist that features the
    words from the expanded query. The function takes two inputs: the artist name (artist_name) and the list of words
    in the expanded query (similar_words).

    The function filters the df to only include the rows where the artist column matches the input artist_name
    and finds the total number of songs by the artist. Afterwards, it will find how many songs by the artist features
    words from the expanded query. Finally, it will caculate the percentage of songs with words from expanded query.
    """

    artist_songs = df[df['artist'] == artist_name]['text'].str.lower()
    total_songs = len(artist_songs)
    songs_with_words = artist_songs.str.contains('|'.join(similar_words)).sum()
    percentage = (songs_with_words / total_songs) * 100 if total_songs > 0 else 0
    percentage = round(percentage, 2)
    return percentage


def main():

    # Enter input word for query expansion
    target_word = input("Enter a target word: ")

    # Expand the query with 10 similar words
    similar_words = expand_query(model, target_word)

    # Enter input artist name
    artist_name = input("Enter a artist's name: ")

    # Calculate the percentage of an artist's songs featuring words from the expanded query
    percentage = calculate_percentage(artist_name, similar_words)

    # Print the results
    print(f"{percentage}% of {artist_name}'s songs contain words related to {target_word}")

    # Save the results as .txt
    filepath = f"out/{artist_name}_{target_word}_results.txt"
    with open(filepath, 'w') as file:
        file.write(f"{percentage}% of {artist_name}'s songs contain words related to {target_word}")
    print("Result saved as .txt")

if __name__ == "__main__":
    main()
