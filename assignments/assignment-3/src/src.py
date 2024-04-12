# Import packages
import pandas as pd
import os
import gensim.downloader as api
import argparse
import string

# Load the song lyric data
df = pd.read_csv("in/Spotify_Million_Song_Dataset_exported.csv")

# Remove punctuation from the 'text' column
df['text'] = df['text'].apply(lambda x: x.translate(str.maketrans('', '', string.punctuation)))

# Load the word embedding model via gensim
model = api.load("glove-wiki-gigaword-50")


def parser():

    """
    Obtain the input target word and artist name using argparse.
    """

    parser = argparse.ArgumentParser()
    parser.add_argument("--word",
                        "-w",
                        required = True,
                        help = "Enter a taget word by writing --word {target_word}")
    parser.add_argument("--artist",
                        "-a",
                        required = True,
                        help = "Enter an artist name by writing --artist {artist_name}")
    args = parser.parse_args()

    args.word = args.word.lower()
    args.artist = args.artist.lower()

    return args


def expand_query(model, target_word, topn = 10):

    """
    First, lets define the expand_query function. The function will take a given string input (taget_word), for 
    which it will find  closely related words using a pre-trained word embedding model. 
    
    For example, if the target word were to be "love", the script will retrive 10 similar words (topn) such as
    "affection", "romance", etc. These words will expand the original query and allow to include a wider range of
    related words in the song lyric dataset.
    """

    similar_words = [word for word, _ in model.most_similar(target_word, topn=topn)]
    return similar_words # returns a list of 10 similar words to the target word and the target word itself


def calculate_percentage(artist_name, similar_words):

    """
    Now, we can define a function, that calculates the percentage of songs by the given artist that features the
    words from the expanded query. The function takes two inputs: the artist name (artist_name) and the list of 
    words in the expanded query (similar_words).

    The function filters the df to only include the rows where the artist column matches the input artist_name
    and finds the total number of songs by the artist. Afterwards, it will find how many songs by the artist
    features words from the expanded query, by iterating over each song's text in a loop. Finally, it will
    caculate the percentage of songs with words from expanded query.
    """

    artist_songs = df[df['artist'].str.lower() == artist_name]
    total_songs = len(artist_songs)

    songs_counted = set()
    songs_with_words = 0
    for word in similar_words:
        for song in artist_songs['text'].str.lower():
            # Check if word appears in the song and if the song hasn't been counted yet
            if word in song and song not in songs_counted:
                songs_counted.add(song)
                songs_with_words += 1
    
    percentage = (songs_with_words / total_songs) * 100 if total_songs > 0 else 0
    percentage = round(percentage, 2)
    return percentage


def main():

    # Obtain inputs using parser
    args = parser()
    target_word = args.word
    artist_name = args.artist

    # Expand the query with 10 similar words
    similar_words = expand_query(model, target_word)

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

