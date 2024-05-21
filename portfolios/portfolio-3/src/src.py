import pandas as pd
import os
import gensim.downloader as api
import argparse
import string
from codecarbon import EmissionsTracker

def emissions_tracker(tracker_outpath):
    """
    """
    tracker = EmissionsTracker(project_name = "assignment 3",
                                experiment_id = "assignment_3",
                                output_dir = tracker_outpath,
                                output_file = "emissions_assignment3.csv")
    return tracker


def load_data(filepath):
    """ Load data from given filepath """
    df = pd.read_csv(filepath)
    return df


def remove_punctuation(df):
    """ Remove punctuation from the 'text' column """
    df['text'] = df['text'].apply(lambda x: x.translate(str.maketrans('', '', string.punctuation)))
    return df

def load_model():
    """ Load the word embedding model via gensim """
    model = api.load("glove-wiki-gigaword-50")
    return model


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


def calculate_percentage(df, artist_name, similar_words):

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

    tracker_outpath = "../portfolio-5/out"
    tracker = emissions_tracker(tracker_outpath)

    tracker.start_task("load and clean data")
    filepath = "in/Spotify_Million_Song_Dataset_exported.csv"
    df = load_data(filepath)
    df = remove_punctuation(df)
    emissions_a3_load_df = tracker.stop_task()

    tracker.start_task("load model")
    model = load_model()
    emissions_a3_load_model = tracker.stop_task()

    # Obtain inputs using parser
    tracker.start_task("parser input")
    args = parser()
    target_word = args.word
    artist_name = args.artist
    emissions_a3_arg_input = tracker.stop_task()

    # Expand the query with 10 similar words
    tracker.start_task("expand query")
    similar_words = expand_query(model, target_word)
    emissions_a3_expand_query = tracker.stop_task()
    
    # Calculate the percentage of an artist's songs featuring words from the expanded query
    tracker.start_task("calculate percentage")
    percentage = calculate_percentage(df, artist_name, similar_words)
    emissions_a3_calculate_percentage= tracker.stop_task()

    tracker.start_task("print and save results")
    filepath = f"out/{artist_name}_{target_word}_results.txt"
    with open(filepath, 'w') as file:
        file.write(f"{percentage}% of {artist_name}'s songs contain words related to {target_word}")
    print("The result has been saved to the out folder")

    emissions_a3_print_save = tracker.stop_task()

    tracker.stop()
    
if __name__ == "__main__":
    main()