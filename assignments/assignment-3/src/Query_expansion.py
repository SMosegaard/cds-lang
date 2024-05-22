import pandas as pd
import os
import gensim.downloader as api
import argparse
import string
from codecarbon import EmissionsTracker


def emissions_tracker(outpath):
    """
    The function initializes an EmissionsTracker object to track carbon emissions associated
    with code execution. The results of this can be found in assignment 5.
    """
    tracker = EmissionsTracker(project_name = "assignment 3",
                                output_dir = outpath)
    return tracker


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



def remove_punctuation(df):
    """ Remove punctuation from the 'text' column """
    df['text'] = df['text'].apply(lambda x: x.translate(str.maketrans('', '', string.punctuation)))
    return df


def load_clean_data(filepath, tracker):
    tracker.start_task("load and clean data")
    df = pd.read_csv(filepath)
    df = remove_punctuation(df)
    emissions_3_load_clean_data = tracker.stop_task()
    return df


def load_model(tracker):
    """ Load the word embedding model via gensim """
    tracker.start_task("load model")
    model = api.load("glove-wiki-gigaword-50")
    emissions_3_load_model = tracker.stop_task()
    return model


def expand_query(model, target_word, tracker, topn = 10):

    """
    First, lets define the expand_query function. The function will take a given string input (taget_word), for 
    which it will find  closely related words using a pre-trained word embedding model. 
    
    For example, if the target word were to be "love", the script will retrive 10 similar words (topn) such as
    "affection", "romance", etc. These words will expand the original query and allow to include a wider range of
    related words in the song lyric dataset.
    Return list of 10 similar words to the target word and the target word itself
    """
    tracker.start_task("expand query")
    similar_words = [word for word, _ in model.most_similar(target_word, topn = topn)]
    emissions_3_expand_query = tracker.stop_task()
    return similar_words


def calculate_percentage(df, artist_name, similar_words, tracker):

    """
    Now, we can define a function, that calculates the percentage of songs by the given artist that features the
    words from the expanded query. The function takes two inputs: the artist name (artist_name) and the list of 
    words in the expanded query (similar_words).

    The function filters the df to only include the rows where the artist column matches the input artist_name
    and finds the total number of songs by the artist. Afterwards, it will find how many songs by the artist
    features words from the expanded query, by iterating over each song's text in a loop. Finally, it will
    caculate the percentage of songs with words from expanded query.
    """
    tracker.start_task("calculate percentage")
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
    emissions_3_calculate_percentage= tracker.stop_task()
    return total_songs, songs_with_words, percentage


def save_results_to_df(df, target_word, artist_name, similar_words, total_songs, songs_with_words, percentage):
    """
    Converts the dictionary to a dataframe.
    Check if the .csv file exists --> if not, write a new table with headers
    If it does excist --> Append the new row to the existing CSV file without the header
    """
    new_row = {
        'Target word': target_word,
        'Artist': artist_name,
        'Query': ', '.join(similar_words),
        'Total songs by artist': total_songs,
        'Songs containing query words': songs_with_words,
        'Percentage': percentage
    }
    new_row_df = pd.DataFrame(new_row, index = [0])
    
    if not os.path.exists(f"out/results.csv"):
        new_row_df.to_csv(f"out/results.csv", index = False, mode = 'w')
    else:
        new_row_df.to_csv(f"out/results.csv", index = False, mode = 'a', header = False)
    print(f"{percentage}% of {artist_name}'s songs contain words related to {target_word}")
    return print("The result has been saved to the out folder")


def main():

    outpath = "../assignment-5/out"
    tracker = emissions_tracker(outpath)
    tracker.start()

    args = parser()

    filepath = "in/Spotify_Million_Song_Dataset_exported.csv"
    df = load_clean_data(filepath, tracker)

    model = load_model(tracker)

    similar_words = expand_query(model, args.word, tracker, topn = 10)
    
    total_songs, songs_with_words, percentage = calculate_percentage(df, args.artist, similar_words, tracker)

    save_results_to_df(df, args.word, args.artist, similar_words, total_songs, songs_with_words, percentage)

    tracker.stop()
    
if __name__ == "__main__":
    main()