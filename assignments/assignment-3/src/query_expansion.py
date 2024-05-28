import pandas as pd
import os
import gensim.downloader as api
import argparse
import string
from codecarbon import EmissionsTracker
import nltk
nltk.download('wordnet')
from nltk.stem import WordNetLemmatizer
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
import numpy as np


def emissions_tracker(outpath):
    """
    The function initializes an EmissionsTracker object to track carbon emissions associated
    with code execution. The results of this can be found in assignment 5.
    """
    tracker = EmissionsTracker(project_name = "assignment 3",
                                output_dir = outpath,
                                output_file = "emissions_assignment_3")
    return tracker


def parser():
    """
    The user can specify a target word and an artist when executing the script.
    The function will then parse command-line arguments and make them lower case.
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
    """
    The function removes punctuation from the 'text' column.
    """
    df['text'] = df['text'].apply(lambda x: x.translate(str.maketrans('', '', string.punctuation)))
    return df


def load_clean_data(filepath, tracker):
    """
    The function loads and cleans the data. The cleaned dataframe will be returned.
    """
    tracker.start_task("load and clean data")
    df = pd.read_csv(filepath)
    df = remove_punctuation(df)
    emissions_3_load_clean_data = tracker.stop_task()
    return df


def load_model(tracker):
    """
    The function loads the 'glove-wiki-gigaword-50' word embedding model via gensim.
    """
    tracker.start_task("load model")
    model = api.load("glove-wiki-gigaword-50")
    emissions_3_load_model = tracker.stop_task()
    return model


def expand_query(model, target_word, tracker, topn = 20):
    """
    The function expands the query by finding 20 similar words to the given target word using
    the word embedding model. Then it filters these words to remove punctuation and any that
    contains the target words. This is done to ensure that each word is unique in its lemmatized
    form. The function will return a list of 10 similar, yet unique words to the target word.
    """
    tracker.start_task("expand query")
    similar_words = [word for word, _ in model.most_similar(target_word, topn = topn)]
    
    lemmatizer = WordNetLemmatizer()
    filtered_words = [word for word in similar_words 
                    if target_word not in word and not any(char in string.punctuation for char in word)]
    
    lemmatized_word_list = set()
    unique_words = []
    for word in filtered_words:
        lemmatized_word = lemmatizer.lemmatize(word)
        if lemmatized_word not in lemmatized_word_list:
            lemmatized_word_list.add(lemmatized_word)
            unique_words.append(word)

    emissions_3_expand_query = tracker.stop_task()
    return  unique_words[:10]


def calculate_percentage(df, artist_name, similar_words, tracker):
    """
    The function calculates the percentage of an artist's songs that features words from the expanded
    query. The function filters the df to only include the rows where the artist column matches the input
    artist_name and finds the total number of songs by the artist. Afterwards, it will find how many songs
    by the artist features words from the expanded query, by iterating over each song's text. Finally, it
    will caculate the percentage of songs with words from the query. The function returns the total number
    of songs by the artist, the number of songs containing the query words, and the percentage of songs
    containing the query words.
    """
    tracker.start_task("calculate percentage")
    artist_songs = df[df['artist'].str.lower() == artist_name]
    total_songs = len(artist_songs)

    songs_counted = set()
    songs_with_words = 0
    for word in similar_words:
        for song in artist_songs['text'].str.lower():
            if word in song and song not in songs_counted:
                songs_counted.add(song)
                songs_with_words += 1

    percentage = (songs_with_words / total_songs) * 100 if total_songs > 0 else 0
    percentage = round(percentage, 2)
    emissions_3_calculate_percentage= tracker.stop_task()
    return total_songs, songs_with_words, percentage



def save_results_to_df(df, target_word, artist_name, similar_words, total_songs, songs_with_words, percentage, tracker):
    """
    The function saves the results as a .csv file. It checks if the .csv file already exists. If not, 
    it writes a new dataframe with headers. If it does exist, it simply append the new row to the existing
    .csv file witout a header. Also, it prints the results in the terminal output.
    """
    tracker.start_task("save results")
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
    emissions_3_save_results = tracker.stop_task()
    return print("The dataframe has been saved to the out folder")


def visualize_multiple_words_from_df(model, df, outpath, tracker, topn = 10):
    """
    The function visualizes all the target words and their query from the saved dataframe
    using t-SNE dimensionality reduction. The plot will show the relations between the words
    in a lower-dimensional space. It will be saved to a specified outpath.
    """
    tracker.start_task("visualise query")
    all_words = []
    word_labels = []

    for index, row in df.iterrows():
        target_word = row['Target word']
        similar_words = row['Query'].split(', ')
        all_words.extend([target_word] + similar_words)
        word_labels.extend([target_word] * (len(similar_words) + 1))

    word_vectors = np.array([model[word] for word in all_words])
    tsne = TSNE(n_components = 2, perplexity = 10, random_state = 123)
    reduced_vectors = tsne.fit_transform(word_vectors)
    
    plt.figure(figsize = (12, 12))
    unique_target_words = list(set(df['Target word']))
    colors = plt.get_cmap('Set2', len(unique_target_words))

    for i, (word, label) in enumerate(zip(all_words, word_labels)):
        if word in df['Target word'].values:
            plt.scatter(reduced_vectors[i, 0], reduced_vectors[i, 1], color='red')
            plt.annotate(word, (reduced_vectors[i, 0], reduced_vectors[i, 1]), color='red')
        else:
            color_idx = unique_target_words.index(label)
            plt.scatter(reduced_vectors[i, 0], reduced_vectors[i, 1], color=colors(color_idx))
            plt.annotate(word, (reduced_vectors[i, 0], reduced_vectors[i, 1]), color=colors(color_idx))

    plt.title("t-SNE visualization of multiple target words and their similar words")
    plt.tight_layout()
    plt.savefig(outpath)
    plt.show()
    emissions_3_plot = tracker.stop_task()
    return print("The visualization has been saved to the out folder")


def main():

    outpath = "../assignment-5/out"
    tracker = emissions_tracker(outpath)
    tracker.start()

    args = parser()

    filepath = "in/Spotify_Million_Song_Dataset_exported.csv"
    df = load_clean_data(filepath, tracker)

    model = load_model(tracker)

    similar_words = expand_query(model, args.word, tracker, topn = 20)
    
    total_songs, songs_with_words, percentage = calculate_percentage(df, args.artist, similar_words, tracker)

    save_results_to_df(df, args.word, args.artist, similar_words, total_songs, songs_with_words, percentage, tracker)

    results_df = pd.read_csv("out/results.csv")
    visualize_multiple_words_from_df(model, results_df, "out/t-SNE_nQueries_from_df.png", tracker, topn = 10)

    tracker.stop()
    
if __name__ == "__main__":
    main()