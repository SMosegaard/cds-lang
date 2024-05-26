import os
import pandas as pd
import glob
import re
import spacy
from codecarbon import EmissionsTracker
import seaborn as sns
import matplotlib.pyplot as plt


def emissions_tracker(tracker_outpath):
    """
    The function initializes an EmissionsTracker object to track carbon emissions associated
    with code execution. The results of this can be found in assignment 5.
    """
    tracker = EmissionsTracker(project_name = "assignment 1",
                                output_dir = tracker_outpath)
    return tracker


def load_spacy():
    """
    The function loads the spaCy 'en_core_web_md' model.
    """
    nlp = spacy.load("en_core_web_md")
    return nlp


def remove_metadata(text):
    """
    The function removes metadata from a text input.
    """
    return re.sub(r"<*?>", "", text)


def count_pos(doc):
    """
    The function counts the occurrences of each part-of-speech (POS) tag in a spaCy document.
    The function takes a spaCy doc object as input and returns a tuple containing counts of
    nouns, verbs, adverbs, and adjectives.
    """
    noun_count, verb_count, adv_count, adj_count = 0, 0, 0, 0

    for token in doc:
        if token.pos_ == "NOUN":
            noun_count += 1
        elif token.pos_ == "VERB":
            verb_count += 1
        elif token.pos_ == "ADV":
            adv_count += 1
        elif token.pos_ == "ADJ":
            adj_count += 1

    return noun_count, verb_count, adv_count, adj_count


def rel_freq(count, len_doc): 
    """
    The function calculates the relative frequency of a count within a document.
    The function takes the count of a POS tag and the total number of tokens in the given text,
    while returns the relative frequency (scaled by 10,000) of the count within the document.
    """
    return round((count/len_doc * 10000), 2)


def no_unique_ents(doc):
    """
    The function counts the total number of unique entities (PERSON, LOC, ORG). The function
    takes a spaCy document as input and returns a list containing counts of unique entities.
    """
    enteties = []

    for ent in doc.ents: 
        enteties.append((ent.text, ent.label_))

    enteties_df = pd.DataFrame(enteties, columns=["enteties", "label"])
    enteties_df = enteties_df.drop_duplicates()
    unique_counts = enteties_df.value_counts(subset = "label")
    
    unique_labels = ['PERSON', 'LOC', 'ORG']
    unique_row = []

    for label in unique_labels:
        if label in (unique_counts.index):
            unique_row.append(unique_counts[label])
        else:
            unique_row.append(0)

    return unique_row


def process_text(filepath, nlp):
    """ 
    The function iterates over the text files from a given filepath and extracts linguistic features.
    The function creasea a Pandas DataFrame to store and append the extracted features for each file.
    Finally, the dataframes with the extracted features are saved as .csv files.
    """
    for subfolder in sorted(os.listdir(filepath)):
        subfolder_path = os.path.join(filepath, subfolder)

        out_df = pd.DataFrame(columns = ("Filename", "RelFreq NOUN", "RelFreq VERB", "RelFreq ADV",
                                        "RelFreq ADJ", "No. Unique PER", "No. Unique LOC", "No. Unique ORG"))

        for file in sorted(glob.glob(os.path.join(subfolder_path, "*.txt"))):
            with open(file, "r", encoding = "latin-1") as f:
                text = f.read()
                doc = nlp(remove_metadata(text))

            noun_count, verb_count, adv_count, adj_count = count_pos(doc)
            len_doc = len(doc)
            noun_rel_freq, verb_rel_freq, adv_rel_freq, adj_rel_freq = rel_freq(noun_count, len_doc), rel_freq(verb_count, len_doc), rel_freq(adv_count, len_doc), rel_freq(adj_count, len_doc)
            No_unique_per, No_unique_loc, No_unique_org = no_unique_ents(doc)
            
            text_name = file.split("/")[-1]

            text_row = [text_name, noun_rel_freq, verb_rel_freq, adv_rel_freq, adj_rel_freq,
                        No_unique_per, No_unique_loc, No_unique_org]
            out_df.loc[len(out_df)] = text_row

        outpath = os.path.join("out", f"{subfolder}_data.csv")
        out_df.to_csv(outpath)

    return print("The dataframe has been saved in the out folder")


def combine_df(dataframes):
    """
    The function combines multiple dataframes into a single dataframe.
    """
    dfs = []
    for dataframe in dataframes:
        df = pd.read_csv(dataframe)
        subfolder = dataframe.split('/')[1].split('_')[0] 
        df['subfolder'] = subfolder
        dfs.append(df)
        
    combined_df = pd.concat(dfs, ignore_index = True)
    return combined_df


def plot_word_class(combined_df, outpath):
    """
    The function plots the relative frequency of word classes (noun, verb, adverb, adjective) across subfolders.
    The plot will be saved to the specified outpath.
    """
    aggregated_df = combined_df.groupby('subfolder').agg({'RelFreq NOUN': 'sum',
                                                        'RelFreq VERB': 'sum',
                                                        'RelFreq ADV': 'sum',
                                                        'RelFreq ADJ': 'sum'
                                                        }).reset_index()

    total_freq = aggregated_df[['RelFreq NOUN', 'RelFreq VERB', 'RelFreq ADV', 'RelFreq ADJ']].sum()
    total_subfolder = aggregated_df[['RelFreq NOUN', 'RelFreq VERB', 'RelFreq ADV', 'RelFreq ADJ']].sum(axis = 1)
    relfreq_class_df = aggregated_df[['RelFreq NOUN', 'RelFreq VERB', 'RelFreq ADV', 'RelFreq ADJ']].div(total_subfolder, axis = 0)
    relfreq_class_df = relfreq_class_df * 100
    relfreq_class_df['subfolder'] = aggregated_df['subfolder']

    word_class_df = pd.melt(relfreq_class_df, id_vars = 'subfolder', var_name = 'Word class',
                            value_name = 'Relative Frequency')

    plt.figure(figsize = (10, 6))
    sns.barplot(data = word_class_df, x = 'subfolder', y = 'Relative Frequency', 
                hue = 'Word class', palette = 'viridis')
    plt.title('Word classes across subfolders')
    plt.xlabel('Subfolder')
    plt.ylabel('Relative frequency (%)')
    plt.legend(title = 'Word class')
    plt.tight_layout()
    plt.savefig(outpath)
    return print("The plot has been saved to the out folder")


def plot_entities(combined_df, outpath):
    """
    The function plots the average number of unique entities (PERSON, LOC, ORG) across subfolders.
    The plot will be saved to the specified outpath.
    """
    cols = ['No. Unique PER', 'No. Unique LOC', 'No. Unique ORG']
    avg_ents = combined_df.groupby('subfolder')[cols].mean().reset_index()
    avg_ens_melted = pd.melt(avg_ents, id_vars = 'subfolder',  var_name = 'Entity', value_name = 'Average count')

    plt.figure(figsize = (10, 6))
    sns.barplot(data = avg_ens_melted, x = 'subfolder', y = 'Average count', hue = 'Entity', palette = 'viridis')
    plt.title('Entities across subfolders')
    plt.xlabel('Subfolder')
    plt.ylabel('Average number of unique entities')
    plt.legend(title = 'Entity')
    plt.tight_layout()
    plt.savefig(outpath)
    return print("The plot has been saved to the out folder")


def main():
    
    filepath = os.path.join("..", "..", "..", "..", "cds-lang-data", "USEcorpus", "USEcorpus") # "in", "USEcorpus"

    tracker = emissions_tracker("../assignment-5/out")
    tracker.start()

    tracker.start_task("load spacy model")
    nlp = spacy.load("en_core_web_md")
    emissions_1_load_model = tracker.stop_task()

    tracker.start_task("load data, process text, and save results")
    results = process_text(filepath, nlp)
    emissions_1_process_save = tracker.stop_task()

    tracker.start_task("plot results")
    dataframes = ['out/a1_data.csv', 'out/a2_data.csv', 'out/a3_data.csv', 'out/a4_data.csv', 'out/a5_data.csv',
                'out/b1_data.csv', 'out/b2_data.csv', 'out/b3_data.csv', 'out/b4_data.csv', 'out/b5_data.csv',
                'out/b6_data.csv', 'out/b7_data.csv', 'out/b8_data.csv', 'out/c1_data.csv']
    combined_df = combine_df(dataframes)
    plot_word_class(combined_df, "out/wordclass.png")
    plot_entities(combined_df, "out/entity.png")
    emissions_1_plot = tracker.stop_task()

    tracker.stop()

if __name__ == "__main__":
    main()