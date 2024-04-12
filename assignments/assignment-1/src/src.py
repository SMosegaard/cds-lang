import os
import pandas as pd
import glob
import re
import spacy


nlp = spacy.load("en_core_web_md")


def remove_metadata(text):
    return re.sub(r"<*?>", "", text)


def count_pos(doc):
    """
    Count the number of each part-of-speech (POS) tag in the document. The function takes a spaCy doc
    object as input and returns a tuple containing counts of nouns, verbs, adjectives, and adverbs.
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
    Calculate the relative frequency per 10,000 words and round the decimals. The function takes the number of
    POS and the total number of tokens in the given text, while returns the relative frequency.
    """
    return round((count/len_doc * 10000), 2)


def no_unique_ents(doc):
    """
    The function counts the total number of unique PER, LOC, and ORG entities. The function takes a spaCy doc
    object as input and returns a list containing counts of unique entities.

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



def process_text(filepath):

    """ 
    The function iterates over .txt files in the subfolders and extracts linguistic features. Also, it creates a
    Pandas DataFrame to store and append the extracted features for each file.
    """

    for subfolder in sorted(os.listdir(filepath)):
        subfolder_path = os.path.join(filepath, subfolder)

        out_df = pd.DataFrame(columns=("Filename", "RelFreq NOUN", "RelFreq VERB", "RelFreq ADV",
                                        "RelFreq ADJ", "No. Unique PER", "No. Unique LOC", "No. Unique ORG"))

        csv_outpath = os.path.join("out", f"{subfolder}.csv")

        for file in glob.glob(os.path.join(subfolder_path, "*.txt")):
            with open(file, "r", encoding="latin-1") as f:
                text = f.read()
                doc = nlp(remove_metadata(text))

            noun_count, verb_count, adv_count, adj_count = count_pos(doc)
            len_doc = len(doc)
            noun_rel_freq, verb_rel_freq, adv_rel_freq, adj_rel_freq = rel_freq(noun_count, len_doc), rel_freq(verb_count, len_doc), rel_freq(adv_count, len_doc), rel_freq(adj_count, len_doc)
            No_unique_per, No_unique_loc, No_unique_org = no_unique_ents(doc)
            text_name = text.split("/")[-1]
            text_row = [text_name, noun_rel_freq, verb_rel_freq, adv_rel_freq, adj_rel_freq,
                        No_unique_per, No_unique_loc, No_unique_org]
            out_df.loc[len(out_df)] = text_row
        
        #print(out_df)
        

def save_dataframe_to_csv(df, csv_outpath):
    df.to_csv(csv_outpath)


def main():

    filepath = os.path.join("in", "USEcorpus")
    results = process_text(filepath)
    
    csv_outpath = os.path.join("out", f"{subfolder}_data_new.csv")
    save_dataframe_to_csv(results, csv_outpath)


if __name__ == "__main__":
    main()