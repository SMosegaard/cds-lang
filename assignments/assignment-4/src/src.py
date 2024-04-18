import pandas as pd
import tensorflow as tf
import matplotlib.pyplot as plt
import seaborn as sns
from transformers import pipeline, AutoModelForTokenClassification, AutoTokenizer


def load_classifier():
    """
    Load pre-trained text classification model from HuggingFace
    """
    classifier = pipeline("text-classification", 
                      model = "j-hartmann/emotion-english-distilroberta-base", 
                      return_all_scores = False)
    return classifier


def predict_emotion(df, classifier):
    """
    Predict emotions for each sentence in the dataframe
    """
    for index, row in df.iterrows():
        if type(row["Sentence"]) == str:
            predicted_emotion = classifier(row["Sentence"])[0]['label']
            df.at[index, "predicted_emotion"] = predicted_emotion
    return df


def plot_season(df, outpath):
    """
    Plot the distribution of predicted emotions across seasons
    """
    real_freq = df.groupby('Season')['predicted_emotion'].value_counts(normalize = True) * 100
    plot = sns.catplot(data = df, x = "predicted_emotion", hue = "predicted_emotion", col = "Season", kind = "bar", 
                palette = "husl", legend = False)
    plot.set_axis_labels("", "Relative Frequency (%)")
    plot.set_titles("{col_name}")
    plt.savefig(outpath)
    return print("The 'season' plot has been saved to the out folder")


def plot_emotion(df, outpath):
    """
    Plot the relative frequency of each emotion across all seasons
    """
    real_freq = df.groupby('predicted_emotion')['Season'].value_counts(normalize = True) * 100
    plot = sns.catplot(data = df, x = "Season", hue = "Season", col = "predicted_emotion", kind = "bar",
                    palette = "husl")
    plot.set_axis_labels("", "Relative Frequency (%)")
    plot.set_titles("{col_name}")
    plt.savefig(outpath)
    return print("The 'emotion' plot has been saved to the out folder")


def save_df_to_csv(df, csv_outpath):
    """
    Save the dataframe with predicted emotions as .csv
    """
    df.to_csv(csv_outpath)
    return print("The dataframe with the predicted emotion has been saved to the out folder")


def main():

    df = pd.read_csv("in/Game_of_Thrones_Script.csv")
    classifier = load_classifier()
    df = predict_emotion(df, classifier)
    save_df_to_csv(df, "out/data.csv")
    plot_season(df, "out/season2.png")
    plot_emotion(df, "out/emotion2.png")

if __name__ == "__main__":
    main()
