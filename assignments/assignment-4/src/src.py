import pandas as pd
import matplotlib
import tensorflow as tf
import matplotlib.pyplot as plt
import seaborn as sns
from transformers import pipeline, AutoModelForTokenClassification, AutoTokenizer

# Dictionary for colors mapped to emotions
emotion_colors = {
                "disgust": "purple",
                "surprise": "yellow",
                "neutral": "lightgray",
                "fear": "orange",
                "anger": "red",
                "joy": "green",
                "sadness": "blue"}


def load_classifier():
    classifier = pipeline("text-classification", 
                      model = "j-hartmann/emotion-english-distilroberta-base", 
                      return_all_scores = False)
    return classifier


def predict_emotion(df, classifier):
    for index, row in df.iterrows():
        if type(row["Sentence"]) == str:
            predicted_emotion = classifier(row["Sentence"])[0]['label']
            df.at[index, "predicted_emotion"] = predicted_emotion
    return df

def plot_season(df, emotion_colors, outpath):
    total_counts = df.groupby('Season')['predicted_emotion'].value_counts(normalize = True) * 100
    plot = sns.catplot(data = df, x = "predicted_emotion", hue = "predicted_emotion", col = "Season", kind = "count", 
                palette = emotion_colors.values() , col_wrap = 4) #lengend = False
    plot.set_axis_labels("", "Relative Frequency (%)")
    plot.set_titles("{col_name}")
    plt.savefig(outpath)
    return print("The 'season' plot has been saved to the out folder")

def plot_emotion(df, outpath):
    total_counts = df.groupby('predicted_emotion')['Season'].value_counts(normalize = True) * 100
    plot = sns.catplot(data = df, x = "Season", hue = "Season", col = "predicted_emotion", kind = "count",
                    palette = "husl", col_wrap = 4)
    plot.set_axis_labels("", "Relative Frequency (%)")
    plot.set_titles("{col_name}")
    plt.savefig(outpath)
    return print("The 'emotion' plot has been saved to the out folder")


def save_df_to_csv(df, csv_outpath):
    df.to_csv(csv_outpath)
    return print("The dataframe with the predicted emotion has been saved to the out folder")


def main():

    df = pd.read_csv("../../../../cds-lang-data/GoT-scripts/Game_of_Thrones_Script.csv")
    
    classifier = load_classifier()

    df = df.head(10)

    df = predict_emotion(df, classifier)
    print(df)

    save_df_to_csv(df, "out/data.csv")

    plot_season(df, emotion_colors, "out/season.png")

    plot_emotion(df, "out/emotion.png")

if __name__ == "__main__":
    main()
