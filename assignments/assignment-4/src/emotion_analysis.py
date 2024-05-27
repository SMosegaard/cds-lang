import pandas as pd
import tensorflow as tf
import matplotlib.pyplot as plt
import seaborn as sns
from transformers import pipeline, AutoModelForTokenClassification, AutoTokenizer
from codecarbon import EmissionsTracker


def emissions_tracker(outpath):
    """
    The function initializes an EmissionsTracker object to track carbon emissions associated
    with code execution. The results of this can be found in assignment 5.
    """
    tracker = EmissionsTracker(project_name = "assignment 4",
                                output_dir = outpath,
                                output_file = "emissions_assignment_4")
    return tracker


def load_classifier(tracker):
    """
    The function loads the pre-trained text classification model from HuggingFace.
    """
    tracker.start_task("load classifier")
    classifier = pipeline("text-classification", 
                      model = "j-hartmann/emotion-english-distilroberta-base", 
                      return_all_scores = False)
    emissions_4_load_classifier = tracker.stop_task()
    return classifier


def predict_emotion(df, classifier, tracker):
    """
    The function predicts emotions for each sentence in the dataframe using the provided classifier.
    """
    tracker.start_task("predict emotion")
    for index, row in df.iterrows():
        if type(row["Sentence"]) == str:
            predicted_emotion = classifier(row["Sentence"])[0]['label']
            df.at[index, "predicted_emotion"] = predicted_emotion
    emissions_4_predict_emotion = tracker.stop_task()
    return df


def save_df_to_csv(df, tracker, csv_outpath):
    """
    The function saves the dataframe with predicted emotions as .csv to a specified outpath.
    """
    tracker.start_task("save results")
    df.to_csv(csv_outpath)
    emissions_4_save_results = tracker.stop_task()
    return print("The dataframe with the predicted emotion has been saved to the out folder")


def count_emotions(df, tracker):                            
    """
    The function counts the occurrences of each predicted emotion for each season in the input dataframe.
    A dataframe showing the count and relative frequency of each predicted emotion for each season will
    be returned and used for plotting.
    """
    tracker.start_task("reshape data")
    season_list = df["Season"].unique()
    season_list_len = [len(df[df["Season"] == season]) for season in season_list]
    predicted_emotion_count = df.groupby(["Season", "predicted_emotion"]).size().reset_index(name = "count")

    predicted_emotion_count["Relative Frequency"] = ""
    for season, length in zip(season_list, season_list_len):
        predicted_emotion_count.loc[predicted_emotion_count["Season"] == season, "Relative Frequency"] = \
            round(predicted_emotion_count.loc[predicted_emotion_count["Season"] == season, "count"] / length * 100, 2)
    emissions_4_count_emotions = tracker.stop_task()
    return predicted_emotion_count


def plot_season(data, tracker, outpath):
    """
    The function creates a plot showing the relative frequency of each predicted emotion for each season.
    The plot will be saved to a specified outpath.
    """
    tracker.start_task("plot season")
    
    seasons = sorted(data["Season"].unique())
    emotions = sorted(data["predicted_emotion"].unique())

    g = sns.catplot(data, x = "predicted_emotion", y = "Relative Frequency", hue = "predicted_emotion",  
                    col = "Season", kind = "bar", col_wrap = 4, col_order = seasons, order = emotions,
                    palette = "husl")
    g.set_axis_labels("", "Relative frequency (%)")
    g.set_titles("{col_name}")
    g.set_xticklabels(labels = emotions, rotation = 45)
    plt.tight_layout()
    plt.savefig(outpath)
    plt.show()
    emissions_4_plot_season = tracker.stop_task()
    return print("The 'season' plot has been saved to the out folder")


def plot_emotion(data, tracker, outpath):
    """
    The function creates a plot showing the relative frequency of each season for each predicted emotion.
    The plot will be saved to a specified outpath.
    """
    tracker.start_task("plot emotion")

    emotion_totals = data.groupby("predicted_emotion")["count"].sum().reset_index(name = "total_count")
    data = data.merge(emotion_totals, on = "predicted_emotion")
    data["Relative Frequency"] = round(data["count"] / data["total_count"] * 100, 2)

    seasons = sorted(data["Season"].unique())
    emotions = sorted(data["predicted_emotion"].unique())
    palette = sns.color_palette("husl", len(seasons))

    g = sns.catplot(data, x = "Season", y = "Relative Frequency", hue = "Season",  
                    col = "predicted_emotion", kind = "bar", col_wrap = 4, col_order = emotions, 
                    order = seasons, palette = "husl", legend = False)
    g.set_axis_labels("", "Relative frequency (%)")
    g.set_titles("{col_name}")
    g.set_xticklabels(labels = seasons, rotation = 45) 
    plt.tight_layout()
    plt.savefig(outpath)
    plt.show()
    emissions_4_plot_emotion = tracker.stop_task()
    return print("The 'emotion' plot has been saved to the out folder")


def main():

    tracker = emissions_tracker("../assignment-5/out")
    tracker.start()

    df = pd.read_csv("in/Game_of_Thrones_Script.csv")
    classifier = load_classifier(tracker)
    df = predict_emotion(df, classifier, tracker)
    save_df_to_csv(df, tracker, "out/results.csv")

    df = pd.read_csv("out/results.csv")
    predicted_emotion_count = count_emotions(df, tracker)                         
    plot_season(predicted_emotion_count, tracker, "out/season.png")
    plot_emotion(predicted_emotion_count, tracker, "out/emotion.png")

    tracker.stop()

if __name__ == "__main__":
    main()