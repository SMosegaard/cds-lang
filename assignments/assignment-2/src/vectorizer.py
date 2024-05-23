import os
import sys
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
import joblib
from joblib import dump
from codecarbon import EmissionsTracker


def emissions_tracker(outpath):
    """
    The function initializes an EmissionsTracker object to track carbon emissions associated
    with code execution. The results of this can be found in assignment 5.
    """
    tracker = EmissionsTracker(project_name = "assignment 2",
                                output_dir = outpath)
    return tracker


def data_Load_split(tracker, filepath):
    """
    The function loads the .csv file from the specified folder path and extracts the 'text'
    column as features and the 'label' column as labels. Then, it splits the data into training
    and testing sets using an 80-20 split.
    """
    tracker.start_task("Load and split data")
    data = pd.read_csv(filepath)
    X = data["text"]
    y = data["label"]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 123)
    emissions_2_load_data = tracker.stop_task()

    return X_train, X_test, y_train, y_test


def define_vectorizer(tracker, vectorizer_path):
    """
    The function defines a TF-IDF vectorizer object. The vectorizer parameters include unigrams and bigrams,
    lowercase conversion, removal of the most common (top 5%) and least common (bottom 5%) words, and
    maximum features. The vectorizer will then be saved to a specified outpath.
    """
    tracker.start_task("Define vectorizer")
    vectorizer = TfidfVectorizer(ngram_range = (1,2), 
                                lowercase =  True, 
                                max_df = 0.95, 
                                min_df = 0.05, 
                                max_features = 500)
    dump(vectorizer, f"{vectorizer_path}.joblib")
    emissions_2_define_vectorizer = tracker.stop_task()

    return vectorizer


def fit_vectorizer(vectorizer, X_train, X_test, y_train, y_test, tracker, vectorized_data_path):
    """ 
    The function fits the TF-IDF vectorizer to the training data and transforms both the training
    and test data. The transformed data and labels are saved to a specified outpath
    """
    tracker.start_task("Fit vectorizer")

    X_train_features = vectorizer.fit_transform(X_train)
    X_test_features = vectorizer.transform(X_test)
    feature_names = vectorizer.get_feature_names_out()

    vectorized_data = [X_train_features, y_train, X_test_features, y_test, feature_names]
    joblib.dump(vectorized_data, f'{vectorized_data_path}.pkl')

    emissions_2_fit_vectorizer = tracker.stop_task()

    return print("The vectorizer and vectorized data have been saved to the model folder")


def main():
    
    tracker = emissions_tracker("../assignment-5/out")

    if os.path.isfile('models/vectorized_data.pkl'):
        print("The TF-IDF vectorizer object and vectorized data already excist")

    else:
        X_train, X_test, y_train, y_test = data_Load_split("in/fake_or_real_news.csv")
        vectorizer = define_vectorizer("models/tfidf_vectorizer")
        fit_vectorizer(vectorizer, X_train, X_test, y_train, y_test, "models/vectorized_data")
      
    tracker.stop() 

if __name__ == "__main__":
    main()
