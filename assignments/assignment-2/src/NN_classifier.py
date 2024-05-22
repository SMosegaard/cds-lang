import os
import sys
import pandas as pd
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import GridSearchCV, permutation_test_score
from sklearn import metrics
import matplotlib.pyplot as plt
import joblib
from joblib import dump, load
import argparse
from codecarbon import EmissionsTracker
import subprocess


def emissions_tracker(outpath):
    """
    The function initializes an EmissionsTracker object to track carbon emissions associated
    with code execution. The results of this can be found in assignment 5.
    """
    tracker = EmissionsTracker(project_name = "assignment 2",
                                output_dir = outpath)
    return tracker


def parser():
    """
    The user can specify whether to perform GridSearch and/or permutation testing when executing
    the script. The function will then parse command-line arguments and make them lower case.
    """
    parser = argparse.ArgumentParser()
    parser.add_argument("--GridSearch",
                        "-gs",
                        required = True,
                        help = "Perform GridSearch (yes or no)")
    parser.add_argument("--PermutationTest",
                        "-pt",
                        required = True,
                        help = "Perform permutation test (yes or no)")    
    args = parser.parse_args()
    args.GridSearch = args.GridSearch.lower()
    args.PermutationTest = args.PermutationTest.lower()
    return args


def load_vectorised_data(tracker):
    """
    The function loads the vectorized data if it exists. If it does not, it runs vectorizer.py using the
    Python subprocess module to create it. Then, it loads the created vectorized data
    """
    tracker.start_task("Load vectorized data")
    if not os.path.isfile( 'models/vectorized_data.pkl'):
        result = subprocess.run(['python', 'src/vectorizer.py'], capture_output = True, text = True)
    
    if os.path.isfile('models/vectorized_data.pkl'):
        vectorized_data = joblib.load('models/vectorized_data.pkl')
        X_train_features, y_train, X_test_features, y_test = vectorized_data

    emissions_2_NN_load_vectorized_data = tracker.stop_task()
    return X_train_features, y_train, X_test_features, y_test


def define_classifier(tracker):
    """
    The function defines the neural network classifier with default parameters. The parameters are therefore
    only specified for illustration purposes.
    Additionally, 10% of the training data will be used for validation. When the validation score is
    not improving during training, the training will stop due to early stopping.
    """
    tracker.start_task("Define classifier")
    classifier = MLPClassifier(hidden_layer_sizes=(100,),
                                activation = "relu",
                                solver = "adam",
                                learning_rate_init = 0.001,
                                early_stopping = True,
                                random_state = 123,
                                verbose = True)

    emissions_2_NN_define_classifier = tracker.stop_task()
    return classifier


def grid_search(classifier, X_train, y_train, tracker):
    
    tracker.start_task("GridSearch")

    param_grid = {'hidden_layer_sizes': [50, 100, 150],
                'activation': ('logistic', 'relu'),
                'solver': ('adam', 'sgd'),
                'learning_rate_init': [0.01, 0.001, 0.0001]}

    grid_search = GridSearchCV(estimator = classifier, param_grid = param_grid, cv = 5, n_jobs = -1)
    grid_result = grid_search.fit(X_train, y_train)

    print(f'Best Accuracy for {grid_result.best_score_} using the parameters {grid_result.best_params_}')

    means = grid_result.cv_results_['mean_test_score']
    stds = grid_result.cv_results_['std_test_score']
    params = grid_result.cv_results_['params']
    for mean, stdev, param in zip(means, stds, params):
        print(f' mean={mean:.4}, std={stdev:.4} using {param}')

    best_estimator = grid_result.best_estimator_
    emissions_2_NN_grid_search = tracker.stop_task()
    return best_estimator
 

def fit_classifier(classifier, X_train, y_train, tracker, classifier_path):
    """
    The function fits the LR classifier to the training data.
    - fits either the vectorised data to default LR parameters or parameters obtained through GridSearch
    """
    tracker.start_task("Fit classifier")
    classifier = classifier.fit(X_train, y_train)
    dump(classifier, f"{classifier_path}.joblib")
    emissions_2_NN_fit_classifier = tracker.stop_task()
    return classifier


def evaluate_classifier(classifier, X_train_features, y_train, X_test_features, y_test, tracker, outpath):
    """
    The function evaluates the trained classifier on new, unseen data. This includes plotting a confusion
    matrix and calculating a classification report, which will be saved to a specified outpath.
    """
    tracker.start_task("Evaluate classifier")
    y_pred = classifier.predict(X_test_features)

    classifier_metrics = metrics.classification_report(y_test, y_pred, target_names = ["FAKE", "REAL"])
    print(classifier_metrics)

    with open(outpath, 'w') as file:
        file.write(classifier_metrics)
    emissions_2_NN_evaluate_classifier = tracker.stop_task()
    return print("The classification report has been saved to the out folder")


def plot_loss_curve(classifier, tracker, outpath):
    """
    Plots the training loss and validation accuracy curves and saves the plot to a specified outpath.
    """
    tracker.start_task("Plot loss curve")
    plt.figure(figsize = (12, 6))
    plt.subplot(1, 2, 1)
    plt.plot(classifier.loss_curve_)
    plt.title("Loss curve during training for the neural network classifier", fontsize = 10)
    plt.ylabel('Loss score', fontsize = 9)
    plt.xlabel("Iterations", fontsize = 9)

    plt.subplot(1, 2, 2)
    plt.plot(classifier.validation_scores_)
    plt.title("Accuracy curve during validation for the neural network classifier", fontsize = 10)
    plt.ylabel('Accuracy', fontsize = 9)
    plt.xlabel("Iterations", fontsize = 9)

    plt.savefig(outpath)
    plt.show()
    emissions_2_NN_plot = tracker.stop_task()
    return print("The loss curve has been saved to the out folder")


def permutation_test(classifier, X_train_features, y_train, tracker, outpath):
    """
    Performs permutation test on the LR classifier to assess statistical significance of classifier's
    performance. The permutation test will be plotted and saved to a specified outpath.
    """
    tracker.start_task("Permutation test")
    score, permutation_scores, pvalue = permutation_test_score(classifier, X_train_features, y_train,
                                                                cv = 5, n_permutations = 100,
                                                                n_jobs = -1, random_state = 123,
                                                                verbose = True, scoring = None)
    n_classes = 2

    plt.figure(figsize = (8, 6))
    plt.hist(permutation_scores, 20, label = 'Permutation scores', edgecolor = 'black')
    ylim = plt.ylim()
    plt.plot(2 * [score], ylim, '--g', linewidth = 3,label = 'Classification Score'' (pvalue %s)' % pvalue)
    plt.plot(2 * [1. / n_classes], ylim, '--k', linewidth = 3, label = 'Chance level')
    plt.title("Permutation test neural network classifier")
    plt.ylim(ylim)
    plt.legend()
    plt.xlabel('Score')
    plt.savefig(outpath)
    plt.show()
    emissions_2_NN_permutation_test = tracker.stop_task()
    return print("The permutation test has been saved to the out folder")


def main():
    
    tracker = emissions_tracker("../assignment-5/out")

    args = parser()

    X_train_features, y_train, X_test_features, y_test = load_vectorised_data(tracker)

    classifier = define_classifier(tracker)

    if args.GridSearch == 'yes':
        classifier = grid_search(classifier, X_train_features, y_train, tracker)

    fit_classifier(classifier, X_train_features, y_train, tracker, "models/NN_classifier")

    evaluate_classifier(classifier, X_train_features, y_train, X_test_features, y_test,
                        tracker, "out/NN_classification_report.txt")

    if args.PermutationTest == 'yes':
        permutation_test(classifier, X_test_features, y_test, tracker, "out/NN_permutation.png")

    tracker.stop()

if __name__ == "__main__":
    main()