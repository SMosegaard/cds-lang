import os
import sys
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV, permutation_test_score
from sklearn import metrics
import matplotlib.pyplot as plt
import joblib
from joblib import dump, load
import argparse
from codecarbon import EmissionsTracker
import subprocess
import shap
shap.initjs()


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
        X_train_features, y_train, X_test_features, y_test, feature_names = vectorized_data

    emissions_2_LR_load_vectorized_data = tracker.stop_task()
    return X_train_features, y_train, X_test_features, y_test, feature_names


def define_classifier(tracker):
    """
    The function defines the neural network classifier with default parameters.
    The parameters are therefore simply specified for illustration purposes only.
    """
    tracker.start_task("Define classifier")
    classifier = LogisticRegression(tol = 0.0001,
                                    max_iter = 100,
                                    solver = 'lbfgs',
                                    penalty = 'l2',
                                    random_state = 123,
                                    verbose = True)

    emissions_2_LR_define_classifier = tracker.stop_task()
    return classifier


def grid_search(classifier, X_train, y_train, tracker):
    """
    The function performs GridSearch to find the best hyperparameters for the model.
    The best parameters will be printed in the terminal output and returned.
    """
    tracker.start_task("GridSearch")

    tol = [0.01, 0.001, 0.0001, 0.00001]
    max_iter = [100, 200, 300]

    param_grid = (
        [{'tol': tol, 'max_iter': max_iter, 'solver': ['saga', 'liblinear'], 'penalty': ['l1']},
        {'tol': tol, 'max_iter': max_iter, 'solver': ['saga', 'liblinear', 'lbfgs', 'newton-cg'], 'penalty': ['l2']}]
        )

    grid_search = GridSearchCV(estimator = classifier, param_grid = param_grid, cv = 5, n_jobs = -1)
    grid_result = grid_search.fit(X_train, y_train)

    print(f'Best Accuracy for {grid_result.best_score_} using the parameters {grid_result.best_params_}')

    means = grid_result.cv_results_['mean_test_score']
    stds = grid_result.cv_results_['std_test_score']
    params = grid_result.cv_results_['params']
    for mean, stdev, param in zip(means, stds, params):
        print(f' mean={mean:.4}, std={stdev:.4} using {param}')

    best_estimator = grid_result.best_estimator_
    emissions_2_LR_grid_search = tracker.stop_task()
    return best_estimator


def fit_classifier(classifier, X_train, y_train, tracker):
    """
    The function fits the LR classifier to the training data.
    """
    tracker.start_task("Fit classifier")
    classifier = classifier.fit(X_train, y_train)
    emissions_2_LR_fit_classifier = tracker.stop_task()

    return classifier


def evaluate_classifier(classifier, X_train_features, y_train, X_test_features, y_test, tracker, outpath):
    """
    The function evaluates the trained classifier on new, unseen data. This includes plotting calculating
    a classification report, which will be saved to a specified outpath.
    """
    tracker.start_task("Evaluate classifier")
    y_pred = classifier.predict(X_test_features)

    params = {key: classifier.get_params()[key] for key in classifier.get_params().keys() & {'tol', 'max_iter', 'solver', 'penalty'}}

    classifier_metrics = metrics.classification_report(y_test, y_pred, target_names = ["FAKE", "REAL"])
    print(classifier_metrics)

    full_report = f"The classifier utilized the parameters:{params}\n\n{classifier_metrics}"

    with open(outpath, 'w') as file:
        file.write(full_report)

    emissions_2_LR_evaluate_classifier = tracker.stop_task()
    return print("The classification report has been saved to the out folder")


def shap_explainer(classifier, X_train_features, X_test_features, feature_names, tracker, outpath):
    """
    The function uses the SHAP framework to explain the predictions of the classifier by generating a SHAP
    summary plot that visualizes the impact of each feature on the model's output.
    The plot will be saved to a specified outpath.
    """
    tracker.start_task("Shap")
    explainer = shap.LinearExplainer(classifier, X_train_features)
    shap_values = explainer.shap_values(X_test_features)
    X_test_array = X_test_features.toarray() 
    shap.summary_plot(shap_values, X_test_array, feature_names = feature_names, show = False)
    plt.savefig("out/LR_shap_summary.png", dpi = 150, bbox_inches = 'tight')
    emissions_2_LR_shap = tracker.stop_task()
    return print("The SHAP summary plot has been saved to the out folder")


def permutation_test(classifier, X_train_features, y_train, tracker, outpath):
    """
    The function performs permutation test on the LR classifier to assess statistical significance
    of classifier's performance. The permutation test will be plotted and saved to a specified outpath.
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
    plt.title("Permutation test logistic regression classifier")
    plt.ylim(ylim)
    plt.legend()
    plt.xlabel('Score')
    plt.savefig(outpath)
    plt.show()
    emissions_2_LR_permutation_test = tracker.stop_task()
    return print("The permutation test has been saved to the out folder")


def main():
    
    tracker = emissions_tracker("../assignment-5/out")

    args = parser()

    X_train_features, y_train, X_test_features, y_test, feature_names = load_vectorised_data(tracker)

    classifier = define_classifier(tracker)

    if args.GridSearch == 'yes':
        classifier = grid_search(classifier, X_train_features, y_train, tracker)
        classifier = fit_classifier(classifier, X_train_features, y_train, tracker)
        dump(classifier, "models/LR_classifier_GS.joblib")
        evaluate_classifier(classifier, X_train_features, y_train, X_test_features, y_test, tracker,
                            "out/LR_classification_report_GS.txt")

    else:
        classifier = fit_classifier(classifier, X_train_features, y_train, tracker)
        dump(classifier, "models/LR_classifier.joblib")
        evaluate_classifier(classifier, X_train_features, y_train, X_test_features, y_test, tracker,
                            "out/LR_classification_report.txt")

    shap_explainer(classifier, X_train_features, X_test_features, feature_names, tracker, "out/LR_shap.txt")
    
    if args.PermutationTest == 'yes':
        permutation_test(classifier, X_test_features, y_test, tracker, "out/LR_permutation.png")

    tracker.stop()

if __name__ == "__main__":
    main()