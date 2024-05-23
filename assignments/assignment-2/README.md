# Assignment 2: Text Classification Benchmarks
*By Sofie Mosegaard, 07-03-2023*

This assignment is designed to train two benchmark, machine learning classifiers on text data and assess their performances using ```scikit-learn```. The repository will utilize supervised machine learning techniques on a binary classification task, where a Logistic Regression (LR) and Neural Network (NN) will learn patterns from labeled data to make predictions on unseen data. 

The two machine learning models will be trained to classify whether news data is "real" or "fake" by doing the following:
1. Data preparation (```vectorizer.py```):
    - Loads and splits the data into training and testing sets using an 80-20 split.
    - Defines and saves a TF-IDF vectorizer object. 
    - Fits the training data, transforms the training and test data, and saves the vectorized data.
2.  Load vectorised data (```LR_classifier.py```, ```NN_classifier.py```):
    - Loads the vectorized data if it exists. If not, it runs the vectorizer.py script.
3. Model definition
    - Defines the LR and NN classifier with default parameters.
4. Hyperparameter tuning:
    - Optionally, performs GridSearch to tune the hyperparameters through k-fold cross-validation with 5 folds to improve classification accuracy and robustness. For the LR classifier, the tolorance, maximum number of iterations, solver, and penalty will be tuned, while the number of hidden layers, activation, solver, and initial learning rate will be tuned for the NN classifier. 
5. Model training:
    - Fits the classifiers with default or tuned parameters to the training data.
6. Model evaluation:
    - Evaluates the trained classifiers on unseen test data.
7. Generate results:
    - Generates classification reports and saves them further analysis.
    - For the NN classifier, the training loss and validation accuracy curves will be plotted and saved.
    - For the LR classifier, the SHAP framework will be employed to create and save a summary plot of influential features. 
    - Optionally, conducts permutation tests to assess statistical significance of the classifiers' performance.

To better understand the code, all functions in the scripts ```src/XXX.py`` will include a brief descriptive text.

## Data source

In this repository, the two classification models will be trained on the 'Fake or Real News dataset'. As the classification task is binrary, the two classes are whether the news are real or fake.

You can download the dataset [here](https://www.kaggle.com/datasets/jillanisofttech/fake-or-real-news) and place it in the ```in``` folder. Before executing the code, make sure to unzip the data.

## Repository structure

The repository includes:

- 2 bash scripts for setting up the virtual environment, installing requirements, and execution of the code
- 1 .txt file listing the required packages with versioned dependencies
- 1 README.md file
- 4 folders
    - in: holds the data to be processed
    - src: contains the Python code for execution
    - out: stores the saved results, including classification reports in .txt format for both benchmark models and visualizations as .png files.
    - models: stores the vectorizer, vectorizered data, and trained models

## Reproducibility

1.  Clone the repository:
```python
$ git clone "https://github.com/SMosegaard/cds-lang/tree/main/assignments/assignment-2"
```
2.  Navigate into the folder in your terminal:
```python
$ cd assignment-2
```
3. Execute the setup bash script to create a virtual environment and install necessary packages listed in the requirement.txt:
```python
$ source setup.sh
``` 

## Usage

Execute the run bash script in the terminal, specifying whether you want to conduct GridSearch (--GridSearch / -gs) and/or permutation testing (--PermutationTest / -pt):
```python
$ source run.sh -gs {yes/no} -pt {yes/no}
``` 
*Please note that the hyperparameter tuning and permutation testing will take some time to perform.*

The inputs will be converted to lowercase, so it is irrelevant whether it is spelled with or without capital letters.

Both classifiers will be executed sequentially. To run a specific model, you can uncomment the corresponding script within the run.sh file.

After the script completes running, it will display a message in the terminal output indicating that the results have been saved.

## Summary of results

The table below displays the performance of the logistic regression and neural network models in the binary classification task utilizing both default and tuned parameters:

|model|accuracy|macro accuracy|weighted accuracy|
|---|---|---|---|
|LR_default {'tol': 0.0001, 'max_iter': 100, 'solver': 'lbfgs', 'penalty': 'l2'}|0.89|0.89|0.89|
|LR_GridSearch {'tol': 0.01, 'max_iter': 100, 'solver': 'saga', 'penalty': 'l1'}|0.90|0.90|0.90|
|NN_default {'hidden_layer_sizes': 100, 'activation': 'relu', 'solver': 'adam', 'learning_rate_init': 0.001}
|NN_GridSearch {'hidden_layer_sizes': '150', 'activation': 'relu', 'solver': 'adam', 'learning_rate_init': 0.01}|0.90|0.90|0.90|

*The full classification reports and visualisations can be found in the ```out``` folder.*

Both the LR and NN classifiers with default parameters achives high average accuracy scores of 89-90% with balanced performance between both classes. When tuning the hyperparameters, no significant improvements in classification accuracy is detected.

The training loss and and validation accuracy curves of the NN classifiers, respectively with default and tuned parameters, were visualized to assess the models training process and performance:

<p align = "center">
    <img src = "https://github.com/SMosegaard/cds-lang/blob/main/assignments/assignment-2/out/NN_loss_curve.png" width = "400">
    <img src = "https://github.com/SMosegaard/cds-lang/blob/main/assignments/assignment-2/out/NN_loss_curve_GS.png" width = "400">
</p>

The decreasing loss curves and increasing validation accuracies monstrate that bith models are learning effectively. As the trainings progress, the models improve their understanding of the training data, while also being able to generalize its knowledge to new, unseen data. However, the model with default parameters do show the best fit.

Although both benchmark models demonstate excellent perfomance in the binary classification tasks and significantly surpasses the chance level of 50%, it is relevanty to assess whether the results are statistically significant. Therefore, the models were permutation tested:

<p align = "center">
    <img src = "https://github.com/SMosegaard/cds-lang/blob/main/assignments/assignment-2/out/LR_permutation.png" width = "400">
    <img src = "https://github.com/SMosegaard/cds-lang/blob/main/assignments/assignment-2/out/NN_permutation.png" width = "400">
</p>

The permutation tests confirmed that the models are statistically independent and the obtained results are statistically better than expected by chance.

Finally, the methodology SHAP (SHapley Additive exPlanations) was introduced to provide an overview of the most important features and explain the predictions of the classifier. The impact of the most influential features are summarised:

<p align = "center">
    <img src = "https://github.com/SMosegaard/cds-lang/blob/main/assignments/assignment-2/out/NN_shap_summary.png" width = "600">
</p>

The plot illustrate which words are most influential in determining whether the news is real or fake and how they affects the model's predictions. Each point represents a data record, with features ranked by importance from most to least significant. Higher feature values typically have a more positive impact on the prediction.

The words "2016" and "october" have numerous red points with negative SHAP values, which suggest that the words contribute to predicting the news as "fake." In contrast, words like "said" and "but" often appear more frequently in articles that the model predicts as "real" news.

The SHAP framework is not compatable with MLP from scikit-learn, in which the summarised features are extracted from the LR model. However, it can be assumed that the results would look somewhat the same as both classifiers are trained on the same dataset and elicit almost identical results.

## Discussion

The superior results suggest that both models are well-suited for the binary classification task at hand. Interestingly, despite the expected superiority of the NN with its complex architecture, no differences in performance between the models were found. Several factors may explain this outcome. As the binary classification task was relatively straightforward, it allowed the simple LR model to learn and perform as well as the more complex NN. Additionaly, the data was well-structured and clen, which allowed for a simpler algorithm.

A slight improvement was observed when implementing GridSearch for hyperparameter tuning. This could be due to cross validation, that improves the robustness of the model. However, given the simplicity of the task, the quality of the dataset, and the almost identical results, it is difficult to derive the real effect of the individual parameters.

*CodeCarbon was implemented to monitor carbon emissions associated with code execution. The results will be saved and discussed in assignment 5.*