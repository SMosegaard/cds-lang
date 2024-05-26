# Assignment 2: Text Classification Benchmarks
*By Sofie Mosegaard, 07-03-2023*

This assignment is designed to train two benchmark Machine Learning (ML) classifiers on text data and assess their performances using ```scikit-learn```. The repository will utilize supervised ML techniques on a binary classification task, where a Logistic Regression (LR) and Neural Network (NN) will learn patterns from labelled data to make predictions on new, unseen data. 

The two ML models will be trained to classify whether news data is "real" or "fake" by doing the following steps:
1. Data preparation (```vectorizer.py```):
    - Loads and splits the data into training and testing sets using an 80-20 split.
    - Defines and saves a TF-IDF vectorizer object. 
    - Fits the training data, transforms the training and test data, and saves the vectorized data.
2.  Load vectorised data (```LR_classifier.py```, ```NN_classifier.py```):
    - Loads the vectorized data if it exists. If not, it runs the vectorizer.py script.
3. Model definition
    - Defines the LR and NN (i.e., the Multi-Layer Perceptron (MLP)) classifier with default parameters using a standard pipeline of scikit-learn.
4. Hyperparameter tuning:
    - Optionally, performs GridSearch to tune the hyperparameters through k-fold cross-validation with 5 folds to improve classification accuracy and robustness. For the LR classifier, the tolerance, maximum number of iterations, solver, and penalty will be tuned, while the number of hidden layers, activation, solver, and initial learning rate will be tuned for the NN classifier. 
5. Model training:
    - Fits the classifiers with default or tuned parameters to the training data.
6. Model evaluation:
    - Evaluates the trained classifiers on unseen test data.
7. Generate results:
    - Generates classification reports and saves them further analysis.
    - For the NN classifier, the training loss and validation accuracy curves will be plotted and saved.
    - For the LR classifier, the SHAP framework will be employed to create and save a summary plot of influential features. 
    - Optionally, conducts permutation tests to assess statistical significance of the classifiers' performance.

To better understand the code, all functions in the scripts ```src/XXX.py``` will include a brief descriptive text.

## Data

In this repository, the two classification models will be trained on the 'Fake or Real News dataset'. As the classification task is binary, the two classes are whether the news is real or fake.

You can download the dataset [here](https://www.kaggle.com/datasets/jillanisofttech/fake-or-real-news) and place it in the ```in``` folder. Remember to unzip the data before proceeding with the analysis.

## Project Structure

The repository includes:

- 2 bash scripts for setting up the virtual environment, installing requirements, and execution of the code
- 1 .txt file listing the required packages with versioned dependencies
- 1 README.md file
- 4 folders
    - in: holds the raw data to be processed
    - src: contains the Python code for performing the repository objective
    - out: stores the saved results, including classification reports in .txt format for both benchmark models and visualizations as .png files.
    - models: stores the vectorizer, vectorizered data, and trained models

## Setup and Installation Guide

To ensure reproducibility and facilitate collaboration, follow these steps:

1.  Clone the repository using the following command:
```python
$ git clone "https://github.com/SMosegaard/cds-lang/tree/main/assignments/assignment-2"
```
2. Navigate into the 'assignment-2' folder in your terminal:
```python
$ cd assignment-2
```
3. Execute the 'setup.sh' script to create a virtual environment and install the necessary packages listed in 'requirement.txt':
```python
$ source setup.sh
``` 

## Getting Started

Execute the 'run.sh' script in the terminal, specifying whether you want to conduct GridSearch (--GridSearch / -gs) and/or permutation testing (--PermutationTest / -pt):
```python
$ source run.sh -gs {yes/no} -pt {yes/no}
``` 
*Please note that the hyperparameter tuning and permutation testing will take some time to perform.*

The inputs will be converted to lowercase, so it is irrelevant whether it is spelled with or without capital letters.

Both classifiers will be executed sequentially. To run a specific model, you can uncomment the corresponding script within the run.sh file.

Upon completion, a message will be displayed in the terminal output, confirming that the results have been successfully saved.

## Results

The table below displays the performance of the logistic regression and neural network models in the binary classification task utilizing both default and tuned parameters:

|model|accuracy|macro accuracy|weighted accuracy|
|---|---|---|---|
|LR_default {'tol': 0.0001, 'max_iter': 100, 'solver': 'lbfgs', 'penalty': 'l2'}|0.89|0.89|0.89|
|LR_GridSearch {'tol': 0.01, 'max_iter': 100, 'solver': 'saga', 'penalty': 'l1'}|0.90|0.90|0.90|
|NN_default {'hidden_layer_sizes': 100, 'activation': 'relu', 'solver': 'adam', 'learning_rate_init': 0.001}|0.89|0.89|0.89|
|NN_GridSearch {'hidden_layer_sizes': '150', 'activation': 'relu', 'solver': 'adam', 'learning_rate_init': 0.01}|0.90|0.90|0.90|

*The full classification reports and visualisations can be found in the ```out``` folder.*

Both the LR and NN classifiers with default parameters archives high average accuracy scores of 89-90% with balanced performance between both classes. When tuning the hyperparameters, no significant improvements in classification accuracy is detected.

The training loss and validation accuracy curves of the NN classifiers, respectively with default and tuned parameters, were visualized to assess the models training process and performance:

<p align = "center">
    <img src = "https://raw.githubusercontent.com/SMosegaard/cds-lang/main/assignments/assignment-2/out/NN_loss_curve.png" width = "400">
    <img src = "https://raw.githubusercontent.com/SMosegaard/cds-lang/main/assignments/assignment-2/out/NN_loss_curve_GS.png" width = "400">
</p>

The decreasing loss curves and increasing validation accuracies demonstrate that both models are learning effectively. As the trainings progress, the models improve their understanding of the training data, while still being able to generalize its knowledge to new, unseen data. However, the model with default parameters do show the best fit.

Although both benchmark models prove excellent performance in the binary classification tasks and significantly surpasses the chance level of 50%, it is relevant to assess whether the results are statistically significant. Therefore, the models were permutation tested:

<p align = "center">
    <img src = "https://raw.githubusercontent.com/SMosegaard/cds-lang/main/assignments/assignment-2/out/LR_permutation.png" width = "400">
    <img src = "https://raw.githubusercontent.com/SMosegaard/cds-lang/main/assignments/assignment-2/out/NN_permutation.png" width = "400">
</p>

The permutation tests confirmed that the models are statistically independent, and the obtained results are statistically better than expected by chance.

Finally, the methodology SHAP (SHapley Additive exPlanations) was introduced to provide an overview of the most important features and explain the predictions of the classifier. The impact of the most influential features is summarised:

<p align = "center">
    <img src = "https://raw.githubusercontent.com/SMosegaard/cds-lang/main/assignments/assignment-2/out/LR_shap_summary.png" width = "400">
</p>

The plot illustrate which words are most influential in determining whether the news is real or fake and how they affect the model's predictions. Each point represents a data record, with features ranked by importance from most to least significant. Higher feature values typically have a more positive impact on the prediction.

The words "2016" and "October" have numerous red points with negative SHAP values, which suggest that the words contribute to predicting the news as "fake." In contrast, words like "said" and "but" often appear more frequently in articles that the model predicts as "real" news.

The SHAP framework is not compatible with MLP from scikit-learn, in which the summarised features are extracted from the LR model. However, it can be assumed that the results would look somewhat the same as both classifiers are trained on the same dataset and elicit almost identical results.

## Discussion

The results of this study provide insights into the performance of two benchmark models in a binary classification task. Despite the inherent complexity of the NN architecture, both models demonstrated comparable performance. The superior results suggest that both models are well-suited for the binary classification task at hand. 

Several factors could contribute to the similar performance of the two models. Firstly, the binary classification task was relatively straightforward and may not have been sufficiently complex to fully exploit the capabilities of the NN model. This allowed the simpler LR algorithm to learn and generalize effectively. Additionally, the data was well-structured and cleaned, which meant that the LR classifier could capture the underlying patterns without the need for more complex modeling. 

A slight improvement was observed when implementing GridSearch for hyperparameter tuning. This could be due to cross validation, that improves the robustness of the model. However, given the simplicity of the task, the quality of the dataset, and the almost identical results, it is difficult to derive the real effect of the individual parameters.

The findings demonstrate the importance of model selection based on the specific problem domain and data characteristics. While NN models have gained popularity due to their impressive performance in various complex tasks, simpler models like LR should not be overlooked. Future work could involve exploring more complex datasets to identify scenarios where NNs superior performance becomes evident.

*CodeCarbon was implemented to monitor carbon emissions associated with code execution. The results will be saved and discussed in assignment 5.*