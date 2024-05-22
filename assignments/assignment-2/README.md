# Assignment 2: Text Classification Benchmarks
*By Sofie Mosegaard, 07-03-2023*

This assignment is designed to train two benchmark, machine learning classifiers on text data and assess their performances using ```scikit-learn```. The repository will utilize supervised machine learning techniques on a binary classification task, where a Logistic Regression (LR) and Neural Network (NN) will learn patterns from labeled data to make predictions on unseen data. 

The two machine learning models will be trained to classify whether news data is "real" or "fake" by doing the following:


- the vectorizer script will check for the existence of saved vectorized data and either skip or execute the code to create the objects based on that check. 



To better understand the code, each function in the script ```feature_extractor.py`` will include a brief descriptive text.

## Data source

In this repository, the two classification models will be trained on the 'Fake or Real News dataset'. 
The dataset consists of ... articles....
As the classification task is binrary, the two classes are whether the news are real or fake.

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



.....


The training loss and and validation accuracy curves of the NN classifier was visualized to assess the models training process and performance:

<p align = "center">
    <img src = "https://github.com/SMosegaard/cds-lang/blob/main/assignments/assignment-2/out/NN_loss_curve.png" width = "600">
</p>

The decreasing loss curve and increasing validation accuracy monstrates that the model is learning effectively. As training progresses, the model improves its understanding of the training data, while also being able to generalize its knowledge to new, unseen data.

Although both benchmark models demonstate excellent perfomance in the binary classification tasks and significantly surpasses the chance level of 50%, it is relevanty to assess whether the results are statistically significant. Therefore, the models were permutation tested:

<p align = "center">
    <img src = "https://github.com/SMosegaard/cds-lang/blob/main/assignments/assignment-2/out/LR_permutation.png" width = "400">
    <img src = "https://github.com/SMosegaard/cds-lang/blob/main/assignments/assignment-2/out/NN_permutation.png" width = "400">
</p>

The permutation tests confirmed that the models are statistically independent and the obtained results are statistically better than expected by chance.

## Discussion

- the NN outperforms the LG model by achieving higher accuracy and demonstrating more balanced performance across labels
- NN more complex omplex architecture and superiority at learning relationships and patterns




*CodeCarbon was implemented to monitor carbon emissions associated with code execution. The results will be saved and discussed in assignment 5.*