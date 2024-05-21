# Portfolio 2: Text Classification Benchmarks
*By Sofie Mosegaard, 07-03-2023*

This assignment is designed to train two binary classification models on text data and assess their performances using ```scikit-learn```. 

The assignment has the objective:
-   Train simple benchmark machine learning classifiers on structured text data;
-   Produce understandable outputs and trained models which can be reused;
-   Save those results in a clear way which can be shared or used for future analysis

## Data source

In this repository, the two classification models will be trained on the 'Fake or Real News dataset'. 
The dataset consists of ... articles....
As the classification task is binrary, the two classes are whether the news are real or fake.

You can download the dataset [here](https://www.kaggle.com/datasets/jillanisofttech/fake-or-real-news) and place it in the ```in``` folder. Make sure to unzip the data within the folder before executing the code.

## Repository structure

The repository consists of the following elements:

- 2 bash scripts for setup of the virtual environments, installation of requirements, and execution of the code
- 1 .txt file specifying the required packages including versioned dependencies
- 1 README.md file
- 4 folders
    - in: contains data to be processed
    - src: consists of the Python code to be executed
    - out: stores the saved results, i.e., classification reports in .txt format for both benchmark models
    - models: stores the vectorizer, vectorizered data, and trained models

## Reproducibility

1.  Clone the repository
```python
$ git clone "https://github.com/SMosegaard/cds-lang/tree/main/portfolios/portfolio-2"
```
2.  Navigate into the folder in your terminal
```python
$ cd portfolio-2
```
3.  Run the setup bash script to create a virtual envoriment and install required packages specified in the requirement.txt:
```python
$ source setup.sh
``` 
4.  Run the run bash script in the terminal to execude the code:
```python
 $ source run.sh
``` 

Once the script has finished running, it will print that the results have been saved in the terminal output.

## Summary of results

## Discussion



*CodeCarbon was implemented to monitor carbon emissions associated with code execution. The results will be saved and discussed in portfolio 5.*