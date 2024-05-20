# Portfolio 2: Text classification benchmarks
*By Sofie Mosegaard, 07-03-2023*

This assignment is designed to train two binary classification models on text data and assess their performances using ```scikit-learn```. 

The assignment has the objective:
-   Train simple benchmark machine learning classifiers on structured text data;
-   Produce understandable outputs and trained models which can be reused;
-   Save those results in a clear way which can be shared or used for future analysis

## Data source

insert link, post au


## Installation and requirements
-   Clone the repository: $ git clone "https://github.com/SMosegaard/cds-lang/tree/main/assignments/assignment-2"
-   Select Python 3 kernel   
-   Install the required packages (`pandas`, `seaborn`, `scikit-learn`, `matplot`, `joblib`)

## Usage

When cloned, your repository 'assignment 2' will contain four folders:
-   ```in```: contains the data 'fake_or_real_news.csv', which is texts from different news articles. Each article will have a label, that indicates whether it is 'real' or 'fake'.
-   ```src```: consist of *three different scripts*. In the first script (`assignment2_src1_vectorizer`), the data will be vectorized and the new feature extracted data will be saved as objects. By doing so, you only have to vectorize the data once instead of once per script, which can be timesaving when working with larger datasets. In the second script (`assignment2_src2_LR`), a Logistic Regression (LR) classifier will be trained and evaluated. Finnally, in the third script (`assignment2_src3_NN`), a Neural Network (NN) will be trained and evaluated.
-   ```models```:  contains the vectorizer and the feature extracted objects from script 1, as well as the trained models from script 2 and 3.
-   ```out```: contains classification reports for the LR and NN classifier.


*CodeCarbon was implemented to monitor carbon emissions associated with code execution. The results will be saved and discussed in portfolio 5.*