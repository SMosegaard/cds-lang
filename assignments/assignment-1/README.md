# Assignment 1: Extracting Linguistic Features using spaCy
*By Sofie Mosegaard, 22-02-2023*

This repository is designed to utilize the open-source Natural Language Processing (NLP) framework ```spaCy``` to extract relevant linguistic information from text data.  

This project will specifically retrieve linguistic features from the Uppsala Student English (USE) corpus by doing the following:
- Loads the 'en_core_web_md' English spaCy model
- Initializes a dataframe to store the results
- Iterates over each text in each subfolder and performs...
    - Converts the text into a spaCy document and removes its metadata
    - Counts the occurrences of each part-of-speech (POS) tag (nouns, verbs, adverbs, adjectives)
    - Calculates the relative frequency per 10,000 words of each POS
    - Counts the number of unique entities (persons, locations, organisations)
- Saves a dataframe for each subfolder in the ```out``` folder.
- Visualizes the relative frequency of entities and word types across subfolders. Saves the plots in the ```out``` folder.

To better understand the code, each function in the script ```feature_extractor.py`` will include a brief descriptive text.

## Data source

In this repository, the linguistic features will be extracted from a series of essays. The dataset is a collection of english essays written by Swedish university students at the Uppsala University. The corpus consists of 1,489 essays with an average lenght of 820 words.

The essays are written at three different terms, which is reflected by the subfolder (a, b, and c), encompassing the period from 1999 to 2001.  

You can download the dataset [here](https://ota.bodleian.ox.ac.uk/repository/xmlui/handle/20.500.12024/2457) and place it in the ```in``` folder. Before executing the code, make sure to unzip the data.

## Repository structure

The repository includes:

- 2 bash scripts for setting up the virtual environment, installing requirements, and execution of the code
- 1 .txt file listing the required packages with versioned dependencies
- 1 README.md file
- 3 folders
    - in: holds the data to be processed
    - src: contains the Python code for execution
    - out: stores the saved results, including a dataframe in .csv format for each sub-folder of data and visualizations of the results as .png files.

## Reproducibility

1.  Clone the repository:
```python
$ git clone "https://github.com/SMosegaard/cds-lang/tree/main/assignments/assignment-1"
```
2.  Navigate into the folder in your terminal:
```python
$ cd assignment-1
```
3. Execute the setup bash script to create a virtual environment and install necessary packages listed in the requirement.txt:
```python
$ source setup.sh
``` 
4.  Run the run bash script in the terminal to execude the code:
```python
 $ source run.sh
``` 

After the script completes running, it will display a message in the terminal output indicating that the results have been saved.

## Summary of results

The code presented is designed to extract linguistic information from a corpus of texts using spaCy. It calculates the relative frequency of nouns, verbs, adjectives, and adverbs per 10,000 words, as well as the total number of unique persons (PER), locations (LOC), and organizations (ORG) mentioned in each text. 

Utilizing the Uppsala Student English Corpus, the code successfully generates tables summarizing the linguistic features for each text. To illustrate the results obtained from four essays within the corpus, see the table below:

|Filename|RelFreq NOUN|RelFreq VERB|RelFreq ADJ|RelFreq ADV|Unique PER|Unique LOC|Unique ORG|
|---|---|---|---|---|---|---|---|
|1112.a1.txt|1237.96|1465.61|631.34|823|0|0|0|
|1071.a1.txt|1396.71|1302.82|622.07|481.22|3|0|4|
|0191.a1.txt|1251.49|1358.76|750.89|679.38|0|0|2|
|3045.a1.txt|1144.58|1385.54|622.49|401.61|0|1|0|
|etc|---|---|---|---|---|---|---|

In order to derive insights from the results, the various features are illustrated:

<p align = "center">
    <img src = "https://raw.githubusercontent.com/SMosegaard/cds-lang/main/assignments/assignment-1/out/wordclass.png" width = "400">
    <img src = "https://raw.githubusercontent.com/SMosegaard/cds-lang/main/assignments/assignment-1/out/entity.png" width = "400">
</p>


By visually inspecting the first plot, it is notable that nouns are the most frequently used POS followed by verbs. Contrary, adjectives are the significantly least used word class. No significant variations nor patterns emerge when comparing different subfolders nor terms.

However, examining the average number of unique named entities unveils a different pattern. Overall, the average count of unique entities remains quite low, except for four subfolders (b3, b4, b5, and c1). These folders especially exhibit a notably higher number of unique people. This can be attributed to the presence of longer essays and sometimes references (mainly in b3), which naturally leads to an increased mention of unique individuals.

## Discussion

The project utilized the 'en_core_web_md' spaCy model, which is trained on web data including news, comments, and blogs. Given the semantic and stylistic differences between web text and the dataset utlized in this repository, it is important to acknowledge that the model may struggle to capture certain nuances unique to the dataset, such as domain-specific terminology.

Another limitation worth considering lies in the variation within the USE corpus itself. The essays encompassed by the corpus vary significantly in terms of themes and length, as seen in the distribution of unique entities. This can make it difficult to establish patterns or generalise about the corpus as a whole.  

While the distribution of POS tags did not reveal any apparent correlation with terms and the differences in counts of unique entities might be due to the 'skewed' data, the tool employed still proves to effectively extract linguistic features. The tool has successfully uncovered patterns and linguistic insights within the USE corpus and can be applied to other text datasets.

*CodeCarbon was implemented to monitor carbon emissions associated with code execution. The results will be saved and discussed in assignment 5.*