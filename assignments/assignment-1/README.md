# Portfolio 1: Extracting linguistic features using spaCy
*By Sofie Mosegaard, 22-02-2023*

This repository concerns using ```spaCy``` to extract linguistic information from a corpus of texts.

The project has the objective:
1.  Work with multiple input data arranged hierarchically in folders
2.  Use spaCy to extract linguistic information from text data
3.  Save those results in a clear way which can be shared or used for future analysis

## Data source

The dataset is a collection of essays from Uppsala University encompassing the period from 1999 to 2001. The corpus *The Uppsala Student English Corpus (USE)* can be found [here](https://ota.bodleian.ox.ac.uk/repository/xmlui/handle/20.500.12024/2457). 

Please download the data and place it in the folder called ```in```. Note, that you will need to unzip the data inside the folder before, you can execude the code as described below (see: Reproducibility). 

## Repository structure
The repository consists of 2 bash scripts, 1 README.md file, 1 txt file, and 3 folders. The folders contains the following:

-   ```in```: where you should locate your data
-   ```src```: consists of the scipt that will perform the repository objective
-   ```out```: holds the saved results in a .csv format for each sub-folder of data.

## Reproducibility

1.  Clone the repository
```python
$ git clone "https://github.com/SMosegaard/cds-lang/tree/main/assignments/assignment-1"
```
2.  Navigate into the folder in your terminal
3.  Run the setup bash script to create a virtual envoriment and install required packages specified in the requirement.txt:
```python
$ source setup.sh
``` 
4.  Run the run bash script in the terminal to execude the code:
```python
 $ source run.sh
``` 

## Discussion

The code presented is designed to extract linguistic information from a corpus of texts using spaCy. It calculates the relative frequency of nouns, verbs, adjectives, and adverbs per 10,000 words, as well as the total number of unique persons (PER), locations (LOC), and organizations (ORG) mentioned in each text. 

Utilizing the Uppsala Student English Corpus, the code successfully generates tables summarizing the linguistic features for each text. To illustrate the results obtained from four essays within the corpus, see the table below:

|Filename|RelFreq NOUN|RelFreq VERB|RelFreq ADJ|RelFreq ADV|Unique PER|Unique LOC|Unique ORG|
|---|---|---|---|---|---|---|---|
|1112.a1.txt|1237.96|1465.61|631.34|823|0|0|0|
|1071.a1.txt|1396.71|1302.82|622.07|481.22|3|0|4|
|0191.a1.txt|1251.49|1358.76|750.89|679.38|0|0|2|
|3045.a1.txt|1144.58|1385.54|622.49|401.61|0|1|0|
|etc|---|---|---|---|---|---|---|

The implementation involves looping over each text file, tokenizing, and analyzing the text with spaCy, and then computing the required linguistic metrics. The tool can be applied to other text datasets and offers researcher a tool to extract linguistic features and entity distribution within a corpus.