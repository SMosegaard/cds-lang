# Portfolio 3 - Query expansion with word embeddings
*By Sofie Mosegaard, 21-03-2023*

This repository is designed to conduct ```query expansion``` with word embeddings via ```Gensim```'s word embedding model ('glove-wiki-gigaword-50').

The project has the objective:
1.  Pre-process texts in sensible ways
2.  Use pretrained word embeddings for query expansion
3.  Create a resusable command line tool for calculating results based on user inputs.

## Data source

The dataset comprises a corpus of lyrics from 57,650 English-language songs from Spotify. You can read more about the data [on Kaggle](https://www.kaggle.com/datasets/joebeachcapital/57651-spotify-songs).

Please download the data and place it in the folder called ```in```. Note, that you will need to unzip the data inside the folder before, you can execude the code as described below (see: Reproducibility).

## Repository structure

The repository consists of 2 bash scripts for setup and execution of the code, 1 .txt file with the required packages, and 3 folders. The folders contains the following:

-   ```in```: where you should locate your data.
-   ```src```: consists of the scipt that will perform the repository  objective.
-   ```out```: holds the saved results in a .txt format. The results will show the percentage of a given artist's songs featuring the given input word from the expanded query.

## Reproducibility

1.  Clone the repository
```python
$ git clone "https://github.com/SMosegaard/cds-lang/tree/main/assignments/assignment-3"
```
2.  Navigate into the folder in your terminal
3.  Run the setup bash script to create a virtual envoriment and install required packages specified in the requirement.txt:
```python
$ source setup.sh
```

## Usage

Run the run bash script in the terminal with the required input information (--word / -w and --artist / -a):
```python
$ source run.sh --word {target_word} --artist {artist_name}
```
As the inputs and the corpus will be made in lower case, it makes no difference how the target word or the artist is spelled.
Based on the input word, a pre-trained word embedding model will find 10 closely related words. Afterwards, the percentage of songs by the given artist that features the words from the expanded query will be calculated. 

## Discussion

The code facilitates query expansion with word embeddings. It integrates Gensim's word embedding model and allows the users to identify related words to a given target word. The, it calculates the percentage of songs by a given artist that feature words from the expanded query.

When utilizing lyrics of 57,650 English-language songs from Spotify, it returns the following results:

```python
62.6% of justin bieber's songs contain words related to baby 
30.67% of radiohead's songs contain words related to sky
100.0% of abba's songs contain words related to love

```
Overall, the repository presents a valuable tool for researchers and music enthusiasts, as it enables them to explore the thematic trends within song lyrics through the lens of word embeddings. Furthermore, the code could be applied to query expansion and semantic search in text datasets beyong music.
