# Assignment 3 - Query Expansion with Word Embeddings
*By Sofie Mosegaard, 21-03-2023*

This repository is designed to conduct ```query expansion``` with word embeddings via ```Gensim```'s word embedding model ('glove-wiki-gigaword-50').

The project has the objective:
1.  Pre-process texts in sensible ways
2.  Use pretrained word embeddings for query expansion
3.  Create a resusable command line tool for calculating results based on user inputs.

## Data source

In this repository, the query expansion with word embeddings will be conducted on the dataset 'Spotify Million Song Dataset'. The dataset comprises a corpus of lyrics from 57,650 English-language songs from Spotify including the the title and artist of the song.

You can download the dataset [here](https://www.kaggle.com/datasets/joebeachcapital/57651-spotify-songs) and place it in the ```in``` folder. Before executing the code, make sure to unzip the data.

## Repository structure

The repository includes:

- 2 bash scripts for setting up the virtual environment, installing requirements, and execution of the code
- 1 .txt file listing the required packages with versioned dependencies
- 1 README.md file
- 3 folders
    - in: holds the data to be processed
    - src: contains the Python code for execution
    - out: stores the saved results, including a dataframe in .csv format. Each row in the dataframe will show the percentage of a given artist's songs featuring the given input word from the expanded query. 

## Reproducibility

1.  Clone the repository:
```python
$ git clone "https://github.com/SMosegaard/cds-lang/tree/main/assignments/assignment-3"
```
2.  Navigate into the folder in your terminal:
```python
$ cd assignment-3
```
3. Execute the setup bash script to create a virtual environment and install necessary packages listed in the requirement.txt:
```python
$ source setup.sh
``` 

## Usage

Execute the run bash script in the terminal, specifying the target work (--word / -w) and artist name (--artist / -a):
```python
$ source run.sh -w {'target_word'} -a {'artist_name'}
``` 
The inputs will be converted to lowercase, so it is irrelevant whether it is spelled with or without capital letters.

Based on the input word, a pre-trained word embedding model will find 10 closely related words. Afterwards, the percentage of songs by the given artist that features the words from the expanded query will be calculated. 

After the script completes running, it will display the results in the terminal output together with a messeage indicating that the results have been saved to the ```out``` folder.

## Summary of results

The code facilitates query expansion with word embeddings. It integrates Gensim's word embedding model and allows the users to identify related words to a given target word. Then, it calculates the percentage of songs by a given artist that feature words from the expanded query. When utilizing lyrics of 57,650 English-language songs from Spotify, it returns the following results:

|Target word|Artist|Query|Total songs by artist|Songs containing query words|Percentage|
|---|---|---|---|---|---|
|hello|adele|goodbye, hey, !, kiss, wow, daddy, mama, bitch, dear, cry|54|26|48.15|
|sky|radiohead|horizon, bright, light, blue, dark, cloud, skies, lights, clouds, rainbow|150|46|30.67|
|friday|katy perry|thursday, monday, wednesday, tuesday, week, sunday, saturday, earlier, month, last|89|11|12.36|
|love|abba|dream, life, dreams, loves, me, my, mind, loving, wonder, soul|113|113|100|
|etc|---|---|---|---|---|

The findings present the percentage of artist's songs contain words related to a given target word. It can be seen, that the model returns the same word simply in a different conjugation, for example is 'skies' returned as a close word to 'sky' and 'loves' and 'loving' for the target word 'love'.

## Discussion

- the genism model uses word embeddings to find the most similar words
- word embeddings --> dense vector representations of words in a continuous vector space where words with similar meanings are closer to each other
- predictions --> 

- modellen er trænet på wikipedia
--> fakta baseret
--> hvis den var trænet på fx dagbog eller web, havde den måske retuneret ord som "friends, beer, party"
--> samme med baby retunerer ord med familie fremfor noget seksuelt eller kærligt

because it returns versions of the same word --> would have been relevant to extract the root of the returned word and like so limit the model, so it cant return identical words. 
If they are removed, there will also be room for real synonyms that will expand the query and might elicit more interesting results!


The example with Katy Perry and the target word 'friday' is quite interesting, as the model returns realistic and reasonable words. However, if one would to know the song 'Last Friday Night', it might have been more correct to return words like 'party', 'night', 'shots', etc. This demonstrates that the model purely provides a query of the similar words and does not take semantics behind the song, the word, nor the artist into account.

Overall, the repository presents a valuable tool for researchers and music enthusiasts, as it enables them to explore the thematic trends within song lyrics through the lens of word embeddings. Furthermore, the code could be applied to query expansion and semantic search in text datasets beyong music.

*CodeCarbon was implemented to monitor carbon emissions associated with code execution. The results will be saved and discussed in assignment 5.*