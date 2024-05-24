# Assignment 3 - Query Expansion with Word Embeddings
*By Sofie Mosegaard, 21-03-2023*

This repository demonstrates the use of word embeddings for query expansion and analysis of song lyrics. The assignment is designed to explore the relationship between a target word and an artist's song lyrics by leveraging ```Gensim```'s word embedding model to expand the query and calculate the percentage of songs containing related words. 

An overview of the process:

- The user specifies a target word and an artist
- Loads the dataset containing containing song lyrics and artist names. It then removes punctuation from the lyrics to prepare the data for analysis.
- Loads the 'glove-wiki-gigaword-50' word embedding model from gensim. This pre-trained model will capture semantic relations between words and find similar words to the given target word.
- Expands the query by finding similar words to the target word using the word embedding model. Using the Natural Language Toolkit (nltk) library, the words will be filtered to ensure unique words in lemmatized form.
- Calculates the percentage of the artist's songs that contain words related to the expanded query.
- Displays the percentage in the terminal output and saves the results as a dataframe in .csv format.
- Visualises all target words and queries from the saved dataframe using t-SNE dimensionality reduction to illustarte the relationship between the words.

By utilizing word embeddings, the script offers an approach to explore song lyrics. To better understand the code, all functions in the script ```src/query_expansion.py`` will include a brief descriptive text.

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
    - out: stores the saved results, including a t-SNE plot of the findings and a dataframe in .csv format. Each row in the dataframe will show the percentage of a given artist's songs featuring the given input word from the expanded query. 

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
|sky|radiohead|horizon, bright, light, blue, dark, cloud, skies, rainbow, ocean, eyes|150|53|35.33|
|friday|katy perry|thursday, monday, wednesday, tuesday, week, sunday, saturday, earlier, month, last|89|11|12.36|
|love|abba|dream, life, me, my, mind, loving, wonder, soul, crazy, happy|113|113|100|
|baby|justin bieber|babies, boy, girl, newborn, pregnant, mom, child, toddler, mother, cat|131|82|62.6|

The findings present the percentage of artist's songs contain words related to a given target word. Despite the implementation of the nltk lemmatization technique, it can be seen that the model sometimes returns the same word in different conjugation, for example is "skies"  as a close word to 'sky' and 'loving' for the target word 'love'.

<p align = "center">
    <img src = "https://raw.githubusercontent.com/SMosegaard/cds-lang/main/assignments/assignment-3/out/t-SNE_nQueries_from_df.png" width = "600">
</p>

The t-SNE plot visualizes the relationships between target words and their associated queries. It illustartes the semantic similarities among words and how certain clusters overlap while others remain distinct. For instance, the clusters "hello" and "bitch" reveal a overlap, which indicates a shared semantic context, while the clusters "thriller" and "friday" appear distinctly separate, suggesting unique semantic associations for these target words.

## Discussion

The choice of the pre-trained word embedding model plays a crucial role in the results obtained from query expansion. In this case, the Gensim 'glove-wiki-gigaword-50' model is trained on Wikipedia data, which will exhibit some biases. As the training corpus Wikipedia primarily contains factual information, it will influence the semantic relationships captured by the model.

For example, when analysing the songs of Katy Perry with the target word 'friday' and Justin Bieber with word 'baby', the model returns other weekdays and words related to children and pregnancy. These results are very reasonable given Wikipedia's focus on factual information and reflects the general context in which the words are used on Wikipedia. Thus it is important to acknowledge that the model does not capture all semantic nuances, especially not the creative language commonly found in song lyrics. If the model were trained on web data or personal writings, the results could be significantly different. Instead, the model might associate 'friday' with words like "party, friends, beer", and 'baby' with more romantic or sexual connotations for Justin Bieber's songs.

This limitation shows the importance of the domain of the employed model. The relevance and accuracy of the query expansion results could potentially be improved by fine-tuning the model on domain-specific data such as song lyrics. 

Another limitation observed is the model's tendency to return different conjugations or variations of the same word. While these words are semantically related, they do not contribute to expanding the query in a meaningful way. In addition to the implemented lemmatization technique, it would have been relevant to extract the root form of the returned word and remove identical or very similar words. By doing so, there will be room for real synonyms and related words, that can enhance the query expansion and elicit more interesting results!

Overall, the repository presents a valuable tool for researchers and music enthusiasts, as it enables them to explore the thematic trends within song lyrics through the lens of word embeddings. Furthermore, the code could be applied to other text datasets beyond music. 

*CodeCarbon was implemented to monitor carbon emissions associated with code execution. The results will be saved and discussed in assignment 5.*