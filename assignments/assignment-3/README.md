# Portfolio 3 - Query expansion with word embeddings
*By Sofie Mosegaard, 21-03-2023*

This assignment is designed to conduct ```query expansion``` with word embeddings via ```gensim```.

The assignment has the objective:
-   Pre-process texts in sensible ways
-   Use pretrained word embeddings for query expansion
-   Create a resusable command line tool for calculating results based on user inputs.


### Repository structure
The repository consists of 2 bash scripts, 1 README.md file, and 3 folders. The folders contains the following:
-   ```in```: contains on the corpus of lyrics from 57,650 English-language songs. You can read more about the data here https://www.kaggle.com/datasets/joebeachcapital/57651-spotify-songs.
-   ```src```: consists of the scipt that will perform the assignments objective.
-   ```out```: the percentage of a given artist's songs featuring the given input word from the expanded query will be saved as .txt in this folder.


### Prerequisite work
-   Clone this repository: $ git clone "https://github.com/SMosegaard/cds-lang/tree/main/assignments/assignment-3"
-   Navigate to the 'assignment-3' folder in your terminal
-   Run the setup bash script to create virtual envoriment and install required packages: $ source setup.sh


### Usage
Run the run bash script in the terminal: $ source run.sh

When you run the command above, you will be asked to enter a target word and a artist name. Based on the input word, a pre-trained word embedding model will find 10 closely related words. Afterwards, the percentage of songs by the given artist that features the words from the expanded query will be calculated. 
