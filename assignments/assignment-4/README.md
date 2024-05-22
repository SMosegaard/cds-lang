# Assignment 4 - Emotion Analysis with Pretrained Language Models
*By Sofie Mosegaard, 18-04-2023*

This repository is dedicated to conducting ```computational text analysis``` using a ```pretrained sentiment analysis model```. The pretrained language model 'emotion-english-distilroberta-base' will be extracted from HuggingFace and applied to a corpus of English text to assess its emotional profile and whether it changes over time. 

The model is a pretrained DistilRoBERTa-base model and is finetuned on emotion data. The pretrained sentiment model will predict emotion scores randing from 0-1 for Ekman's 6 basic emotions (joy, surprise, sadness, fear, anger, and disgust) as well as a neutral class. 
You can read about the model [here](https://huggingface.co/j-hartmann/emotion-english-distilroberta-base).

The code will load the classifier and dataset. Secondly, the classifier will be applied to all sentences in the dataset, returning the emotion label with the greatest emotion score. Finally, it will visualise the relative frequency of all emotions.

## Data source

The projects objective will be performed on scripts from the TV show *Game of Thrones*. The scripts have been segmented into lines, comprising a total of 23,912 rows. Each rows includes additional metadata, indicating who said the line, what episode it came from, and season of the line.

## Repository structure

The repository consists of 2 bash scripts for setup and execution of the code, 1 .txt file with the required packages, and 3 folders. The folders contains the following:

-   ```in```: where the data is located.
-   ```src```: consists of the scipt that will perform the repository  objective.
-   ```out```: holds the saved results in a .png format. The results will show the distribution of all emotion labels for all seasons, as well as the relative frequency of each emotion across all seasons.

## Reproducibility and usage

1.  Clone the repository
```python
$ git clone "https://github.com/SMosegaard/cds-lang/tree/main/assignments/assignment-4"
```
2.  Navigate into the folder in your terminal
```python
$ cd assignment-4
```
3.  Run the setup bash script to create a virtual envoriment and install required packages specified in the requirement.txt:
```python
$ source setup.sh
```
4.  Run the run bash script in the terminal to execude the code:
```python
$ source run.sh
```

## Discussion

The code facilitates computational text analysis to extract meaningful information in terms of emotion scores. It integrates HuggingFace's model 'emotion-english-distilroberta-base' and allows the users to assess the series emotional profile and how it changes over the seasons.

Then, it plots the distribution of all emotions in all seasons and the relative frequency of each emotion across all seasons.

When utilizing the script of the 8 seasons of *Game of Thrones*, it returns the following results:

![emotion.png](https://github.com/SMosegaard/cds-lang/tree/main/assignments/assignment-4/out/emotion.png?raw=true)
![season.png](https://github.com/SMosegaard/cds-lang/tree/main/assignments/assignment-4/out/season.png?raw=true)

Anger is the most dominant emotion throughout the series, however it seems to fade gradually as the story progresses towards the end. This trend suggests a possible evolution in the emotional landscape, where the characters probably have handled their battles and conflicts.

Secondly, the "emotion" neutral also a very prominent emotion throughout the whole series. This makes great sense, as all sentences in an entire series can't have significant emotional charge.

Across the whole of the series, there is very little joy, sadness, and fear present. This suggests that the narrative of the series predominantly revolves around themes of conflict and battles rather than happiness, sorrow, or fear. Also, that the characters might not possess fear of the battles, they are fighting.

The emotions surprice and disgust is most presented during the earlier seasons, particularly in seasons 2-4. This could demonstrate plot twists, conflicts, and character revelations, that needs to be presented in the beginning of the series to motivate the the viewer to watch more.

However, it's crucial to note that emotional labels are predictions. As human language is nuanced and complex, sentences can convey multiple emotions simultaneously. Therefore, the emotion score with the highest value for a given sentence is selected as the emotional label. This approach might also explain the prominence of the emotion neutral, as emotions can be discrete and subtle, and anger, which is a very dominant emotion compared to for example fear, that could be expressed very differently.

The code provides a valuable tool for exploring emotions in texts. It allows for the application of sentiment analysis to various English text datasets, enabling users to delve into emotional nuances of literary work, social media conversations, speeches, lyrics of music, etc.


*CodeCarbon was implemented to monitor carbon emissions associated with code execution. The results will be saved and discussed in assignment 5.*