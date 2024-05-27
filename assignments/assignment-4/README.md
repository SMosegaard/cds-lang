# Assignment 4 - Emotion Analysis with Pretrained Language Models
*By Sofie Mosegaard, 18-04-2023*

This repository is dedicated to conducting ```computational text analysis``` using a ```pretrained sentiment analysis model```. The pretrained language model 'emotion-english-distilroberta-base' will be extracted from HuggingFace and applied to a corpus of English text to assess its emotional profile and whether it changes over time. 

The model is a pretrained DistilRoBERTa-base model and is finetuned on emotion data. The pretrained sentiment model will predict emotion scores ranging from 0-1 for Ekman's 6 basic emotions (anger, disgust, fear, joy, sadness, and surprise) as well as a neutral class. You can read about the model [here](https://huggingface.co/j-hartmann/emotion-english-distilroberta-base).

Specifically, the project will conduct the emotion analysis on the TV series "Games of Thrones" by doing the following steps:
- Loads the pre-trained text classification model from HuggingFace
- Applies the classifier to all sentences in the dataset and returns the emotion label with the greatest emotion score
- Saves the dataframe with predicted emotions as .csv in the ```out``` folder
- Visualises the relative frequency of all emotions across seasons and saves the plots in the ```out``` folder

To better understand the code, all functions in the script ```src/emotion_analysis.py`` will include a brief descriptive text.

## Data

The projects’ objective will be performed on scripts from the TV show *Game of Thrones*. The scripts have been segmented into lines, comprising a total of 23,912 rows. Each rows includes additional metadata, indicating who said the line, what episode it came from, and season of the line.

You can download the dataset [here](https://www.kaggle.com/datasets/albenft/game-of-thrones-script-all-seasons?select=Game_of_Thrones_Script.csv) and place it in the ```in``` folder. Remember to unzip the data before proceeding with the analysis.

## Project Structure

The repository includes:

- 2 bash scripts for setting up the virtual environment, installing requirements, and execution of the code
- 1 .txt file listing the required packages with versioned dependencies
- 1 README.md file
- 3 folders
    - in: holds the raw data to be processed
    - src: contains the Python code for performing the repository objective
    - out: stores the saved results in a .png format. The results will show the distribution of all emotion labels for all seasons, as well as the relative frequency of each emotion across all seasons.

## Getting Started

To ensure reproducibility and facilitate collaboration, follow these steps:

1.  Clone the repository using the following command:
```python
$ git clone "https://github.com/SMosegaard/cds-lang/tree/main/assignments/assignment-4"
```
2. Navigate into the 'assignment-4' folder in your terminal:
```python
$ cd assignment-4
```
3. Execute the 'setup.sh' script to create a virtual environment and install the necessary packages listed in 'requirement.txt':
```python
$ source setup.sh
```
4. Run the 'run.sh' script in the terminal to execute the code and perform the analysis:
```python
$ source run.sh
```

Upon completion, a message will be displayed in the terminal output, confirming that the results have been successfully saved to the ```out``` folder.

## Results

The code facilitates computational text analysis to extract meaningful information in terms of emotion scores. It integrates HuggingFace's model 'emotion-english-distilroberta-base' and allows the users to assess the series emotional profile and how it changes over the seasons. Then, it plots the distribution of all emotions in all seasons and the relative frequency of each emotion across all seasons.

When utilizing the script of the 8 seasons of *Game of Thrones*, it returns the following results:

<p align = "center">
    <img src = "https://raw.githubusercontent.com/SMosegaard/cds-lang/main/assignments/assignment-4/out/emotion.png" width = "800">
</p>

<p align = "center">
    <img src = "https://raw.githubusercontent.com/SMosegaard/cds-lang/main/assignments/assignment-4/out/season.png" width = "800">
</p>

The most predominant emotion observed throughout the series is neutrality, with over 40% of all sentences categorized as neutral. This aligns with expectations, as a narrative spanning an entire series encompasses a wide array of scenarios, not all of which can evoke strong emotions. 

Anger emerges as another significant emotion across all seasons. This emotion represents the nature of the series, as many battles and conflicts are fought throughout the all seasons of the series.

Across the whole of the series, the relative frequency of joy, sadness, and fear are very similar, each consistently below 10%. It could suggest that the narrative of the series predominantly revolves around conflict rather than other emotional arcs. Also, that the characters might not possess fear of the battles, they are fighting.

Surprise and disgust are most prevalent in the earlier seasons, particularly in seasons 2-4, with disgust showing a notable decline as the series advances. These findings could demonstrate plot twists, conflicts, and character revelations, that needs to be presented in the beginning of the series to motivate the viewer to watch more.

## Discussion

First of all, it's important to acknowledge the predictive nature of the emotional labels. As human language is nuanced and complex, sentences can convey multiple emotions simultaneously. Therefore, the emotion score with the highest value for a given sentence is selected as the emotional label. This approach might also explain the prominence of the emotion neutral, as emotions can be discrete and subtle, and anger, which is a very dominant emotion compared to for example fear, that could be expressed very differently.

Removing the neutral class or plotting the other emotions independently could offer further insights into the emotional landscape of the series. This could shedd light on nuances that might be obscured by the dominant presence of the emotion neutral.

Another limitation could be the varying lenght's of the seasons, as all emotion labels seem to fade gradually as the story progresses towards the end. The first five seasons are over twice as long as season 8, with which they may exhibit greater diversity in emotional expression thus potentially impacts the distribution of emotions observed. Plotting the results against season length could provide clarity on whether the observed patterns simply is a reflection of these variations. 

The code provides a valuable tool for exploring emotions in texts. It allows for the application of sentiment analysis to various English text datasets, enabling users to delve into emotional nuances of various textual contexts.

*CodeCarbon was implemented to monitor carbon emissions associated with code execution. The results will be saved and discussed in assignment 5.*