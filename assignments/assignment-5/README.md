# Assignment 5 - Evaluating Environmental Impact of the Current Exam Portfolio
*By Sofie Mosegaard, 03-05-2024*

As the realms of machine learning, automatic analytics, and AI continue to surge in popularity, it is crucial to pay attention to the often-overlooked environmental implications. This repository is designed to explore the envorimental footprint of the current exam portfolio.

In our enthusiasm for technological progress, we often forget about the environmental effects of our digital activities. However, it is essential to carefully consider how our technological advancements affect the planet.

To address this important issue, the open-source software ```CodeCarbon``` has been implemented in each of the four assignments in this portfolio exam. By utilizing the EmissionsTracker class from the CodeCarbon library, the goal is to measure the approximate CO₂eq emissions produced by running the assignments, both the whole scripts and their subtasks. This is done to see the total emission as well as to investigate the individual tasks impact. Feel free to explore the scripts to understand how CodeCarbon is integrated.

This repository serves as a platform for discussing the insights derived from implementing CodeCarbon. The results will be summarized, visualized, and discussed with an aim to deepen our understanding of how technology affects the envorirment.

## Project Structure

The repository includes:

- 1 bash scripts for installing requirements
- 1 .txt file listing the required packages with versioned dependencies
- 1 README.md file
- 3 folders
    - out: holds the .csv files obtained from the implementation of CodeCarbon in the previous four assignments 
    - src: contains a Python notebook for data wrangling and visualisations
    - plots: stores the saved plots in a .png format

## Getting Started

To ensure reproducibility and facilitate collaboration, follow these steps:

1.  Clone the repository using the following command:
```python
$ git clone "https://github.com/SMosegaard/cds-lang/tree/main/assignments/assignment-5"
```
2. Navigate into the 'assignment-4' folder in your terminal:
```python
$ cd assignment-5
```
3. Execute the 'setup.sh' script to create a virtual environment and install the necessary packages listed in 'requirement.txt':
```python
$ source setup.sh
```

Now, you can execute the ```src/inspect_environmental_impact.ipynb```. You will find additional information on how to proceed inside the notebook. The visualizations generated in the notebook will be saved to the ```plots``` folder.

## Results

To evaluate the environmental impact of the portfolio exam, it is first identified which assignments generated the highest emissions in terms of CO₂eq:

|Assignment|CO₂eq Emission|
|---|---|
|Assignment 1|0.00242|
|Assignment 2|0.00999|
|Assignment 3|0.00019|
|Assignment 4|0.02275|

<p align = "center">
    <img src = "https://raw.githubusercontent.com/SMosegaard/cds-lang/main/assignments/assignment-5/plots/Emissions_per_Assignment.png" width = "600">
</p>

It's evident that assignment 4 emits significantly more CO₂eq compared to the others. Following is assignment 2, with less than half the emissions, assignments 1, and assignment 3.

To assess which specific subtaks within each assignment contributed the most to emissions in terms of CO₂eq, the emissions of individual subtasks across assignments are visualized:

<p align = "center">
    <img src = "https://raw.githubusercontent.com/SMosegaard/cds-lang/main/assignments/assignment-5/plots/Subtask_Emissions_across_Assignments.png" width = "500">
</p>

It is clear that certain tasks within each assignment contributed significantly to the CO₂ emission. In assignment 1, the subtask 'data processing', which involves transforming raw data into insights by extracting linguistic features, emerged as the primary contributor to emissions. For assignment 2, both 'hyperparameter tuning' and 'permutation testing' played central roles. Hyperparameter, especially conducted using GridSearch, and permutation testing proved to be as computationally heavy as expected.

<p align = "center">
    <img src = "https://raw.githubusercontent.com/SMosegaard/cds-lang/main/assignments/assignment-5/plots/Emissions_per_pipeline_a2.png" width = "500">
</p>

In assignment 3, it was notably the loading of the 'glove-wiki-gigaword-50' word embedding model that predominantly contributed to emissions. Finally, for assignment 4 'model prediction' emits the most, here a pre-trained text classification model from HuggingFace predicts emotions for all sentences in a huge dataframe. 

The results are as expected, as the assignments with the highest emission were also the ones that took the longest to run.

## Discussion

The analysis revealed that assignment 4 produced the highest emissions, approximately 0.0228 kg of CO₂eq. This outcome can be attributed to the demanding task involved, which was to extracti emotions from the TV show Game of Thrones. The assignment utilized a pretrained large language model on a massive dataset, which led to substantial CO₂ emission.

Assignment 2 had the second highest emission. Especially, the heavy computational burden of GridSearch led to significant emissions in this task. In the hypoerparamter tuning using GridSearch, all combinations of predefined parameters are tested through k-fold cross validation with 5 folds. This also explains why permutation testing emits a significant amount, as it conducts 100 permutations through 5-fold cross validation. Both subtasks takes a long time to compute and are computationally heavy. This demonstrates a correlation between how long it takes to compute a given task and its task's emission.

While GridSearch offers the potential for optimizing model performance, its computational intensity may not always justify the emissions generated. Therefore, users had the option to reduce emissions by deselecting GridSearch and PermutationTesting when running the code.

Assignment 3 was the lest expencive assignment, where loading the model had the largest impact. Unlike the other scripts, it did not iterate through all the data but rather identified the most similar words to a given target word and searched for them in the lyrics from one specified artist. This reduced the  computational load and resulted in comparatively lower emissions.

While the emission values provided by CodeCarbon offer valuable insights, it's essential to acknowledge that they are estimates rather than exact measurements of emissions. Therefore, one must be causious when interpreting the results. However, they do provide a great overview of the emissions generated by each assignment. 

Considering that the portfolio is designed for learning purposes, the generated emissions might not be justified solely by the educational value of the assignments. This raises a question about the balance between learning objectives and environmental impact. Future improvements could therefore involve implementing alternative methods or optimizations that could minimize emissions without compromising learning outcomes.