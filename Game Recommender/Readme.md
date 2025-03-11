# Steam Game Recommender


## Overview

The project includes Python scripts for various tasks such as importing data, preprocessing, partitioning data, computing similarities, training models, and making predictions. It consists of two main parts:

1. **Game Prediction:**
   - Predict whether a user has played a particular game.
   - Build models, including logistic regression, for game prediction.
   - Evaluate model performance and test on provided data.

2. **Hours Played Prediction:**
   - Predict the number of hours a user has played a game.
   - Perform preprocessing steps and define functions for iteration.
   - Train the model and make predictions on test data.

## File Structure
    Data/
        train.json.gz
        pairs_Played.csv
        pairs_Hours.csv
    README.md
    assignment1.py
    predictions_Played.csv
    predictions_Hours.csv
## Usage

1. **Data Preparation:**
   - Place the provided data files (`train.json.gz`, `pairs_Played.csv`, `pairs_Hours.csv`) in the `Data/` directory.

2. **Running the Code:**
   - Execute the `assignment1.py` script to run the entire analysis pipeline.
   - Ensure all required libraries are installed (`gzip`, `scipy`, `sklearn`, `numpy`, etc.).

3. **Viewing Results:**
   - The predictions for game plays (`predictions_Played.csv`) and hours played (`predictions_Hours.csv`) will be generated.
   - Explore the results and evaluate model performance based on accuracy metrics.
