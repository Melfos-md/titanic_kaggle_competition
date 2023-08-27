# Titanic kaggle competition with tensorflow in python

Each tag is a model.

## Project organization
- "doc" folder regroup documentation for each model and an image of the tensorflow model. 
  - each folder in doc has model name, associated with markdown file for documentation
  - sub-folder "model_version.tf" is tensorflow model backup
- main.py: file where the model is tested.
- submission.py: kaggle test set to submit


## Models
- v1.0: simple logistic regression with few preprocessing.
  - Kaggle submission: 0.75837
- v1.1: some feature engineering
  - Kaggle submission: 0.79186