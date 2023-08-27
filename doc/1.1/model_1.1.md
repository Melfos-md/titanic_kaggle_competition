# Model 1.1

A simple logistic regression.

## What's new
- Include Cabin, replacing NaN by 'Missing' str
- Include Ticket number length and prefix (separate features)*
- Include title from Name feature
- Accuracy threshold stick to 0.5; since for that problem we don't need to penalize 0 or 1.
- Changed CategoryEncoding from multi-hot to one-hot

## Data preprocessing
(preprocessing module)
Since we are with a small CSV dataset, data is loaded in memory as a pandas Dataframe then passed to a Tensorflow model (preprocessing_pipeline)
Missing value handling and feature exclusion are done with pandas before passing data to tensorflow.

**Steps with pandas**
- Load data
- Exclude features
    - Name: try it later
    - Ticket: has 681 unique values on 891 entry, I am not sure
    - PassengerId: useless
- Handling missing value for Embarked feature
    - "Missing" value created
- Send data to Tensorflow

**Preprocess pipeline with Tensorflow**
![preprocessing_pipeline](preprocessing_pipeline.png)

- Missing value for numeric data are replaced by mean
- Numeric data are normalized by mean and standard deviation
- Categorical data are one-hot encoded

**Notes on EDA**
![Survival Rate by Ticket number length](ticket_length_survival.png)


## Model

Simple logistic regression



**Parameters**
- Total params: 26 (108.00 Byte)

- Trainable params: 17 (68.00 Byte)

- Non-trainable params: 9 (40.00 Byte)


**Hyperparameters**
- learning_rate: 0.001
- epoch: 100


**Results**

|          | Train  | Dev    | Test   |
|----------|--------|--------|--------|
| Loss     | 0.4031 | 0.4032 | 0.4459 |
| Accuracy | 0.8352 | 0.8090 | 0.7932 |


<table>
  <tr>
    <td> <img src="loss.png" alt="Image 1" style="width: 250px;"/> </td>
    <td> <img src="accuracy.png" alt="Image 2" style="width: 250px;"/> </td>
  </tr>
</table>


File: doc/model_1.0/model_1.0.tf

Submission 2023-27-08: 0.79186 rank 853