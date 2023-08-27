# Model 1.0

A simple logistic regression to use as baseline model.

## Data preprocessing
(preprocessing module)
Since we are with a small CSV dataset, data is loaded in memory as a pandas Dataframe then passed to a Tensorflow model (preprocessing_pipeline)
Missing value handling and feature exclusion are done with pandas before passing data to tensorflow.

**Steps with pandas**
- Load data
- Exclude features
    - Name: try it later
    - Ticket: explore later
    - Cabin: explore later
    - PassengerId: useless
- Handling missing value for Embarked feature
    - "Missing" value created
- Send data to Tensorflow

**Preprocess pipeline with Tensorflow**
![preprocessing_pipeline](preprocessing_pipeline.png)

- Missing value for numeric data are replaced by mean
- Numeric data are normalized by mean and standard deviation
- Categorical data are one-hot encoded


## Model

Simple logistic regression

![model_1.0](simple_sigmoid_model.png)

**Parameters**
- Total params: 26 (108.00 Byte)

- Trainable params: 17 (68.00 Byte)

- Non-trainable params: 9 (40.00 Byte)


**Hyperparameters**
- learning_rate: 0.01
- epoch: 50


**Results**

|          | Train  | Dev    | Test   |
|----------|--------|--------|--------|
| Loss     | 0.4676 | 0.4277 | 0.4676 |
| Accuracy | 0.7921 | 0.8202 | 0.7765 |


<table>
  <tr>
    <td> <img src="loss.png" alt="Image 1" style="width: 250px;"/> </td>
    <td> <img src="metrics.png" alt="Image 2" style="width: 250px;"/> </td>
  </tr>
</table>


File: doc/model_1.0/model_1.0.tf

Submission 2023-27-08: 0.75837 rank 11624