# Module that exclude features, split data, normalize, missing value handling.
# Model graph image is generated with the following
# 
# ```
# from preprocessing import preprocess_model
# 
# titanic_preprocessing = preprocess_model()
# tf.keras.utils.plot_model(model=titanic_preprocessing, to_file='preprocessing_pipeline.png', rankdir="LR", dpi=72, show_shapes=True)
# ```
import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers
from typing import Tuple, List, Dict

# Load data without excluded features
# Split features and labels
# Handle missing values
# Argument:
# - path: string to csv
# - live: if True path must be csv for kaggle submission, else pass data to train
# - nrows: number of rows to read in csv, for unit testing
# all columns: ['PassengerId', 'Survived', 'Pclass', 'Name', 'Sex', 'Age', 'SibSp', 'Parch', 'Ticket', 'Fare', 'Cabin', 'Embarked']
def load_and_prepare_data(path: str, live:bool=False, nrows:int=None) \
        -> Tuple[pd.DataFrame, pd.DataFrame]:
    usecols = [
                'PassengerId', 'Survived', 'Pclass', 'Sex', 'Age', 
                'SibSp', 'Parch', 'Fare', 'Embarked', 'Ticket', 'Cabin',
                'Name'
            ]
    if live:
        usecols.remove('Survived')
    else:
        usecols.remove('PassengerId')
    titanic_features = pd.read_csv(path, usecols=usecols, nrows=nrows)
    titanic_features['Embarked'].fillna("Missing", inplace=True)  
    titanic_features['Pclass'] = titanic_features['Pclass'].astype(str)
    titanic_features['Cabin'] = titanic_features['Cabin'].fillna('Missing')
    titanic_features['Ticket_prefix'] = extract_ticket_prefix(titanic_features['Ticket'])
    titanic_features['Ticket_number_length'] = get_ticket_number_length(titanic_features['Ticket']).astype(str)
    titanic_features['title'] = extract_title(titanic_features['Name'])
    titanic_features.drop(['Ticket', 'Name'], inplace=True, axis=1)
    if live:
        titanic_id = titanic_features.pop('PassengerId')
        return titanic_features, titanic_id
    else:
        titanic_labels = titanic_features.pop('Survived')
        return titanic_features, titanic_labels

def extract_ticket_prefix(ticket: pd.Series):
    return ticket.str.extract(r'([a-zA-Z]+)', expand=False)\
                .fillna('Missing')

def extract_ticket_number(ticket: pd.Series):
    return ticket.str.extract(r'(\d+)$', expand=False).fillna('Missing')

def get_ticket_number_length(ticket: pd.Series):
    ticket_numbers = extract_ticket_number(ticket)
    return ticket_numbers.apply(lambda x: 0 if x == 'Missing' else len(x))

def extract_title(name: pd.Series) -> pd.Series:
    # extract title from 'Name' feature
    return name.str.extract(r' ([A-Za-z]+)\.', expand=False)

# Instantiate keras tensors for each column.
# Return dictionnary {name: KeraTensor} where
# - name is column name 
# - KeraTensor is an instantiated keras Tensor
def instantiate_tensors(titanic_features: pd.DataFrame) -> Dict[str, tf.Tensor]:
    inputs = {}

    for name, dtype in titanic_features.items():
        if name in ['Age', 'Parch', 'Fare', 'SibSp']: # columns that I want to be numeric
            dtype = tf.float32
        else:
            dtype = tf.string
        
        inputs[name] = tf.keras.Input(shape=(1,), name=name, dtype=dtype)
    return inputs

# Symbolic preprocessing: numeric features
# Arguments:
# - titanic_features: pd.DataFrame of features
# - inputs: Dict[str, tf.Tensor] from instantiate_tensors() function
# Return processed inputs graph in a list
def preprocess_numeric_features(titanic_features: pd.DataFrame, inputs: Dict[str, tf.Tensor]) -> List[tf.Tensor]:
    # symbolic preprocessing numeric features
    numeric_inputs = {name:input for name,input in inputs.items() if input.dtype==tf.float32}
    x = layers.Concatenate()(list(numeric_inputs.values()))
    norm = layers.Normalization()
    norm.adapt(np.array(titanic_features[numeric_inputs.keys()]))
    normalized_inputs = norm(x)

    # Replace NaN by 0
    all_numeric_inputs = layers.Lambda(lambda x: tf.where(tf.math.is_nan(x), 0.0, x))(normalized_inputs)
    return [all_numeric_inputs]


# symbolic preprocess categorical features
# Arguments:
# - titanic_features: pd.DataFrame of features
# - inputs: Dict[str, tf.Tensor] from instantiate_tensors() function
# Return: processed inputs graph in a list
def preprocess_categorical_features(titanic_features: pd.DataFrame, inputs: Dict[str, tf.Tensor]) -> List[tf.Tensor]:
    preprocessed_inputs = []
    for name, input in inputs.items():
        if input.dtype != tf.float32:
            unique_values = np.unique(titanic_features[name].astype(str).fillna("Missing"))
            lookup = layers.StringLookup(vocabulary=unique_values)
            one_hot = layers.CategoryEncoding(num_tokens=lookup.vocabulary_size(), output_mode='one_hot')
            x = lookup(input)
            x = one_hot(x)
            preprocessed_inputs.append(x)
    return preprocessed_inputs

"""
 Entry point of the module
 Return inputs, preprocess_pipeline, titanic_features, titanic_labels
 - inputs: dict of symbolic tf.keras.Input objects matching the names and data-types of the columns
 - preprocess_pipeline: keras model, see dot format in README.md
 - titanic_features
 - titanic_labels: labels if not live, PassengerId if live
 To generate dot format of preprocess_pipeline model:
 ```
 from preprocessing import preprocess_model
 import tensorflow as tf
 
 _,titanic_preprocessing, _, _ = preprocess_model()
 tf.keras.utils.plot_model(model=titanic_preprocessing, to_file='preprocessing_pipeline.png', rankdir="LR", dpi=300, show_shapes=True)
 ```
"""
def preprocess_model(path: str, live:bool):
    titanic_features, titanic_labels = load_and_prepare_data(path=path, live=live)
    inputs = instantiate_tensors(titanic_features)

    # concatenate tensor graph in a list
    preprocessed_inputs = preprocess_numeric_features(titanic_features, inputs) + preprocess_categorical_features(titanic_features, inputs)

    preprocessed_inputs_cat = layers.Concatenate()(preprocessed_inputs)
    preprocess_pipeline = tf.keras.Model(inputs, preprocessed_inputs_cat, name='preprocessing_pipeline')
    return inputs, preprocess_pipeline, titanic_features, titanic_labels