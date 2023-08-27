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
# - col: dict, {column_name:[]}
# all columns: ['PassengerId', 'Survived', 'Pclass', 'Name', 'Sex', 'Age', 'SibSp', 'Parch', 'Ticket', 'Fare', 'Cabin', 'Embarked']
def load_and_prepare_data():
    usecols = ['Survived', 'Pclass', 'Sex', 'Age', 'SibSp', 'Parch', 'Fare', 'Embarked']
    titanic_features = pd.read_csv('data/train.csv', usecols=usecols)
    titanic_features['Embarked'].fillna("Missing", inplace=True)  
    titanic_features['Pclass'] = titanic_features['Pclass'].astype(str)
    titanic_labels = titanic_features.pop('Survived')
    return titanic_features, titanic_labels

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
            one_hot = layers.CategoryEncoding(num_tokens=lookup.vocabulary_size())
            x = lookup(input)
            x = one_hot(x)
            preprocessed_inputs.append(x)
    return preprocessed_inputs

# Entry point of the module
# Return inputs, preprocess_pipeline
# - inputs: dict of symbolic tf.keras.Input objects matching the names and data-types of the columns
# - preprocess_pipeline: keras model, see dot format in README.md
# To generate dot format of preprocess_pipeline model:
# ```
# from preprocessing import preprocess_model
# import tensorflow as tf
# 
# _,titanic_preprocessing, _, _ = preprocess_model()
# tf.keras.utils.plot_model(model=titanic_preprocessing, to_file='preprocessing_pipeline.png', rankdir="LR", dpi=300, show_shapes=True)
# ```
def preprocess_model():
    titanic_features, titanic_labels = load_and_prepare_data()
    inputs = instantiate_tensors(titanic_features)

    # concatenate tensor graph in a list
    preprocessed_inputs = preprocess_numeric_features(titanic_features, inputs) + preprocess_categorical_features(titanic_features, inputs)

    preprocessed_inputs_cat = layers.Concatenate()(preprocessed_inputs)
    preprocess_pipeline = tf.keras.Model(inputs, preprocessed_inputs_cat, name='preprocessing_pipeline')
    return inputs, preprocess_pipeline, titanic_features, titanic_labels