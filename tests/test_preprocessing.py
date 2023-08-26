from data.data import *
from preprocessing.preprocess import preprocess_model, clean_data
import pandas as pd
import numpy as np

def test_preprocess_model():
    preprocessing_model = preprocess_model()
    sample_features, _ = clean_data()
    sample_features_dict = {name: np.array(value) for name, value in sample_features.iloc[:10].items()}
    preprocessed_sample = preprocessing_model.predict(sample_features_dict)

    assert preprocessed_sample.shape == (10,16)