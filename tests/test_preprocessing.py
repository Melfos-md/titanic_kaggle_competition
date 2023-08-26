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


    #print("Preprocessed sample shape:", preprocessed_sample.shape)
    #print("Preprocessed sample data:", preprocessed_sample)

def test_handle_missing_values():
    preprocessing_model = preprocess_model()
    sample_features.iloc[0, 0] = np.nan  # Introduce a NaN value
    sample_features_dict = {name: np.array(value) for name, value in sample_features.iloc[:10].items()}
    print(sample_features_dict['Pclass'])
    print(sample_features_dict['Pclass'].dtype)
    preprocessed_sample = preprocessing_model.predict(sample_features_dict)
    assert not np.isnan(preprocessed_sample).any()

# Add more tests following the same pattern
