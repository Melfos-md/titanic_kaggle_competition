from preprocessing.preprocess import load_and_prepare_data, extract_ticket_prefix, extract_ticket_number, get_ticket_number_length, extract_title
import pandas as pd
import numpy as np


def test_load_and_prepare_data_train():
    features, labels = load_and_prepare_data('data/train.csv', live=False, nrows=10)

    assert features.shape == (10,11)
    assert labels.shape == (10,)


def test_load_and_prepare_data_live():
    features, id = load_and_prepare_data('data/train.csv', live=True, nrows=10)

    assert features.shape == (10,11)
    assert id.shape == (10,)

def test_extract_ticket_prefix():
    test_tickets = pd.Series(["A/5 21171", "PC 17599", 
                              "STON/O2. 3101282", "113803"])
    result_series = extract_ticket_prefix(test_tickets)
    expected_series = pd.Series(['A', 'PC', 'STON', 'Missing'])
    assert result_series.equals(expected_series)


def test_extract_ticket_number():
    test_tickets_series = pd.Series(["A/5 21171", "PC 17599", "STON/O2. 3101282", "113803", "ABCD"])
    expected = pd.Series(["21171", "17599", "3101282", "113803", "Missing"])
    results = extract_ticket_number(test_tickets_series)
    assert results.equals(expected)

def test_get_ticket_number_length():
    test_tickets_series = pd.Series(["A/5 21171", "PC 17599", "STON/O2. 3101282", "113803", "ABCD"])
    expected = pd.Series([5, 5, 7, 6, 0])
    result = get_ticket_number_length(test_tickets_series)
    assert result.equals(expected)

def test_extract_title():
    names_series = pd.Series([
    "Braund, Mr. Owen Harris",
    "Cumings, Mrs. John Bradley (Florence Briggs Thayer)",
    "Heikkinen, Miss. Laina",
    "Futrelle, Mrs. Jacques Heath (Lily May Peel)",
    "Allen, Mr. William Henry",
    "Moran, Mr. James",
    "McCarthy, Mr. Timothy J",
    "Palsson, Master. Gosta Leonard",
    "Johnson, Mrs. Oscar W (Elisabeth Vilhelmina Berg)",
    "Nasser, Mrs. Nicholas (Adele Achem)"
    ])

    expected = pd.Series(['Mr', 'Mrs', 'Miss', 'Mrs', 'Mr', 'Mr', 'Mr', 'Master', 'Mrs', 'Mrs'])
    result = extract_title(names_series)
    assert result.equals(expected)