# -*- coding: utf-8 -*-

"""
Module that defines utils functions
"""

import pickle


def save_pickle(data_dict, filepath):
    """
    Save data to pickle file

    Parameters
    ----------
    data_dict : dict
        data to save
    filepath : str
        absolute path to the file

    Returns
    -------
        None
    """
    with open(filepath, "wb") as handle:
        pickle.dump(data_dict, handle, protocol=pickle.HIGHEST_PROTOCOL)


def load_pickle(filepath):
    """
    Load data from pickle file

    Parameters
    ----------
    filepath : str
        absolute path to the file

    Returns
    -------
        data stored in the pickle file
    """
    with open(filepath, "rb") as handle:
        data = pickle.load(handle)
    return data
