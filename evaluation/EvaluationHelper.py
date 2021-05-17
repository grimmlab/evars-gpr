import numpy as np
import pandas as pd


def nan_checker(series: pd.Series):
    """Function checking if any value in series is nan"""
    return series.isna().values.any()


def drop_dummy_values(actual: pd.Series, prediction: pd.Series) -> tuple:
    """Function dropping indices with dummy values in prediction at both series"""
    indices = prediction.index[prediction == -9999]
    return actual.drop(indices), prediction.drop(indices)


def rmse(actual: pd.Series, prediction: pd.Series) -> float:
    """
    Function delivering Root Mean Squared Error between prediction and actual values
    :param actual: actual values
    :param prediction: prediction values
    :return: RMSE between prediciton and actual values
    """
    if nan_checker(actual) or nan_checker(prediction):
        raise NameError('Found NaNs - stopped calculation of evaluation metric')
    return np.mean((prediction - actual) ** 2) ** 0.5
