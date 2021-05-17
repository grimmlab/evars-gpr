import pandas as pd
import numpy as np
import sklearn
import copy

from evaluation import EvaluationHelper, SimpleBaselines
from training import TrainHelper


class BaseModel:
    """
    Class containing Base model and methods
    """

    def __init__(self, target_column: str, seasonal_periods: int, name: str, one_step_ahead=False):
        self.target_column = target_column
        self.one_step_ahead = one_step_ahead
        self.seasonal_periods = seasonal_periods
        self.name = name

    def train(self, train: pd.DataFrame, cross_val_call: bool = False) -> dict:
        """
        Train model
        :param train: train set
        :param cross_val_call: called to perform cross validation
        :return dictionary with cross validated scores (if specified)
        """
        raise NotImplementedError

    def get_cross_val_score(self, train: pd.DataFrame) -> tuple:
        """
        Deliver cross validated evaluation scores
        :param train: train set
        :return: dictionary with mean and std of cross validated evaluation scores
        """
        # backup model so train on full dataset afterwards is independet of cv training
        backup_model = copy.deepcopy(self.model)
        if train.shape[0] < 30:
            print('Train set too small for Cross Validation')
            return {}, backup_model
        train = train.copy()
        rmse_lst = []
        splitter = sklearn.model_selection.ShuffleSplit(n_splits=5, test_size=0.2, random_state=0)
        prefix = 'shuf_'
        for train_index, test_index in splitter.split(train):
            cv_train, cv_test = train.loc[train.index[train_index]], train.loc[train.index[test_index]]
            self.model = copy.deepcopy(backup_model)
            try:
                self.train(train=cv_train)
                predictions = self.predict(test=cv_test, train=cv_train, cv_call=True)
                rmse_test = EvaluationHelper.rmse(actual=cv_test[self.target_column],
                                                  prediction=predictions['Prediction'])
                rmse_lst.append(rmse_test)
            except Exception as exc:
                print(exc)
                continue
        rmse_mean = np.mean(np.asarray(rmse_lst))
        rmse_std = np.std(np.asarray(rmse_lst))
        cv_dict = {prefix + 'cv_rmse_mean': rmse_mean, prefix + 'cv_rmse_std': rmse_std}
        for cv_number in range(len(rmse_lst)):
            cv_dict[prefix + 'cv_rmse_' + str(cv_number)] = rmse_lst[cv_number]
        return cv_dict, backup_model

    def insample(self, train: pd.DataFrame) -> pd.DataFrame:
        """
        Deliver (back-transformed) insample predictions
        :param train: train set
        :return: DataFrame with insample predictions
        """
        raise NotImplementedError

    def predict(self, test: pd.DataFrame, train: pd.DataFrame, cv_call: bool = False) -> pd.DataFrame:
        """
        Deliver (back-transformed), if specified one step ahead, out-of-sample predictions
        :param test: test set
        :param train: train set
        :param cv_call: do not perform one_step_ahead for cv calls
        :return: DataFrame with predictions, upper and lower confidence level
        """
        raise NotImplementedError

    def evaluate(self, train: pd.DataFrame, test: pd.DataFrame) -> dict:
        """
        Evaluate model against all implemented evaluation metrics and baseline methods.
        Deliver dictionary with evaluation metrics.
        :param train: train set
        :param test: test set
        :return: dictionary with evaluation metrics of model and all baseline methods
        """
        """
        insample_rw, prediction_rw = SimpleBaselines.RandomWalk(one_step_ahead=self.one_step_ahead)\
            .get_insample_prediction(train=train, test=test, target_column=self.target_column)
        insample_seasrw, prediction_seasrw = SimpleBaselines.RandomWalk(one_step_ahead=self.one_step_ahead)\
            .get_insample_prediction(train=train, test=test, target_column=self.target_column,
                                     seasonal_periods=self.seasonal_periods)
        insample_ha, prediction_ha = SimpleBaselines.HistoricalAverage(one_step_ahead=self.one_step_ahead)\
            .get_insample_prediction(train=train, test=test, target_column=self.target_column)
        insample_model = self.insample(train=train)
        """
        prediction_model = self.predict(test=test, train=train)
        """
        rmse_train_rw = EvaluationHelper.rmse(
            actual=train[self.target_column], prediction=insample_rw['Insample'])
        rmse_test_rw = EvaluationHelper.rmse(
            actual=test[self.target_column], prediction=prediction_rw['Prediction'])
        rmse_train_seasrw = EvaluationHelper.rmse(
            actual=train[self.target_column], prediction=insample_seasrw['Insample'])
        rmse_test_seasrw = EvaluationHelper.rmse(
            actual=test[self.target_column], prediction=prediction_seasrw['Prediction'])
        rmse_train_ha = EvaluationHelper.rmse(
            actual=train[self.target_column], prediction=insample_ha['Insample'])
        rmse_test_ha = EvaluationHelper.rmse(
            actual=test[self.target_column], prediction=prediction_ha['Prediction'])
        rmse_train_model = EvaluationHelper.rmse(
            actual=train[self.target_column], prediction=insample_model['Insample'])
        """
        rmse_test_model = EvaluationHelper.rmse(
            actual=test[self.target_column], prediction=prediction_model['Prediction'])
        """
        return {'RMSE_Train_RW': rmse_train_rw,
                'RMSE_Test_RW': rmse_test_rw, 
                'RMSE_Train_seasRW': rmse_train_seasrw,
                'RMSE_Test_seasRW': rmse_test_seasrw,
                'RMSE_Train_HA': rmse_train_ha, 
                'RMSE_Test_HA': rmse_test_ha,
                'RMSE_Train': rmse_train_model,
                'RMSE_Test': rmse_test_model
                }
        """
        return {'RMSE_Test': rmse_test_model}
