import sklearn.gaussian_process
import pandas as pd
import numpy as np
import copy

from training import ModelsBaseClass


class GaussianProcessRegression(ModelsBaseClass.BaseModel):
    """Class containing Gaussian Process Regression Model"""
    def __init__(self, target_column: str, seasonal_periods: int, kernel=None, alpha: float = 1e-10,
                 n_restarts_optimizer: int = 10, one_step_ahead: bool = False, standardize: bool = False,
                 normalize_y: bool = False):
        """
        :param target_column: target_column for prediction
        :param seasonal_periods: period of seasonality
        :param kernel: kernel to use for GPR
        :param alpha: value added to diagonal of kernel matrix
        :param n_restarts_optimizer: number of restarts of optimizer
        :param one_step_ahead: perform one step ahead prediction
        :param standardize: standardize all features according to train mean and std
        :param normalize_y: normalize only target variable
        """
        super().__init__(target_column=target_column, seasonal_periods=seasonal_periods,
                         name='GaussianProcessRegression_sklearn', one_step_ahead=one_step_ahead)
        self.model = sklearn.gaussian_process.GaussianProcessRegressor(kernel=kernel, alpha=alpha, copy_X_train=True,
                                                                       optimizer='fmin_l_bfgs_b', random_state=42,
                                                                       n_restarts_optimizer=n_restarts_optimizer,
                                                                       normalize_y=normalize_y)
        self.standardize = standardize
        self.train_mean = None
        self.train_std = None

    def train(self, train: pd.DataFrame, cross_val_call: bool = False) -> dict:
        """
        Train GPR model
        :param train: train set
        :param cross_val_call: called to perform cross validation
        :return dictionary with cross validated scores (if specified)
        """
        cross_val_score_dict = {}
        if cross_val_call:
            cross_val_score_dict, self.model = self.get_cross_val_score(train=train)
        if self.standardize:
            self.train_mean = train.mean()
            self.train_std = train.std().replace(to_replace=0, value=1)
            train = train.copy().apply(
                lambda x: ((x - self.train_mean[x.name]) / self.train_std[x.name])
                if (not(x.name.startswith('public') or x.name.startswith('school') or x.name == self.target_column))
                else x, axis=0)
        self.model.fit(X=train.drop([self.target_column], axis=1), y=train[self.target_column])
        return cross_val_score_dict

    def update(self, train: pd.DataFrame, model: sklearn.gaussian_process.GaussianProcessRegressor) \
            -> sklearn.gaussian_process.GaussianProcessRegressor:
        """
        Update existing GPR model due to new samples
        :param train: train set with new samples
        :param model: model to update
        :return: updated model
        """
        return model.fit(X=train.drop([self.target_column], axis=1), y=train[self.target_column])

    def insample(self, train: pd.DataFrame) -> pd.DataFrame:
        """
        Deliver (back-transformed) insample predictions
        :param train: train set
        :return: DataFrame with insample predictions
        """
        insample = pd.DataFrame(data=self.model.predict(X=self.model.X_train_), index=train.index, columns=['Insample'])
        return insample

    def predict(self, test: pd.DataFrame, train: pd.DataFrame, cv_call: bool = False) -> pd.DataFrame:
        """
        Deliver (back-transformed), if specified one step ahead, out-of-sample predictions
        :param test: test set
        :param train: train set
        :param cv_call: do not perform one_step_ahead for cv calls
        :return: DataFrame with predictions, upper and lower confidence level
        """
        if self.one_step_ahead is not False and cv_call is False:
            predict_lst = []
            sig_lst = []
            bias = 0
            train_mean = self.train_mean
            train_std = self.train_std
            train_manip = train.copy()
            # deep copy model as predict function should not change class model
            model = copy.deepcopy(self.model)
            for i in range(0, test.shape[0]):
                test_sample = test.iloc[[i]]
                train_manip = train_manip.append(test_sample)
                if self.one_step_ahead == 'mw':
                    train_manip = train_manip.iloc[1:]
                if self.standardize:
                    # Normalize test sample
                    test_sample = test_sample.copy().apply(
                        lambda x: ((x - train_mean[x.name]) / train_std[x.name])
                        if (not (x.name.startswith('public') or
                                 x.name.startswith('school') or
                                 x.name == self.target_column))
                        else x, axis=0)
                fc, sigma = model.predict(X=test_sample.drop([self.target_column], axis=1),
                                          return_std=True)
                if self.standardize:
                    # Update train mean and std
                    train_mean = train_manip.mean()
                    train_std = train_manip.std().replace(to_replace=0, value=1)
                # Refit model if refitting cycle is reached
                if (self.one_step_ahead == 'mw') or \
                        ((self.one_step_ahead != 0) and ((i + 1) % self.one_step_ahead == 0)):
                    if self.standardize:
                        # Normalize train_manip_refit with updated mean and std
                        train_manip_refit = train_manip.copy().apply(
                            lambda x: ((x - train_mean[x.name]) / train_std[x.name])
                            if (not (x.name.startswith('public') or
                                     x.name.startswith('school') or
                                     x.name == self.target_column))
                            else x, axis=0)
                    else:
                        train_manip_refit = train_manip
                    model = self.update(train=train_manip_refit, model=model)
                if self.one_step_ahead == 'mw':
                    # Add Bias
                    fc += bias
                    sigma += bias
                    # Bias Update: Initial bias = 0
                    # bias_0(k) = y_obs(k-1) - y_mod(k-1)
                    bias_0 = (test_sample[self.target_column].values - fc)[0]
                    # bias(k) = w*bias_0(k) + (1-w)*bias(k-1), w = 0.1...0.9, here: w = 0.8 (see (Ni, 2012))
                    w = 0.8
                    bias = w * bias_0 + (1-w) * bias
                predict_lst.append(fc)
                sig_lst.append(sigma)
            predict = np.array(predict_lst).flatten()
            sig = np.array(sig_lst).flatten()
        else:
            if self.standardize:
                test = test.copy().apply(
                        lambda x: ((x - self.train_mean[x.name]) / self.train_std[x.name])
                        if (not (x.name.startswith('public') or
                                 x.name.startswith('school') or
                                 x.name == self.target_column))
                        else x, axis=0)
            predict, sig = self.model.predict(X=test.drop([self.target_column], axis=1), return_std=True)
        predictions = pd.DataFrame({'Prediction': predict,
                                    'LowerConf': predict-1.96*sig, 'UpperConf': predict+1.96*sig},
                                   index=test.index)

        return predictions
