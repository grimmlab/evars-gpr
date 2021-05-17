import numpy as np
import copy
import bocd
import changefinder
import pandas as pd

from training import TrainHelper, ModelsGaussianProcessRegression


def run_evars_gpr(base_model: ModelsGaussianProcessRegression.GaussianProcessRegression,
                  data: pd.DataFrame, season_length: int, target_column: str, train_ind: int,
                  comparison_partners: bool = False, da: str = 'scaled', cpd: str = 'cf',
                  scale_thr: float = 0.1, scale_seasons: int = 2,
                  scale_window: int = None, scale_window_factor: float = 0.1, scale_window_minimum: int = 2,
                  const_hazard: int = None, const_hazard_factor: int = 2,
                  cf_r: float = 0.4, cf_order: int = 1, cf_smooth: int = 4, cf_thr_perc: int = 90,
                  append: str = 'no', max_samples: int = None, max_samples_factor: int = 10,
                  o_perc: float = 1.1, u_perc: float = 0.1, thr: float = 0.2, under_samp: bool = False,
                  rel_thr: float = 0.5, rel_coef: float = 1.5, verbose: bool = False):
    """
    Run EVARS-GPR algo
    :param base_model: base model fitted during offline phase
    :param data: data to use
    :param season_length: length of one season
    :param target_column: target column
    :param train_ind: index of last train sample
    :param comparison_partners: specify whether to include comparison partners in optimization loop
    :param da: data augmentation method
    :param cpd: change point detection method
    :param scale_thr: threshold for output scaling factor
    :param scale_seasons: number of seasons to consider for calculation of output scaling factor
    :param scale_window: number of samples prior to change point for calculation of output scaling factor
    :param scale_window_factor: scale window as a multiple of the season length
    :param scale_window_minimum: minimum of the scale window
    :param const_hazard: constant hazard value in case of bocpd
    :param const_hazard_factor: constant hazard value as a multiple of the season length
    :param cf_r: r value (forgetting factor) for changefinder
    :param cf_order: order of SDAR models for changefinder
    :param cf_smooth: smoothing constant for changefinder
    :param cf_thr_perc: percentile of offline anomaly scores to use for declaration of a change point
    :param append: specify whether to append original and scaled dataset for da or not
    :param max_samples: maximum samples to consider for data augmentation
    :param max_samples_factor: maximum samples to consider for data augmentation as a multiple of the season length
    :param o_perc: oversampling percentage for GN
    :param u_perc: undersampling percentage for GN
    :param thr: threshold for GN
    :param under_samp: specify whether to undersample for SMOGN
    :param rel_thr: relevance threshold for SMOGN
    :param rel_coef: relevance coefficient for SMOGN
    :param verbose: print debug info
    :return: list of detected change points, evars-gpr predictions, dictionary with predictions of comparison partners,
    number of refits
    """
    scale_window = max(scale_window_minimum, int(scale_window_factor * season_length)) \
        if scale_window is None else scale_window
    const_hazard = const_hazard_factor * season_length if const_hazard is None else const_hazard
    max_samples = max_samples_factor * season_length if max_samples is None else max_samples
    data = data.copy()
    data.reset_index(drop=True, inplace=True)
    train = data[:train_ind]
    # setup cpd
    y_deseas = data[target_column].diff(season_length).dropna().values
    y_train_deseas = y_deseas[:train_ind-season_length]
    if cpd == 'bocd':
        mean = np.mean(y_train_deseas)
        std = np.std(y_train_deseas)
        train_std = (y_train_deseas - mean) / std
        bc = bocd.BayesianOnlineChangePointDetection(bocd.ConstantHazard(const_hazard),
                                                     bocd.StudentT(mu=0, kappa=1, alpha=1, beta=1))
        for i, d_bocd_train in enumerate(train_std):
            bc.update(d_bocd_train)
    elif cpd == 'cf':
        scores = []
        cf = changefinder.ChangeFinder(r=cf_r, order=cf_order, smooth=cf_smooth)
        for i in y_train_deseas:
            scores.append(cf.update(i))
        cf_threshold = np.percentile(scores, cf_thr_perc)
        if verbose:
            print('CF_Scores_Train: threshold=' + str(cf_threshold)
                  + ', mean=' + str(np.mean(scores)) + ', max=' + str(np.max(scores))
                  + ', 70perc=' + str(np.percentile(scores, 70)) + ', 80perc=' + str(np.percentile(scores, 80))
                  + ', 90perc=' + str(np.percentile(scores, 90)) + ', 95perc=' + str(np.percentile(scores, 95))
                  )
    # online part
    test = data[train_ind:]
    y_train_deseas_manip = y_train_deseas.copy()
    rt_mle = np.empty(test[target_column].shape)
    predictions = None
    train_manip = train.copy()
    model = copy.deepcopy(base_model)
    # setup comparison partners
    if comparison_partners:
        model_cpd_retrain_full = copy.deepcopy(base_model)
        predictions_cpd_retrain_full = None
        model_cpd_moving_window_full = copy.deepcopy(base_model)
        predictions_cpd_moving_window_full = None
        predictions_cpd_scaled_full = None
    cp_detected = []
    output_scale_old = 1
    output_scale = 1
    n_refits = 0
    # iterate over whole test set
    for index in test.index:
        sample = test.loc[index]
        train_manip = train_manip.append(sample)
        # predict next target value
        prediction = model.predict(test=sample.to_frame().T, train=train_manip)
        if predictions is None:
            predictions = prediction.copy()
        else:
            predictions = predictions.append(prediction)
        # get predictions of comparison partners if specified
        if comparison_partners:
            prediction_cpd_retrain_full = model_cpd_retrain_full.predict(test=sample.to_frame().T, train=train_manip)
            prediction_cpd_moving_window_full = model_cpd_moving_window_full.predict(test=sample.to_frame().T,
                                                                                     train=train_manip)
            prediction_cpd_scaled_full = prediction.copy()
            prediction_cpd_scaled_full *= output_scale_old
            if predictions_cpd_retrain_full is None:
                predictions_cpd_retrain_full = prediction_cpd_retrain_full.copy()
                predictions_cpd_moving_window_full = prediction_cpd_moving_window_full.copy()
                predictions_cpd_scaled_full = prediction_cpd_scaled_full.copy()
            else:
                predictions_cpd_retrain_full = predictions_cpd_retrain_full.append(prediction_cpd_retrain_full)
                predictions_cpd_moving_window_full = \
                    predictions_cpd_moving_window_full.append(prediction_cpd_moving_window_full)
                predictions_cpd_scaled_full = predictions_cpd_scaled_full.append(prediction_cpd_scaled_full)
        # CPD
        change_point_detected = False
        y_deseas = sample[target_column] - data.loc[index-season_length][target_column]
        if cpd == 'bocd':
            d_bocd = (y_deseas - mean) / std
            bc.update(d_bocd)
            rt_mle_index = index-train_ind
            rt_mle[rt_mle_index] = bc.rt
            y_train_deseas_manip = np.append(y_train_deseas_manip, y_deseas)
            mean = np.mean(y_train_deseas_manip)
            std = np.std(y_train_deseas_manip)
            if rt_mle_index > 0 and (rt_mle[rt_mle_index] - rt_mle[rt_mle_index-1] < 0):
                change_point_detected = True
                curr_ind = rt_mle_index
        elif cpd == 'cf':
            score = cf.update(y_deseas)
            scores.append(score)
            if score >= cf_threshold:
                if verbose:
                    print('Anomaly Score ' + str(score) + ' > ' + 'threshold ' + str(cf_threshold))
                change_point_detected = True
                curr_ind = index - train_ind
        # Trigger remaining EVARS-GPR procedures if a change point is detected
        if change_point_detected:
            if verbose:
                print('CP Detected ' + str(curr_ind + train.shape[0]))
            cp_detected.append(curr_ind)
            try:
                # Calculate output scaling factor
                change_point_index = curr_ind + train.shape[0]
                mean_now = np.mean(data[change_point_index-scale_window+1:change_point_index+1][target_column])
                mean_prev_seas_1 = \
                    np.mean(data[change_point_index-season_length-scale_window+1:change_point_index-season_length+1]
                            [target_column])
                mean_prev_seas_2 = \
                    np.mean(data[change_point_index-2*season_length-scale_window+1:change_point_index-2*season_length+1]
                            [target_column])
                if scale_seasons == 1:
                    output_scale = mean_now / mean_prev_seas_1
                elif scale_seasons == 2:
                    output_scale = np.mean([mean_now / mean_prev_seas_1, mean_now / mean_prev_seas_2])
                if output_scale == 0:
                    raise Exception
                if verbose:
                    print('ScaleDiff=' + str(np.abs(output_scale - output_scale_old) / output_scale_old))
                # Check deviation to previous scale factor
                if np.abs(output_scale - output_scale_old) / output_scale_old > scale_thr:
                    n_refits += 1
                    if verbose:
                        print('try to retrain model: ' + str(change_point_index)
                              + ' , output_scale=' + str(output_scale))
                    if output_scale > 1:
                        focus = 'high'
                    else:
                        focus = 'low'
                    # augment data
                    train_samples = TrainHelper.get_augmented_data(data=data, target_column=target_column, da=da,
                                                                   change_point_index=curr_ind + train.shape[0],
                                                                   output_scale=output_scale,
                                                                   rel_coef=rel_coef, rel_thr=rel_thr,
                                                                   under_samp=under_samp, focus=focus,
                                                                   o_perc=o_perc, u_perc=u_perc, thr=thr,
                                                                   append=append, max_samples=max_samples)
                    # retrain current model
                    model = ModelsGaussianProcessRegression.GaussianProcessRegression(
                        target_column=base_model.target_column, seasonal_periods=base_model.seasonal_periods,
                        kernel=base_model.model.kernel_, alpha=base_model.model.alpha,
                        n_restarts_optimizer=base_model.model.n_restarts_optimizer,
                        standardize=base_model.standardize, normalize_y=base_model.model.normalize_y,
                        one_step_ahead=base_model.one_step_ahead)
                    model.train(train_samples, cross_val_call=False)
                    if comparison_partners:
                        train_data = data.copy()[:change_point_index+1]
                        # cpd Retrain
                        model_cpd_retrain_full = ModelsGaussianProcessRegression.GaussianProcessRegression(
                            target_column=base_model.target_column, seasonal_periods=base_model.seasonal_periods,
                            kernel=base_model.model.kernel_, alpha=base_model.model.alpha,
                            n_restarts_optimizer=base_model.model.n_restarts_optimizer,
                            standardize=base_model.standardize, normalize_y=base_model.model.normalize_y,
                            one_step_ahead=base_model.one_step_ahead)
                        model_cpd_retrain_full.train(train_data, cross_val_call=False)
                        # Moving Window
                        model_cpd_moving_window_full = ModelsGaussianProcessRegression.GaussianProcessRegression(
                            target_column=base_model.target_column, seasonal_periods=base_model.seasonal_periods,
                            kernel=base_model.model.kernel_, alpha=base_model.model.alpha,
                            n_restarts_optimizer=base_model.model.n_restarts_optimizer,
                            standardize=base_model.standardize, normalize_y=base_model.model.normalize_y,
                            one_step_ahead=base_model.one_step_ahead)
                        model_cpd_moving_window_full.train(train_data[-season_length:], cross_val_call=False)
                    # in case of a successful refit change output_scale_old
                    output_scale_old = output_scale
            except Exception as exc:
                print(exc)
    if comparison_partners:
        comparison_partners_dict = {'cpd_retrain_full': predictions_cpd_retrain_full,
                                    'cpd_cpd_moving_window_full': predictions_cpd_moving_window_full,
                                    'cpd_scaled_full': predictions_cpd_scaled_full
                                    }
    else:
        comparison_partners_dict = {}
    return cp_detected, predictions, comparison_partners_dict, n_refits
