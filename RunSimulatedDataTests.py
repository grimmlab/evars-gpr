import warnings
import numpy as np
import math
import pandas as pd
from sklearn.gaussian_process.kernels import ExpSineSquared
import random
import scipy
import sklearn
import argparse

from training import ModelsGaussianProcessRegression, EvarsGpr, TrainHelper
from evaluation import EvaluationHelper

if __name__ == '__main__':
    """
    Script to run tests based on different scenarios of manipulated simulated data.
    The DA and CPD method as well as the seasonal length to simulated can be specified.
    Then, a random search of 100 parameter combinations is started.
    """
    parser = argparse.ArgumentParser()
    # General Params
    parser.add_argument("-da", "--data_augmentation", type=str, default='scaled',
                        help="specify data augmentation")
    parser.add_argument("-seaslen", "--seasonal_length", type=int, default=50, help="specify season length")
    parser.add_argument("-cpd", "--change_point_detection", type=str, default='cf',
                        help="specify cpd algo: bocd or cf")
    args = parser.parse_args()
    da = args.data_augmentation
    seas_len = args.seasonal_length
    cpd = args.change_point_detection
    warnings.simplefilter("ignore")
    np.random.seed(0)
    random.seed(0)

    # Generate base sine data
    season_length = seas_len
    n_periods = 5
    X = TrainHelper.get_periodic_noisy_x(x_base=np.linspace(-0.5 * math.pi, 1.5 * math.pi, season_length),
                                         n_periods=n_periods)
    Y = TrainHelper.noisy_sin(X)
    data = pd.DataFrame(columns=['X', 'Y'])
    data['X'] = X
    data['Y'] = Y
    train_ind = int(0.6 * data.shape[0])
    train = data[0:train_ind]

    # Train offline base model
    target_column = 'Y'
    kernel = ExpSineSquared()
    alpha = 0.1
    n_restarts_optimizer = 10
    standardize = False
    normalize_y = True
    model_sine = ModelsGaussianProcessRegression.GaussianProcessRegression(
        target_column=target_column, seasonal_periods=season_length, kernel=kernel,
        alpha=alpha, n_restarts_optimizer=n_restarts_optimizer,
        standardize=standardize, normalize_y=normalize_y,
        one_step_ahead=False)
    model_sine.train(train=train, cross_val_call=False)

    # specify parameters and manipulation
    if seas_len == 20:
        const_hazard = [seas_len, 2*seas_len, 3*seas_len, 5*seas_len, 10*seas_len, 1000]
        scale_window = [1, 2]
        start_ind_full_seas1 = 60
        end_ind_full_seas1 = 80
        start_ind_full_seas2 = 80
        end_ind_full_seas2 = 100
        start_ind_begin_seas_1 = 60
        end_ind_begin_seas1 = 70
        start_ind_end_seas1 = 70
        end_ind_end_seas1 = 80
        start_ind_mid_seas1 = 65
        end_ind_mid_seas1 = 75
        slow_slope = 0.1
        fast_slope = 0.25
        small_inc = 1.5
        big_inc = 2
        big_dec = 0.5
    elif seas_len == 50:
        const_hazard = [seas_len, 2*seas_len, 3*seas_len, 5*seas_len, 10*seas_len, 1000]
        scale_window = [2, 3, 5]
        start_ind_full_seas1 = 150
        end_ind_full_seas1 = 200
        start_ind_full_seas2 = 200
        end_ind_full_seas2 = 250
        start_ind_begin_seas_1 = 150
        end_ind_begin_seas1 = 175
        start_ind_end_seas1 = 175
        end_ind_end_seas1 = 200
        start_ind_mid_seas1 = 162
        end_ind_mid_seas1 = 188
        slow_slope = 0.1
        fast_slope = 0.25
        small_inc = 1.5
        big_inc = 2.5
        big_dec = 0.5
    elif seas_len == 100:
        const_hazard = [seas_len, 2*seas_len, 3*seas_len, 5*seas_len, 10*seas_len]
        scale_window = [5, 10]
        start_ind_full_seas1 = 300
        end_ind_full_seas1 = 400
        start_ind_full_seas2 = 400
        end_ind_full_seas2 = 500
        start_ind_begin_seas_1 = 300
        end_ind_begin_seas1 = 350
        start_ind_end_seas1 = 350
        end_ind_end_seas1 = 400
        start_ind_mid_seas1 = 325
        end_ind_mid_seas1 = 375
        slow_slope = 0.1
        fast_slope = 0.25
        small_inc = 1.5
        big_inc = 2.5
        big_dec = 0.5

    if da == 'scaled':
        param_grid = dict(
            scale_thr=[0.05, 0.1, 0.2],
            scale_seasons=[1, 2],
            scale_window=scale_window,
            append=['not']
        )
    elif da == 'smogn':
        param_grid = dict(
            under_samp=[False, True],
            rel_thr=scipy.stats.uniform(loc=0.01, scale=0.98),
            rel_coef=scipy.stats.uniform(loc=1.0, scale=0.8),
            scale_window=scale_window,
            scale_thr=[0.05, 0.1, 0.2],
            scale_seasons=[1, 2],
            append=['not', 'before']
        )
    elif da == 'gn':
        param_grid = dict(
            o_percentage=scipy.stats.uniform(loc=1.50, scale=2),  # scipy.stats.uniform(loc=1.01, scale=1.99),
            u_percentage=scipy.stats.uniform(loc=0.1, scale=0.2),  # scipy.stats.uniform(loc=0.01, scale=0.98),
            threshold=scipy.stats.uniform(loc=0.2, scale=0.5),  # scipy.stats.uniform(loc=0.1, scale=0.6),
            scale_window=scale_window,
            scale_thr=[0.05, 0.1, 0.2],
            scale_seasons=[1, 2],
            append=['not', 'before']
        )
    if cpd == 'bocd':
        param_grid['const_hazard'] = const_hazard
    elif cpd == 'cf':
        param_grid['cf_r'] = scipy.stats.uniform(loc=0.01, scale=0.69)
        param_grid['cf_order'] = scipy.stats.randint(1, 3)
        param_grid['cf_smooth'] = scipy.stats.randint(3, 10)
        param_grid['cf_thr_perc'] = [70, 80, 90, 95]

    max_iter = 100

    params_lst = list(sklearn.model_selection.ParameterSampler(param_grid, n_iter=max_iter, random_state=0))

    scenario_lst = [
        {'name': 'slow_inc+small+full_season1', 'start_ind': start_ind_full_seas1, 'end_ind': end_ind_full_seas1,
         'slope': slow_slope, 'max_factor': small_inc},
        {'name': 'slow_inc+small+full_season2', 'start_ind': start_ind_full_seas2, 'end_ind': end_ind_full_seas2,
         'slope': slow_slope, 'max_factor': small_inc},
        {'name': 'slow_inc+big+full_season1', 'start_ind': start_ind_full_seas1, 'end_ind': end_ind_full_seas1,
         'slope': slow_slope, 'max_factor': big_inc},
        {'name': 'slow_inc+big+full_season2', 'start_ind': start_ind_full_seas2, 'end_ind': end_ind_full_seas2,
         'slope': slow_slope, 'max_factor': big_inc},
        {'name': 'fast_inc+small+full_season1', 'start_ind': start_ind_full_seas1, 'end_ind': end_ind_full_seas1,
         'slope': fast_slope, 'max_factor': small_inc},
        {'name': 'fast_inc+small+full_season2', 'start_ind': start_ind_full_seas2, 'end_ind': end_ind_full_seas2,
         'slope': fast_slope, 'max_factor': small_inc},
        {'name': 'fast_inc+big+full_season1', 'start_ind': start_ind_full_seas1, 'end_ind': end_ind_full_seas1,
         'slope': fast_slope, 'max_factor': big_inc},
        {'name': 'fast_inc+big+full_season2', 'start_ind': start_ind_full_seas2, 'end_ind': end_ind_full_seas2,
         'slope': fast_slope, 'max_factor': big_inc},
        {'name': 'slow_dec+big+full_season1', 'start_ind': start_ind_full_seas1, 'end_ind': end_ind_full_seas1,
         'slope': slow_slope, 'max_factor': big_dec},
        {'name': 'slow_dec+big+full_season2', 'start_ind': start_ind_full_seas2, 'end_ind': end_ind_full_seas2,
         'slope': slow_slope, 'max_factor': big_dec},
        {'name': 'slow_inc+small+begin_season1', 'start_ind': start_ind_begin_seas_1, 'end_ind': end_ind_begin_seas1,
         'slope': slow_slope, 'max_factor': small_inc},
        {'name': 'fast_inc+small+begin_season1', 'start_ind': start_ind_begin_seas_1, 'end_ind': end_ind_begin_seas1,
         'slope': fast_slope, 'max_factor': small_inc},
        {'name': 'fast_inc+big+begin_season1', 'start_ind': start_ind_begin_seas_1, 'end_ind': end_ind_begin_seas1,
         'slope': fast_slope, 'max_factor': big_inc},
        {'name': 'slow_dec+big+begin_season1', 'start_ind': start_ind_begin_seas_1, 'end_ind': end_ind_begin_seas1,
         'slope': slow_slope, 'max_factor': big_dec},
        {'name': 'slow_inc+small+end_season1', 'start_ind': start_ind_end_seas1, 'end_ind': end_ind_end_seas1,
         'slope': slow_slope, 'max_factor': small_inc},
        {'name': 'fast_inc+small+end_season1', 'start_ind': start_ind_end_seas1, 'end_ind': end_ind_end_seas1,
         'slope': fast_slope, 'max_factor': small_inc},
        {'name': 'fast_inc+big+end_season1', 'start_ind': start_ind_end_seas1, 'end_ind': end_ind_end_seas1,
         'slope': fast_slope, 'max_factor': big_inc},
        {'name': 'slow_dec+big+end_season1', 'start_ind': start_ind_end_seas1, 'end_ind': end_ind_end_seas1,
         'slope': slow_slope, 'max_factor': big_dec},
        {'name': 'slow_inc+small+mid_season1', 'start_ind': start_ind_mid_seas1, 'end_ind': end_ind_mid_seas1,
         'slope': slow_slope, 'max_factor': small_inc},
        {'name': 'slow_inc+big+mid_season1', 'start_ind': start_ind_mid_seas1, 'end_ind': end_ind_mid_seas1,
         'slope': slow_slope, 'max_factor': big_inc},
        {'name': 'fast_inc+small+mid_season1', 'start_ind': start_ind_mid_seas1, 'end_ind': end_ind_mid_seas1,
         'slope': fast_slope, 'max_factor': small_inc},
        {'name': 'fast_inc+big+mid_season1', 'start_ind': start_ind_mid_seas1, 'end_ind': end_ind_mid_seas1,
         'slope': fast_slope, 'max_factor': big_inc},
        {'name': 'slow_dec+big+mid_season1', 'start_ind': start_ind_mid_seas1, 'end_ind': end_ind_mid_seas1,
         'slope': slow_slope, 'max_factor': big_dec},
    ]

    # create documentation info
    columns = ['sampler', 'cpd', 'seas_len', 'scenarios', 'const_hazard', 'scale_window', 'scale_seasons', 'scale_thr',
               'append', 'cf_r', 'cf_order', 'cf_smooth', 'cf_thr_perc',
               'o_perc', 'u_perc', 'threshold',
               'under_samp', 'rel_thr', 'rel_coef', 'n_cps_detected', 'n_refits']
    columns.extend([metric + '_' + stat for metric in ['rmse', 'rmse_ratio', 'rmse_base']
                    for stat in ['mean', 'std']])
    columns.extend(['rmse_ratio_median'])
    columns.extend([metric + '_scen' + str(number) for metric in ['rmse', 'rmse_ratio', 'rmse_base']
                    for number in range(0, len(scenario_lst))])

    doc_results = pd.DataFrame(columns=columns)
    rmse_dict = {}
    rmse_ratio_dict = {}
    n_cps_detected_dict = {}
    n_refits_dict = {}
    rmse_base_lst = []

    # iterate over all scenarios
    for scen_ind, scenario in enumerate(scenario_lst):
        print('----------- Scenario ' + str(scen_ind + 1) + '/' + str(len(scenario_lst)) + '-----------')
        scenario_name = scenario['name']
        start_ind = scenario['start_ind']
        end_ind = scenario['end_ind']
        slope = scenario['slope']
        max_factor = scenario['max_factor']
        try:
            # get results for base model
            data_manip = TrainHelper.get_sloped_dataset(dataset=data, target_column=target_column, start_ind=start_ind,
                                                        end_ind=end_ind, max_factor=max_factor, slope=slope)
            base_predict = model_sine.predict(train=data_manip[:train_ind], test=data_manip[train_ind:])
            rmse_base = EvaluationHelper.rmse(actual=data_manip[train_ind:][target_column],
                                              prediction=base_predict['Prediction'])
            rmse_base_lst.append(rmse_base)
            print('Base Prediction: RMSE=' + str(rmse_base))
            # iterate over all parameter combinations
            for param_ind, params in enumerate(params_lst):
                print('### Param ' + str(param_ind + 1) + '/' + str(len(params_lst))
                      + ' (Scen ' + str(scen_ind + 1) + '/' + str(len(scenario_lst)) + ') ###')
                o_percentage = params['o_percentage'] if 'o_percentage' in params else np.nan
                u_percentage = params['u_percentage'] if 'u_percentage' in params else np.nan
                threshold = params['threshold'] if 'threshold' in params else np.nan
                under_samp = params['under_samp'] if 'under_samp' in params else np.nan
                rel_thr = params['rel_thr'] if 'rel_thr' in params else np.nan
                rel_coef = params['rel_coef'] if 'rel_coef' in params else np.nan
                append = params['append']
                const_hazard = params['const_hazard'] if 'const_hazard' in params else np.nan
                scale_window = params['scale_window']
                scale_thr = params['scale_thr']
                scale_seasons = params['scale_seasons']
                cf_r = params['cf_r'] if 'cf_r' in params else np.nan
                cf_order = params['cf_order'] if 'cf_order' in params else np.nan
                cf_smooth = params['cf_smooth'] if 'cf_smooth' in params else np.nan
                cf_thr_perc = params['cf_thr_perc'] if 'cf_thr_perc' in params else np.nan
                # get results for current configuration
                cp_detected, predictions, _, n_refits = \
                    EvarsGpr.run_evars_gpr(base_model=model_sine, data=data_manip, season_length=season_length,
                                           target_column=target_column, train_ind=train_ind, scale_thr=scale_thr, da=da,
                                           o_perc=o_percentage, u_perc=u_percentage, thr=threshold,
                                           rel_thr=rel_thr, rel_coef=rel_coef, under_samp=under_samp,
                                           append=append, const_hazard=const_hazard, scale_window=scale_window,
                                           scale_seasons=scale_seasons, cpd=cpd, cf_r=cf_r, cf_order=cf_order,
                                           cf_smooth=cf_smooth, cf_thr_perc=cf_thr_perc)
                rmse = EvaluationHelper.rmse(actual=data_manip[train_ind:][target_column],
                                             prediction=predictions['Prediction'])
                print('append=' + append + ', scale_thr=' + str(scale_thr) + ', scale_window=' + str(scale_window))
                if cpd == 'bocd':
                    print(', const_hazard=' + str(const_hazard))
                elif cpd == 'cf':
                    print(', cf_r=' + str(cf_r) + ', cf_order=' + str(cf_order)
                          + ', cf_smooth' + str(cf_smooth) + ', cf_thr_perc=' + str(cf_thr_perc))
                if da == 'gn':
                    print(', o_percentage=' + str(o_percentage) + ' , u_percentage=' + str(u_percentage)
                          + ' , threshold=' + str(threshold))
                elif da == 'smogn':
                    print(', under_samp=' + str(under_samp) + ' , rel_coef=' + str(rel_coef)
                          + ' , rel_thr=' + str(rel_thr))

                print('RMSE: ' + str(rmse))
                # store results
                if param_ind not in rmse_dict.keys():
                    rmse_dict[param_ind] = [rmse]
                    rmse_ratio_dict[param_ind] = [rmse / rmse_base]
                    n_cps_detected_dict[param_ind] = len(cp_detected)
                    n_refits_dict[param_ind] = n_refits
                else:
                    rmse_dict[param_ind].append(rmse)
                    rmse_ratio_dict[param_ind].append(rmse / rmse_base)
                    n_cps_detected_dict[param_ind] += len(cp_detected)
                    n_refits_dict[param_ind] += n_refits
                doc_results.at[param_ind, 'sampler'] = da
                doc_results.at[param_ind, 'cpd'] = cpd
                doc_results.at[param_ind, 'cf_r'] = float(cf_r)
                doc_results.at[param_ind, 'cf_order'] = float(cf_order)
                doc_results.at[param_ind, 'cf_smooth'] = float(cf_smooth)
                doc_results.at[param_ind, 'cf_thr_perc'] = float(cf_thr_perc)
                doc_results.at[param_ind, 'o_perc'] = float(o_percentage)
                doc_results.at[param_ind, 'u_perc'] = float(u_percentage)
                doc_results.at[param_ind, 'threshold'] = float(threshold)
                doc_results.at[param_ind, 'under_samp'] = under_samp
                doc_results.at[param_ind, 'rel_thr'] = float(rel_thr)
                doc_results.at[param_ind, 'rel_coef'] = float(rel_coef)
                doc_results.at[param_ind, 'append'] = append
                doc_results.at[param_ind, 'seas_len'] = season_length
                doc_results.at[param_ind, 'const_hazard'] = float(const_hazard)
                doc_results.at[param_ind, 'scale_window'] = scale_window
                doc_results.at[param_ind, 'scale_thr'] = scale_thr
                doc_results.at[param_ind, 'scale_seasons'] = scale_seasons
                doc_results.at[param_ind, 'rmse_scen' + str(scen_ind)] = float(rmse)
                doc_results.at[param_ind, 'rmse_ratio_scen' + str(scen_ind)] = float(rmse / rmse_base)
                doc_results.at[param_ind, 'rmse_base_scen' + str(scen_ind)] = float(rmse_base)
        except Exception as exc:
            print(exc)
            doc_results['rmse_scen' + str(scen_ind)] = 'Failure'
    # postprocess results for documentation
    for index in doc_results.index:
        doc_results.at[index, 'scenarios'] = {scen_lst_ind: scenario_lst[scen_lst_ind] for scen_lst_ind in
                                              range(0, len(scenario_lst))}
        metric_lst = rmse_dict[index]
        ratio_lst = rmse_ratio_dict[index]
        doc_results.at[index, 'rmse_mean'] = np.mean(metric_lst)
        doc_results.at[index, 'rmse_std'] = np.std(metric_lst)
        doc_results.at[index, 'rmse_ratio_mean'] = np.mean(ratio_lst)
        doc_results.at[index, 'rmse_ratio_std'] = np.std(ratio_lst)
        doc_results.at[index, 'rmse_ratio_median'] = np.median(ratio_lst)
        doc_results.at[index, 'rmse_base_mean'] = np.mean(rmse_base_lst)
        doc_results.at[index, 'rmse_base_std'] = np.std(rmse_base_lst)
        doc_results.at[index, 'n_cps_detected'] = n_cps_detected_dict[index]
        doc_results.at[index, 'n_refits'] = n_refits_dict[index]
    # save results to OptimResults folder
    doc_results.to_csv(
        'OptimResults/' + da + '_' + cpd + '_SimulatedScenarios_SeasLen'
        + str(seas_len) + '.csv', sep=';', decimal=',', float_format='%.10f')
