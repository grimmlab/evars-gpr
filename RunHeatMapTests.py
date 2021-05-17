import warnings
import numpy as np
import math
import pandas as pd
from sklearn.gaussian_process.kernels import ExpSineSquared
import random
import argparse

from training import ModelsGaussianProcessRegression, EvarsGpr, TrainHelper
from evaluation import EvaluationHelper


if __name__ == '__main__':
    """
    Script to run heatmap tests based on different scenarios of manipulated simulated data.
    All parameters of EVARS-GPR and its components can be specified
    """
    warnings.simplefilter("ignore")
    np.random.seed(0)
    random.seed(0)
    doc_results = None
    parser = argparse.ArgumentParser()
    ### General Params ###
    # Data Setup
    parser.add_argument("-seaslen", "--seasonal_length", type=int, default=50, help="specify season length")
    parser.add_argument("-nper", "--n_periods", type=int, default=5, help="specify number of periods")
    parser.add_argument("-noise", "--noise", type=float, default=0.01, help="specify noise for sine x and y")
    parser.add_argument("-offset", "--offset", type=float, default=10, help="specify offset for sine y")
    # Pipeline
    parser.add_argument("-scalethr", "--scale_threshold", type=float, default=0.1,
                        help="specify output scale threshold")
    parser.add_argument("-scaleseas", "--scale_seasons", type=int, default=2,
                        help="specify output scale seasons taken into account")
    parser.add_argument("-scalew-factor", "--scale_window_factor", type=float, default=0.1,
                        help="specify scale window factor based on seasonal periods")
    parser.add_argument("-scalew-min", "--scale_window_minimum", type=int, default=2,
                        help="specify scale window minimum")
    ### DA Params ###
    # General
    parser.add_argument("-da", "--data_augmentation", type=str, default='scaled',
                        help="specify data augmentation")
    parser.add_argument("-app", "--append", type=str, default='no', help="specify append mechanism")
    # GN Params
    parser.add_argument("-operc", "--o_percentage", type=float, default=1.1, help="specify oversampling perentage gn")
    parser.add_argument("-uperc", "--u_percentage", type=float, default=0.8, help="specify undersampling perentage gn")
    parser.add_argument("-thr", "--threshold", type=float, default=0.1, help="specify threshold gn")
    # SMOGN Params
    parser.add_argument("-relthr", "--relevance_threshold", type=float, default=1.5,
                        help="specify relevance threshold smogn")
    parser.add_argument("-relcoef", "--relevance_coefficient", type=float, default=0.5,
                        help="specify relevance coefficient smogn")
    parser.add_argument("-undersamp", "--undersampling", type=bool, default=False, help="specify undersampling smogn")

    ### CPD Params ###
    parser.add_argument("-cpd", "--change_point_detection", type=str, default='cf',
                        help="specify cpd algo: bocd or cf")
    # BOCPD
    parser.add_argument("-chaz", "--constant_hazard", type=int, default=9999, help="specify constant hazard")
    parser.add_argument("-chaz-factor", "--constant_hazard_factor", type=int, default=2,
                        help="specify constant hazard factor based on seasonal periods")
    # CF
    parser.add_argument("-cfr", "--changefinder_r", type=float, default=0.4,
                        help="specify changefinders r param (decay factor older values)")
    parser.add_argument("-cforder", "--changefinder_order", type=int, default=1,
                        help="specify changefinders SDAR model order param")
    parser.add_argument("-cfsmooth", "--changefinder_smooth", type=int, default=4,
                        help="specify changefinders smoothing param")
    parser.add_argument("-cfthrperc", "--changefinder_threshold_percentile", type=int, default=70,
                        help="specify percentile of train set anomaly factors as threshold for cpd with changefinder")

    args = parser.parse_args()

    # General Params
    season_length = args.seasonal_length
    n_periods = args.n_periods
    noise = args.noise
    offset = args.offset
    scale_thr = args.scale_threshold
    scale_window_factor = args.scale_window_factor
    scale_window_minimum = args.scale_window_minimum
    scale_seasons = args.scale_seasons
    scale_window = max(scale_window_minimum, int(scale_window_factor * season_length))

    # DA Params
    da = args.data_augmentation
    append = args.append
    o_percentage = args.o_percentage
    u_percentage = args.u_percentage
    threshold = args.threshold
    rel_thr = args.relevance_threshold
    rel_coef = args.relevance_coefficient
    under_samp = args.undersampling

    # CPD Params
    cpd = args.change_point_detection
    const_hazard_factor = args.constant_hazard_factor
    const_hazard = args.constant_hazard
    const_hazard = const_hazard_factor * season_length if const_hazard == 9999 else const_hazard
    cf_r = args.changefinder_r
    cf_order = args.changefinder_order
    cf_smooth = args.changefinder_smooth
    cf_thr_perc = args.changefinder_threshold_percentile

    # get base data
    X = TrainHelper.get_periodic_noisy_x(x_base=np.linspace(-0.5 * math.pi, 1.5 * math.pi, season_length),
                                         n_periods=n_periods, noise=noise)
    Y = TrainHelper.noisy_sin(X, noise=noise, offset=offset)
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

    columns = ['sampler', 'cpd', 'seas_len', 'start_ind', 'end_ind', 'slope', 'max_factor', 'noise',
               'const_hazard', 'scale_window', 'scale_seasons', 'scale_thr', 'append', 'o_perc', 'u_perc', 'threshold',
               'under_samp', 'rel_thr', 'rel_coef', 'cf_r', 'cf_order', 'cf_smooth', 'cf_thr_perc',
               'rmse', 'rmse_base', 'rmse_ratio']
    # heat map Configs
    step_size = 10 if season_length <= 50 else 20  # adjust step size for longer scenarios to limit amount of scenarios
    heat_map_configs = [
        {'type': 'start vs. end index', 'slope': 0.2, 'max_factor': 2, 'step_size': step_size},
        {'type': 'start vs. end index', 'slope': 0.4, 'max_factor': 3, 'step_size': step_size},
        {'type': 'slope vs. max factor', 'start_ind': train_ind + season_length,
         'end_ind': train_ind + 2 * season_length,
         'start_slope': 0.1, 'end_slope': 1, 'step_size_slope': 0.1,
         'start_max_factor': 0.5, 'end_max_factor': 5, 'step_size_max_factor': 0.5},
        {'type': 'slope vs. max factor', 'start_ind': train_ind + season_length,
         'end_ind': train_ind + 2 * season_length,
         'start_slope': 0.1, 'end_slope': 1, 'step_size_slope': 0.1,
         'start_max_factor': 0.4, 'end_max_factor': 3, 'step_size_max_factor': 0.2}
    ]
    # iterate over all heat map configs
    for config in heat_map_configs:
        doc_results = pd.DataFrame(columns=columns)
        results_index = 0
        heat_map_type = config['type']
        print('####### ' + heat_map_type + ' #######')
        print(args)
        # heat map start vs. end index params
        slope = config['slope'] if 'slope' in config else None
        max_factor = config['max_factor'] if 'max_factor' in config else None
        step_size = config['step_size'] if 'step_size' in config else None
        # heat map slope vs. max factor params
        start_ind = config['start_ind'] if 'start_ind' in config else None
        end_ind = config['end_ind'] if 'end_ind' in config else None
        start_slope = config['start_slope'] if 'start_slope' in config else None
        end_slope = config['end_slope'] if 'end_slope' in config else None
        step_size_slope = config['step_size_slope'] if 'step_size_slope' in config else None
        start_max_factor = config['start_max_factor'] if 'start_max_factor' in config else None
        end_max_factor = config['end_max_factor'] if 'end_max_factor' in config else None
        step_size_max_factor = config['step_size_max_factor'] if 'step_size_max_factor' in config else None
        # setup iterators for heat map
        if heat_map_type == 'start vs. end index':
            min_diff = int((max_factor-1) / slope) * 2
            min_diff = int(math.ceil(min_diff / 10.0)) * 10 if min_diff % 10 != 0 else min_diff
            if season_length > 50:
                min_diff = max(20, min_diff)
            iterator_outer = np.arange(train_ind, len(data), step_size)
            iterator_inner = np.arange(train_ind + min_diff, len(data) + step_size, step_size)
            iterator_outer = iterator_outer[iterator_outer + min_diff <= len(data)]
            iterator_inner = iterator_inner[iterator_inner <= len(data)]
            save_str = \
                heat_map_type + '_slope' + str(slope) + '_maxfact' + str(max_factor) + '_stepsize' + str(step_size)
        elif heat_map_type == 'slope vs. max factor':
            iterator_outer = np.arange(start_slope, end_slope + step_size_slope, step_size_slope)
            iterator_inner = np.arange(start_max_factor, end_max_factor + step_size_max_factor, step_size_max_factor)
            save_str = heat_map_type \
                       + '_slopeconf-' + str(start_slope) + '-' + str(end_slope) + '-' + str(step_size_slope) \
                       + '_maxfactconf-' + str(start_max_factor) + '-' + str(end_max_factor) \
                       + '-' + str(step_size_max_factor)
        # iterate over heat map params
        for it_out in iterator_outer:
            for it_in in iterator_inner:
                if heat_map_type == 'start vs. end index':
                    start_ind = it_out
                    end_ind = it_in
                    if start_ind + min_diff > end_ind:
                        continue
                    print('--- Running Variant: start_ind=' + str(start_ind) + ', end_ind=' + str(end_ind) + ' ---')
                    print('slope=' + str(slope) + ', max_factor=' + str(max_factor))
                elif heat_map_type == 'slope vs. max factor':
                    slope = it_out
                    max_factor = it_in
                    print('--- Running Variant: slope=' + str(slope) + ', max_factor=' + str(max_factor) + ' ---')
                    print('start_ind=' + str(start_ind) + ', end_ind=' + str(end_ind))
                try:
                    # manipulate data
                    data_manip = TrainHelper.get_sloped_dataset(dataset=data, target_column=target_column,
                                                                start_ind=start_ind, end_ind=end_ind,
                                                                max_factor=max_factor, slope=slope)
                    # get base predictions
                    base_predict = model_sine.predict(train=data_manip[:train_ind], test=data_manip[train_ind:])
                    rmse_base = EvaluationHelper.rmse(actual=data_manip[train_ind:][target_column],
                                                      prediction=base_predict['Prediction'])
                    print('Base Prediction: RMSE=' + str(rmse_base))
                    # get evars-gpr results
                    cp_detected, predictions, _, _ = \
                        EvarsGpr.run_evars_gpr(base_model=model_sine, data=data_manip, season_length=season_length,
                                               target_column=target_column, train_ind=train_ind, scale_thr=scale_thr,
                                               da=da, o_perc=o_percentage, u_perc=u_percentage, thr=threshold,
                                               rel_thr=rel_thr, rel_coef=rel_coef, under_samp=under_samp,
                                               append=append, const_hazard=const_hazard, scale_window=scale_window,
                                               verbose=False, scale_seasons=scale_seasons, cpd=cpd, cf_r=cf_r,
                                               cf_order=cf_order, cf_smooth=cf_smooth, cf_thr_perc=cf_thr_perc)
                    rmse = EvaluationHelper.rmse(actual=data_manip[train_ind:][target_column],
                                                 prediction=predictions['Prediction'])
                    # store results
                    print('EVARS-GPR: RMSE=' + str(rmse))
                    rmse_ratio = rmse / rmse_base
                    print('RMSE_Ratio=' + str(rmse_ratio))
                    doc_results.at[results_index, 'sampler'] = da
                    doc_results.at[results_index, 'cpd'] = cpd
                    doc_results.at[results_index, 'cf_r'] = float(cf_r)
                    doc_results.at[results_index, 'cf_order'] = float(cf_order)
                    doc_results.at[results_index, 'cf_smooth'] = float(cf_smooth)
                    doc_results.at[results_index, 'cf_thr_perc'] = float(cf_thr_perc)
                    doc_results.at[results_index, 'seas_len'] = season_length
                    doc_results.at[results_index, 'start_ind'] = start_ind
                    doc_results.at[results_index, 'end_ind'] = end_ind
                    doc_results.at[results_index, 'slope'] = slope
                    doc_results.at[results_index, 'max_factor'] = max_factor
                    doc_results.at[results_index, 'noise'] = noise
                    doc_results.at[results_index, 'o_perc'] = float(o_percentage)
                    doc_results.at[results_index, 'u_perc'] = float(u_percentage)
                    doc_results.at[results_index, 'threshold'] = float(threshold)
                    doc_results.at[results_index, 'under_samp'] = under_samp
                    doc_results.at[results_index, 'rel_thr'] = float(rel_thr)
                    doc_results.at[results_index, 'rel_coef'] = float(rel_coef)
                    doc_results.at[results_index, 'append'] = append
                    doc_results.at[results_index, 'const_hazard'] = float(const_hazard)
                    doc_results.at[results_index, 'scale_window'] = scale_window
                    doc_results.at[results_index, 'scale_seasons'] = scale_seasons
                    doc_results.at[results_index, 'scale_thr'] = scale_thr
                    doc_results.at[results_index, 'rmse'] = float(rmse)
                    doc_results.at[results_index, 'rmse_base'] = float(rmse_base)
                    doc_results.at[results_index, 'rmse_ratio'] = float(rmse_ratio)
                    results_index += 1
                except Exception as exc:
                    print(exc)
                    doc_results['rmse'] = 'Failure'
                    results_index += 1
            # save results to disk
            doc_results.to_csv(
                'OptimResults/HeatMap_'
                + save_str + '_' + cpd + '_' + da + '_seaslen' + str(season_length) + '_offset' + str(offset) + '.csv',
                sep=';', decimal=',', float_format='%.10f')
