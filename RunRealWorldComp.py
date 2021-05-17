import warnings
import numpy as np
import copy
import pandas as pd
import random
import configparser
import argparse
# import time

from training import ModelsGaussianProcessRegression, EvarsGpr, TrainHelper
from evaluation import EvaluationHelper


if __name__ == '__main__':
    warnings.simplefilter("ignore")
    np.random.seed(0)
    random.seed(0)
    doc_results = None
    parser = argparse.ArgumentParser()

    ### General Params ###
    # Pipeline
    parser.add_argument("-scalethr", "--scale_threshold", type=float, default=0.1,
                        help="specify output scale threshold")
    parser.add_argument("-scaleseas", "--scale_seasons", type=int, default=2,
                        help="specify output scale seasons taken into account")
    parser.add_argument("-scalew-factor", "--scale_window_factor", type=float, default=0.1,
                        help="specify scale window factor based on seasonal periods")
    parser.add_argument("-scalew-min", "--scale_window_minimum", type=int, default=2,
                        help="specify scale window minimum")
    parser.add_argument("-max-samples", "--max_samples", type=int, default=None,
                        help="specify max samples for gpr pipeline")
    parser.add_argument("-max-samples-fact", "--max_samples_factor", type=int, default=10,
                        help="specify max samples factor of seasons to keep for gpr pipeline")
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
    scale_thr = args.scale_threshold
    scale_window_factor = args.scale_window_factor
    scale_window_minimum = args.scale_window_minimum
    scale_seasons = args.scale_seasons
    max_samples_user = args.max_samples
    max_samples_factor = args.max_samples_factor

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
    const_hazard_user = args.constant_hazard
    cf_r = args.changefinder_r
    cf_order = args.changefinder_order
    cf_smooth = args.changefinder_smooth
    cf_thr_perc = args.changefinder_threshold_percentile

    realworld_results_overview = None
    # Get and Train CV config
    optim_results_dir = 'OptimResults/'
    # Dictionary with all offline optimization result files
    result_file_dict = {
        'PotTotal': 'CV-General-gp-sklearn_raw-PotTotal-CashierData_weekly-CashierData_before_break'
                    '-feat_full-imp_mean-splitperc_08-SeasPer_52-19-Apr-2021_15-25.csv'
    }

    i = 1
    for target_column in result_file_dict.keys():
        print('++++++ Processing Dataset ' + str(i) + '/' + str(len(result_file_dict.keys())) + ' ++++++')
        i += 1
        # set standard values
        split_perc = 0.8
        company = 'General'
        doc_results = None
        result_file_str = result_file_dict[target_column]

        # read config file
        config = configparser.ConfigParser()
        config.read('Configs/dataset_specific_config.ini')
        # get optim parameters
        base_dir, seasonal_periods, split_perc, init_train_len, test_len, resample_weekly = \
            TrainHelper.get_optimization_run_parameters(config=config, company=company, target_column=target_column,
                                                        split_perc=split_perc)
        # set const hazard and scale window based on seasonal periods
        const_hazard = const_hazard_factor * seasonal_periods if const_hazard_user == 9999 else const_hazard_user
        scale_window = max(scale_window_minimum, int(scale_window_factor * seasonal_periods))
        max_samples = max_samples_factor * seasonal_periods if max_samples_factor is not None else max_samples_user
        # read result file config
        result_file = pd.read_csv(optim_results_dir + result_file_str, sep=';', decimal=',', index_col=False)
        result_file.drop('Unnamed: 0', axis=1, inplace=True)
        result_file.replace(to_replace='NaN', value=np.nan, inplace=True)
        result_file.drop(result_file.index[result_file['shuf_cv_rmse_std'].isna()], inplace=True)
        result_file.dropna(subset=[el for el in result_file.columns if 'cv' in el], inplace=True)
        result_file.drop(result_file.index[result_file['shuf_cv_rmse_std'] == 0], inplace=True)
        sort_col = 'shuf_cv_rmse_mean'
        sorted_results = result_file.sort_values(sort_col)
        top_config = sorted_results.head(1).iloc[0]
        dict_top_config = TrainHelper.read_config_info(top_config, seasonal_periods)

        # load datasets
        datasets = TrainHelper.load_datasets(config=config, company=company, target_column=target_column)
        dataset = datasets[0]
        dataset_name = dataset.name

        # train offline cv model
        train_test_list = TrainHelper.get_ready_train_test_lst(dataset=dataset, config=config,
                                                               init_train_len=None,
                                                               test_len=None, split_perc=split_perc,
                                                               imputation=dict_top_config['imputation'],
                                                               target_column=target_column,
                                                               dimensionality_reduction=
                                                               dict_top_config['dim_reduction'],
                                                               featureset=dict_top_config['featureset'])
        train = train_test_list[0][0]
        test = train_test_list[0][1]
        dataset = train.append(test)
        print('#######################' + target_column + '#######################')
        print(args)
        print('---------- Retrain CV ----------')
        model = ModelsGaussianProcessRegression.GaussianProcessRegression(
            target_column=target_column, seasonal_periods=seasonal_periods, kernel=dict_top_config['kernel'],
            alpha=dict_top_config['alpha'], n_restarts_optimizer=dict_top_config['n_restarts_optimizer'],
            standardize=dict_top_config['standardize'], normalize_y=dict_top_config['normalize_y'],
            one_step_ahead=False)
        cross_val_dict = model.train(train=train, cross_val_call=True)
        # start_proc = time.process_time_ns() / 1000000
        # start_time = time.time_ns() / 1000000
        eval_dict = model.evaluate(train=train, test=test)
        # end_proc = time.process_time_ns() / 1000000
        # end_time = time.time_ns() / 1000000
        # runtime_ms_dict = {'CV_Runtime_ms': end_proc - start_proc,
        #                    'CV_Time_ms': end_time - start_time}
        base_model = copy.deepcopy(model)
        print('RMSE_CV=' + str(eval_dict['RMSE_Test']))
        predictions_cv = model.predict(test=test, train=train)

        print('---------- EVARS-GPR ----------')
        # start_proc = time.process_time_ns() / 1000000
        # start_time = time.time_ns() / 1000000
        cp_detected, predictions_full, comparison_partners_dict, _ = \
            EvarsGpr.run_evars_gpr(base_model=base_model, data=dataset, season_length=seasonal_periods,
                                   comparison_partners=True,
                                   target_column=target_column, train_ind=int(split_perc*dataset.shape[0]),
                                   scale_thr=scale_thr, da=da, o_perc=o_percentage, u_perc=u_percentage, thr=threshold,
                                   rel_thr=rel_thr, rel_coef=rel_coef, under_samp=under_samp,
                                   append=append, const_hazard=const_hazard, scale_window=scale_window,
                                   scale_seasons=scale_seasons, cpd=cpd, cf_r=cf_r, cf_order=cf_order,
                                   cf_smooth=cf_smooth, cf_thr_perc=cf_thr_perc, max_samples=max_samples)
        # end_proc = time.process_time_ns() / 1000000
        # end_time = time.time_ns() / 1000000
        # runtime_ms_dict['Full_pipeline_Runtime_ms'] = end_proc - start_proc
        # runtime_ms_dict['Full_pipeline_Time_ms'] = end_time - start_time
        actual = test[target_column].copy()
        actual.reset_index(drop=True, inplace=True)
        pred_evars = predictions_full['Prediction'].copy()
        pred_evars.reset_index(drop=True, inplace=True)
        rmse_evars = EvaluationHelper.rmse(actual=actual, prediction=pred_evars)
        print('RMSE_EVARS-GPR=' + str(rmse_evars))

        print('---------- PR 1 ----------')
        model_period_retr_1 = ModelsGaussianProcessRegression.GaussianProcessRegression(
            target_column=target_column, seasonal_periods=seasonal_periods, kernel=dict_top_config['kernel'],
            alpha=dict_top_config['alpha'], n_restarts_optimizer=dict_top_config['n_restarts_optimizer'],
            standardize=dict_top_config['standardize'], normalize_y=dict_top_config['normalize_y'],
            one_step_ahead=1)
        cross_val_dict = model_period_retr_1.train(train=train, cross_val_call=True)
        # start_proc = time.process_time_ns() / 1000000
        # start_time = time.time_ns() / 1000000
        eval_dict = model_period_retr_1.evaluate(train=train, test=test)
        # end_proc = time.process_time_ns() / 1000000
        # end_time = time.time_ns() / 1000000
        # runtime_ms_dict['PR1_Runtime_ms'] = end_proc - start_proc
        # runtime_ms_dict['PR1_Time_ms'] = end_time - start_time
        print('RMSE_PR1=' + str(eval_dict['RMSE_Test']))
        predictions_period_retr_1 = model_period_retr_1.predict(test=test, train=train)

        print('---------- PR 2 ----------')
        model_period_retr_2 = ModelsGaussianProcessRegression.GaussianProcessRegression(
            target_column=target_column, seasonal_periods=seasonal_periods, kernel=dict_top_config['kernel'],
            alpha=dict_top_config['alpha'], n_restarts_optimizer=dict_top_config['n_restarts_optimizer'],
            standardize=dict_top_config['standardize'], normalize_y=dict_top_config['normalize_y'],
            one_step_ahead=2)
        cross_val_dict = model_period_retr_2.train(train=train, cross_val_call=True)
        # start_proc = time.process_time_ns() / 1000000
        # start_time = time.time_ns() / 1000000
        eval_dict = model_period_retr_2.evaluate(train=train, test=test)
        # end_proc = time.process_time_ns() / 1000000
        # end_time = time.time_ns() / 1000000
        # runtime_ms_dict['PR2_Runtime_ms'] = end_proc - start_proc
        # runtime_ms_dict['PR2_Time_ms'] = end_time - start_time
        print('RMSE_PR2=' + str(eval_dict['RMSE_Test']))
        predictions_period_retr_2 = model_period_retr_2.predict(test=test, train=train)

        print('---------- MWGPR ----------')
        model_mwgpr = ModelsGaussianProcessRegression.GaussianProcessRegression(
            target_column=target_column, seasonal_periods=seasonal_periods, kernel=dict_top_config['kernel'],
            alpha=dict_top_config['alpha'], n_restarts_optimizer=dict_top_config['n_restarts_optimizer'],
            standardize=dict_top_config['standardize'], normalize_y=dict_top_config['normalize_y'],
            one_step_ahead='mw')
        cross_val_dict = model_mwgpr.train(train=train, cross_val_call=True)
        # start_proc = time.process_time_ns() / 1000000
        # start_time = time.time_ns() / 1000000
        eval_dict = model_mwgpr.evaluate(train=train, test=test)
        # end_proc = time.process_time_ns() / 1000000
        # end_time = time.time_ns() / 1000000
        # runtime_ms_dict['MWGPR_Runtime_ms'] = end_proc - start_proc
        # runtime_ms_dict['MWGPR_Time_ms'] = end_time - start_time
        print('RMSE_MWGPR=' + str(eval_dict['RMSE_Test']))
        predictions_mwgpr = model_mwgpr.predict(test=test, train=train)

        eval_pipeline_dict = {}
        print('##### Comparison Results #####')
        comparison_partners_dict['CV'] = predictions_cv
        comparison_partners_dict['EVARS-GPR'] = predictions_full
        comparison_partners_dict['PR1'] = predictions_period_retr_1
        comparison_partners_dict['PR2'] = predictions_period_retr_2
        comparison_partners_dict['MWGPR'] = predictions_mwgpr
        for key in comparison_partners_dict.keys():
            pred = comparison_partners_dict[key]['Prediction'].copy()
            pred.reset_index(drop=True, inplace=True)
            try:
                rmse = EvaluationHelper.rmse(actual=actual, prediction=pred)
                eval_pipeline_dict['RMSE_' + key] = rmse
            except NameError:
                print('Found NaNs')
                eval_pipeline_dict['RMSE_' + key] = np.nan
            print(key + ': RMSE=' + str(rmse))
        params_dict = {'dataset': dataset_name, 'target_column': target_column, 'da': da, 'cpd': cpd,
                       'append': append, 'scale_thr': scale_thr, 'scale_window_factor': scale_window_factor,
                       'max_samples':  max_samples}
        save_str = 'app-' + append + '_scthr-' + str(scale_thr) + '_scw-' + str(scale_window)
        if da == 'smogn':
            params_dict['rel_thr'] = rel_thr
            params_dict['rel_coef'] = rel_coef
            params_dict['under_samp'] = under_samp
            save_str = save_str + '_relt-' + str(rel_thr) + '_relc-' + str(rel_coef) + '_us-' + str(under_samp)
        elif da == 'gn':
            params_dict['o_perc'] = o_percentage
            params_dict['u_perc'] = u_percentage
            params_dict['threshold'] = threshold
            save_str = save_str + '_op-' + str(o_percentage) + '_up-' + str(u_percentage) + '_thr-' + str(threshold)
        if cpd == 'bocd':
            params_dict['const_hazard'] = const_hazard
            save_str = save_str + '_chaz-' + str(const_hazard)
        elif cpd == 'cf':
            params_dict['cf_r'] = cf_r
            params_dict['cf_order'] = cf_order
            params_dict['cf_smooth'] = cf_smooth
            params_dict['cf_thr_perc'] = cf_thr_perc
            save_str = save_str + '_cfr-' + str(cf_r) + '_cforder-' + str(cf_order) + '_cfsmooth-' + str(cf_smooth) \
                       + '_cfthr-' + str(cf_thr_perc)

        save_dict = params_dict.copy()
        save_dict.update(dict_top_config)
        save_dict.update(eval_pipeline_dict)
        save_dict.update(comparison_partners_dict)
        # save_dict.update(runtime_ms_dict)
        if doc_results is None:
            doc_results = pd.DataFrame(columns=save_dict.keys())
        doc_results = doc_results.append(save_dict, ignore_index=True)

        doc_results.to_csv(
            optim_results_dir + 'RealWorldComp_' + target_column + '_' + cpd + '_' + da + '_' + save_str + '.csv',
            sep=';', decimal=',', float_format='%.10f')
        if realworld_results_overview is None:
            realworld_results_overview = pd.DataFrame(columns=save_dict.keys())
        realworld_results_overview = realworld_results_overview.append(save_dict, ignore_index=True)

    realworld_results_overview.to_csv(
        optim_results_dir + 'RealWorldResultsOverview.csv', sep=';', decimal=',', float_format='%.10f')
