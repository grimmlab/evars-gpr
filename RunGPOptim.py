import configparser
import os
import sys
import warnings
import argparse
import itertools
import pandas as pd
from sklearn.gaussian_process.kernels import RBF, Matern, RationalQuadratic, ExpSineSquared, ConstantKernel, WhiteKernel
from tqdm import tqdm

from training import TrainHelper, ModelsGaussianProcessRegression


def run_gp_optim(company: str, target_column: str, split_perc: float, imputation: str, featureset: str):
    """
    Run GPR offline optimization loop
    :param company: prefix for data in case company data is also used
    :param target_column: target column to use
    :param split_perc: share of train data
    :param imputation: imputation method
    :param featureset: featureset to use
    """
    config = configparser.ConfigParser()
    config.read('Configs/dataset_specific_config.ini')
    # get optim parameters
    base_dir, seasonal_periods, split_perc, init_train_len, test_len, resample_weekly = \
        TrainHelper.get_optimization_run_parameters(config=config, company=company, target_column=target_column,
                                                    split_perc=split_perc)

    # load datasets
    datasets = TrainHelper.load_datasets(config=config, company=company, target_column=target_column)

    # prepare parameter grid
    kernels = []
    base_kernels = [ConstantKernel(constant_value=1000, constant_value_bounds=(1e-5, 1e5)),
                    Matern(length_scale=1.0, length_scale_bounds=(1e-5, 1e5)),
                    ExpSineSquared(length_scale=1.0, periodicity=seasonal_periods,
                                   length_scale_bounds=(1e-5, 1e5),
                                   periodicity_bounds=(int(seasonal_periods * 0.8), int(seasonal_periods*1.2))),
                    RBF(length_scale=1.0, length_scale_bounds=(1e-5, 1e5)),
                    RationalQuadratic(length_scale=1.0, alpha=1.0,
                                      length_scale_bounds=(1e-5, 1e5), alpha_bounds=(1e-5, 1e5)),
                    WhiteKernel(noise_level=1.0, noise_level_bounds=(1e-5, 1e5))]
    TrainHelper.extend_kernel_combinations(kernels=kernels, base_kernels=base_kernels)
    param_grid = {'dataset': [datasets[0]],
                  'imputation': [imputation],
                  'featureset': [featureset],
                  'dim_reduction': ['None', 'pca'],
                  'kernel': kernels,
                  'alpha': [1e-5, 1e-3, 1e-1, 1, 1e1, 1e3],
                  'n_restarts_optimizer': [0, 5, 10],
                  'standardize': [False, True],
                  'norm_y': [False, True],
                  'osa': [False]
                  }
    # random sample from parameter grid
    sample_share = 0.1
    params_lst = TrainHelper.random_sample_parameter_grid(param_grid=param_grid, sample_share=sample_share)

    doc_results = None
    best_rmse = 5000000.0
    dataset_last_name = 'Dummy'
    imputation_last = 'Dummy'
    dim_reduction_last = 'Dummy'
    featureset_last = 'Dummy'

    for i in tqdm(range(len(params_lst))):
        warnings.simplefilter('ignore')
        dataset = params_lst[i]['dataset']
        imputation = params_lst[i]['imputation']
        featureset = params_lst[i]['featureset']
        dim_reduction = None if params_lst[i]['dim_reduction'] == 'None' else params_lst[i]['dim_reduction']
        kernel = params_lst[i]['kernel']
        alpha = params_lst[i]['alpha']
        n_restarts_optimizer = params_lst[i]['n_restarts_optimizer']
        stand = params_lst[i]['standardize']
        norm_y = params_lst[i]['norm_y']
        one_step_ahead = params_lst[i]['osa']

        # dim_reduction can only be done without NaNs
        if imputation is None and dim_reduction is not None:
            continue
        # 'dim_reduction does not make sense for few features
        if featureset == 'none' and dim_reduction is not None:
            continue

        if not((dataset.name == dataset_last_name) and (imputation == imputation_last) and
               (dim_reduction == dim_reduction_last) and (featureset == featureset_last)):
            if resample_weekly and 'weekly' not in dataset.name:
                dataset.name = dataset.name + '_weekly'
            print(dataset.name + ' ' + str('None' if imputation is None else imputation) + ' '
                  + str('None' if dim_reduction is None else dim_reduction) + ' '
                  + featureset + ' ' + target_column)
            train_test_list = TrainHelper.get_ready_train_test_lst(dataset=dataset, config=config,
                                                                   init_train_len=init_train_len,
                                                                   test_len=test_len, split_perc=split_perc,
                                                                   imputation=imputation,
                                                                   target_column=target_column,
                                                                   dimensionality_reduction=dim_reduction,
                                                                   featureset=featureset)
            if dataset.name != dataset_last_name:
                best_rmse = 5000000.0
            dataset_last_name = dataset.name
            imputation_last = imputation
            dim_reduction_last = dim_reduction
            featureset_last = featureset

        sum_dict = None
        try:
            for train, test in train_test_list:
                model = ModelsGaussianProcessRegression.GaussianProcessRegression(target_column=target_column,
                                                                                  seasonal_periods=seasonal_periods,
                                                                                  kernel=kernel,
                                                                                  alpha=alpha,
                                                                                  n_restarts_optimizer=
                                                                                  n_restarts_optimizer,
                                                                                  one_step_ahead=one_step_ahead,
                                                                                  standardize=stand,
                                                                                  normalize_y=norm_y)
                cross_val_dict = model.train(train=train, cross_val_call=True)
                eval_dict = model.evaluate(train=train, test=test)
                eval_dict.update(cross_val_dict)
                if sum_dict is None:
                    sum_dict = eval_dict
                else:
                    for k, v in eval_dict.items():
                        sum_dict[k] += v
            evaluation_dict = {k: v / len(train_test_list) for k, v in sum_dict.items()}
            params_dict = {'dataset': dataset.name, 'featureset': featureset,
                           'imputation': str('None' if imputation is None else imputation),
                           'dim_reduction': str('None' if dim_reduction is None else dim_reduction),
                           'init_train_len': init_train_len, 'test_len': test_len, 'split_perc': split_perc,
                           'kernel': kernel, 'alpha': alpha, 'n_restarts_optimizer': n_restarts_optimizer,
                           'standardize': stand, 'normalize_y': norm_y, 'one_step_ahead': one_step_ahead,
                           'optimized_kernel': model.model.kernel_}
            save_dict = params_dict.copy()
            save_dict.update(evaluation_dict)
            if doc_results is None:
                doc_results = pd.DataFrame(columns=save_dict.keys())
            doc_results = doc_results.append(save_dict, ignore_index=True)
            best_rmse = TrainHelper.print_best_vals(evaluation_dict=evaluation_dict, best_rmse=best_rmse, run_number=i)
        except KeyboardInterrupt:
            print('Got interrupted')
            break
        except Exception as exc:
            print(exc)
            params_dict = {'dataset': 'Failure', 'featureset': featureset,
                           'imputation': str('None' if imputation is None else imputation),
                           'dim_reduction': str('None' if dim_reduction is None else dim_reduction),
                           'init_train_len': init_train_len, 'test_len': test_len, 'split_perc': split_perc,
                           'kernel': kernel, 'alpha': alpha, 'n_restarts_optimizer': n_restarts_optimizer,
                           'standardize': stand, 'normalize_y': norm_y, 'one_step_ahead': one_step_ahead,
                           'optimized_kernel': 'failed'}
            save_dict = params_dict.copy()
            save_dict.update(TrainHelper.get_failure_eval_dict())
            if doc_results is None:
                doc_results = pd.DataFrame(columns=save_dict.keys())
            doc_results = doc_results.append(save_dict, ignore_index=True)
    TrainHelper.save_csv_results(doc_results=doc_results,
                                 save_dir=base_dir+'OptimResults/',
                                 company_model_desc=company+'-gp-sklearn_raw', target_column=target_column,
                                 seasonal_periods=seasonal_periods, datasets=datasets,
                                 featuresets=param_grid['featureset'], imputations=param_grid['imputation'],
                                 split_perc=split_perc)
    print('Optimization Done. Saved Results.')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("-company", "--company_name", type=str, default='General', help="specify company name")
    parser.add_argument("-tc", "--target_column", type=str, default='PotTotal', help="specify target column")
    parser.add_argument("-splitperc", "--split_percentage", type=float, default=0.8,
                        help="specify share of train set")
    args = parser.parse_args()
    company = args.company_name
    target_column = args.target_column
    split_perc = args.split_percentage
    imputations = ['mean']
    featuresets = ['full']
    imp_feat_combis = list(itertools.product(*[imputations, featuresets]))
    for (imputation, featureset) in imp_feat_combis:
        new_pid = os.fork()
        if new_pid == 0:
            run_gp_optim(company=company, target_column=target_column, split_perc=split_perc,
                         imputation=imputation, featureset=featureset)
            sys.exit()
        else:
            os.waitpid(new_pid, 0)
            print('finished run with ' + featureset + ' ' + str('None' if imputation is None else imputation))
