"""
home-credit-default-competition repository

Gradient boosting models using the LightGBM, Sklearn
and HyperOpt libraries.
"""

import csv
import os
import gc
import time
import numpy as np
import pandas as pd
import pickle
import lightgbm as lgb
from lightgbm import LGBMClassifier
from sklearn.model_selection import KFold, StratifiedKFold
from sklearn.metrics import roc_auc_score
from hyperopt import hp, tpe, Trials, fmin, STATUS_OK
import config


def kfold_lightgbm_sklearn(data, categorical_feature=None):
    """LightGBM model using Sklearn KFold for cross-validation.

    Arguments:
        path: Path to the folder where files are saved (string).
        num_rows: Number of rows to read; None reads all rows (int, default: None).

    Returns:
        df: DataFrame with processed data.
    """
    df = data[data['TARGET'].notnull()]
    test = data[data['TARGET'].isnull()]
    del_features = ['TARGET', 'SK_ID_CURR', 'SK_ID_BUREAU', 'SK_ID_PREV', 'index', 'level_0']
    predictors = [feat for feat in df.columns if feat not in del_features]
    print("Train shape: {}, test shape: {}".format(df[predictors].shape, test[predictors].shape))


    if not config.STRATIFIED_KFOLD:
        folds = KFold(n_splits=config.NUM_FOLDS, shuffle=True, random_state=config.RANDOM_SEED)
    else:
        folds = StratifiedKFold(n_splits=config.NUM_FOLDS, shuffle=True, random_state=config.RANDOM_SEED)

    oof_preds = np.zeros(df.shape[0])
    sub_preds = np.zeros(test.shape[0])
    importance_df = pd.DataFrame()
    auc_df = dict()

    for n_fold, (train_idx, valid_idx) in enumerate(folds.split(df[predictors], df['TARGET'])):
        train_x, train_y = df[predictors].iloc[train_idx], df['TARGET'].iloc[train_idx]
        valid_x, valid_y = df[predictors].iloc[valid_idx], df['TARGET'].iloc[valid_idx]

        params = {'random_state': config.RANDOM_SEED, 'nthread': config.NUM_THREADS}
        clf = LGBMClassifier(**{**params, **config.LIGHTGBM_PARAMS})
        if not categorical_feature:
            clf.fit(train_x, train_y, eval_set=[(train_x, train_y), (valid_x, valid_y)],
                    eval_metric='auc', verbose=200, early_stopping_rounds=config.EARLY_STOPPING)
        else:
            clf.fit(train_x, train_y, eval_set=[(train_x, train_y), (valid_x, valid_y)],
                    eval_metric='auc', verbose=200, early_stopping_rounds=config.EARLY_STOPPING,
                    feature_name=predictors, categorical_feature=categorical_feature)

        best_iter = clf.best_iteration_
        oof_preds[valid_idx] = clf.predict_proba(valid_x, num_iteration=best_iter)[:, 1]
        sub_preds += clf.predict_proba(test[predictors], num_iteration=best_iter)[:, 1] / folds.n_splits

        # Feature importance by GAIN and SPLIT
        fold_importance = pd.DataFrame()
        fold_importance["feature"] = predictors
        fold_importance["gain"] = clf.booster_.feature_importance(importance_type='gain')
        fold_importance["split"] = clf.booster_.feature_importance(importance_type='split')
        importance_df = pd.concat([importance_df, fold_importance], axis=0)
        # Save metric value for each iteration in train and validation sets
        auc_df['train_{}'.format(n_fold+1)]  = clf.evals_result_['training']['auc']
        auc_df['valid_{}'.format(n_fold + 1)] = clf.evals_result_['valid_1']['auc']

        print('Fold %2d AUC : %.6f' % (n_fold + 1, roc_auc_score(valid_y, oof_preds[valid_idx])))
        del clf, train_x, train_y, valid_x, valid_y
        gc.collect()

    print('Full AUC score %.6f' % roc_auc_score(df['TARGET'], oof_preds))
    test['TARGET'] = sub_preds.copy()

    # Get the average feature importance between folds
    mean_importance = importance_df.groupby('feature').mean().reset_index()
    mean_importance.sort_values(by='gain', ascending=False, inplace=True)
    # Save feature importance, test predictions and oof predictions as csv
    if config.GENERATE_SUBMISSION_FILES:
        # Generate oof csv
        oof = pd.DataFrame()
        oof['SK_ID_CURR'] = df['SK_ID_CURR'].copy()
        oof['PREDICTIONS'] = oof_preds.copy()
        oof['TARGET'] = df['TARGET'].copy()
        file_name = 'oof{}.csv'.format(config.SUBMISSION_SUFIX)
        oof.to_csv(os.path.join(config.SUBMISSION_DIRECTORY, file_name), index=False)

        # Submission and feature importance csv
        sub_path = os.path.join(config.SUBMISSION_DIRECTORY,'submission{}.csv'.format(config.SUBMISSION_SUFIX))
        test[['SK_ID_CURR', 'TARGET']].to_csv(sub_path, index=False)
        imp_path = os.path.join(config.SUBMISSION_DIRECTORY,'feature_importance{}.csv'.format(config.SUBMISSION_SUFIX))
        mean_importance.to_csv(imp_path, index=False)
    return mean_importance


class HyperOpt():
    """Use bayesian optimization to search hyperparameters for model.

    The hyperopt library is used to search the grid_params distributions
    for the best hyperparameters for the LightGBM Model. Each iteration
    is saved in a csv file with the score, runtime and hyperparameters.
    The score is obtained with a KFold cross-validation method. There are
    two differenct functions that can be used evaluate_cv and evaluate_sklearn,
    the first uses lightgbm internal cross-validation with the .cv function,
    while the second uses KFold Sklearn.

    Attributes:
        num_folds: Number of folds for KFold cross-validation (int).
        stratified: Use stratified KFold or not (boolean).
        seed: Random seed (int).
        early_stopping: Check LightGBM API documentation (int).
        fixed_params: Dictionary with fixed hyperparameters.
    """

    def __init__(self):
        self._iter = 0
        self.num_folds = config.NUM_FOLDS
        self.stratified = config.STRATIFIED_KFOLD
        self.seed = config.RANDOM_SEED
        self.early_stopping = config.EARLY_STOPPING
        self.grid_params_list = []
        self.fixed_params = {
            'objective': 'binary',
            'nthread': config.NUM_THREADS,
            'verbosity': -1,
        }

    def fit(self, data, file_name, num_iter=50, kfold_type='sklearn'):
        """Run the hyperparameter search.

        Arguments:
        data: pandas DataFrame with train set.
        file_name: Name of the file to save results (string).
        num_iter: Number of runs (int, default: 50).
        kfold_type: 'sklearn' or 'lgbcv' (string).
        """

        self.fn = file_name
        self.train_df = data[data['TARGET'].notnull()]

        # Remove target, index and id columns from predictive features.
        del_features = ['TARGET', 'SK_ID_CURR', 'SK_ID_BUREAU', 'SK_ID_PREV', 'index', 'level_0']
        self.predictors = [feat for feat in self.train_df.columns if feat not in del_features]

        # Parameters to be tested
        grid_params = {
            'boosting_type': hp.choice('boosting_type',
                                       [{'boosting_type': 'gbdt',
                                         'subsample': hp.uniform('gdbt_subsample', 0.6, 1)},
                                        {'boosting_type': 'goss', 'subsample': 1.0}]),

            #'boosting_type': hp.choice('boosting_type', ['gbdt', 'dart', 'goss']),
            'num_leaves': hp.quniform('num_leaves', 36, 86, 1),
            'max_depth': hp.quniform('max_depth', 4, 14, 1),
            'learning_rate': hp.loguniform('learning_rate', np.log(0.004), np.log(0.2)),
            'subsample_for_bin': hp.quniform('subsample_for_bin', 120000, 300000, 20000),
            'min_child_samples': hp.quniform('min_child_samples', 80, 400, 10),
            'reg_alpha': hp.uniform('reg_alpha', 0.0, 1.0),
            'reg_lambda': hp.uniform('reg_lambda', 0.0, 1.0),
            'colsample_bytree': hp.uniform('colsample_by_tree', 0.5, 1.0),
            'is_unbalance': hp.choice('is_unbalance', [True, False]),
            'min_split_gain': hp.uniform('min_split_gain', 0.01, 0.05),
            #'min_child_weight': hp.quniform('min_child_weight', 10, 50, 1),
            #'subsample': hp.uniform('subsample', 0.6, 1.0)
        }

        self.grid_params_list = [
            'boosting_type', 'num_leaves', 'max_depth', 'learning_rate',
            'subsample_for_bin', 'min_child_samples', 'reg_alpha', 'reg_lambda',
            'colsample_bytree', 'is_unbalance', 'min_split_gain', 'subsample'
        ]
        print("HyperOpt for LGBM - Hyperparameters: {}, iterations: {}".format(len(grid_params), num_iter))
        if kfold_type == 'lgbcv':
            self.train_set = lgb.Dataset(data=self.train_df[self.predictors],
                                         label=self.train_df['TARGET'], silent=False)
        trials = Trials()
        if kfold_type == 'lgbcv':
            _ = fmin(fn=self._evaluate_cv, space=grid_params, trials=trials,
                     algo=tpe.suggest, max_evals=num_iter)
        else:
            _ = fmin(fn=self._evaluate_sklearn, space=grid_params, trials=trials,
                     algo=tpe.suggest, max_evals=num_iter)

    def _evaluate_cv(self, hyperparameters):
        """ Objective function. Returns the cross validation score from a set of hyperparameters. """
        t0 = time.time()

        # Make sure int parameters are integers
        for param in ['max_depth', 'num_leaves', 'subsample_for_bin', 'min_child_samples']:
            if param in hyperparameters:
                hyperparameters[param] = int(hyperparameters[param])

        params = {**hyperparameters, **self.fixed_params}

        # Perform n_folds cross validation
        cv_results = lgb.cv(params, self.train_set, num_boost_round=10000, nfold=self.num_folds,
                            early_stopping_rounds=self.early_stopping, metrics='auc',
                            stratified=self.stratified, seed=self.seed)

        self._iter += 1
        result = cv_results['auc-mean'][-1]
        t0 = int(time.time() - t0)
        print("{}. CV AUC Score: {:.4f} - model run time: {}s".format(self._iter, result, t0))
        self._write_to_csv(hyperparameters, result, t0)
        return {'loss': 1 - result, 'hyperparameters': hyperparameters,
                'iteration': self._iter, 'train_time': t0, 'status': STATUS_OK}

    def _evaluate_sklearn(self, hyperparameters):
        """ Objective function. Returns the cross validation score from a set of hyperparameters. """
        t0 = time.time()

        # Make sure int parameters are integers
        for param in ['max_depth', 'num_leaves', 'subsample_for_bin', 'min_child_samples']:
            if param in hyperparameters:
                hyperparameters[param] = int(hyperparameters[param])

        params = {**hyperparameters, **self.fixed_params}

        # Perform n_folds cross validation
        if not self.stratified:
            folds = KFold(n_splits=self.num_folds, shuffle=True, random_state=self.seed)
        else:
            folds = StratifiedKFold(n_splits=self.num_folds, shuffle=True, random_state=self.seed)

        oof_preds = np.zeros(self.train_df.shape[0])

        for (train_idx, valid_idx) in folds.split(self.train_df[self.predictors], self.train_df['TARGET']):
            train_x = self.train_df[self.predictors].iloc[train_idx]
            train_y = self.train_df['TARGET'].iloc[train_idx]
            valid_x = self.train_df[self.predictors].iloc[valid_idx]
            valid_y = self.train_df['TARGET'].iloc[valid_idx]

            clf = LGBMClassifier(
                nthread= self.fixed_params['nthread'],
                n_estimators=10000,
                boosting_type= params['boosting_type']['boosting_type'],
                learning_rate= params['learning_rate'],
                num_leaves= params['num_leaves'],
                max_depth= params['max_depth'],
                subsample_for_bin= params['subsample_for_bin'],
                min_child_samples= params['min_child_samples'],
                reg_alpha= params['reg_alpha'],
                reg_lambda= params['reg_lambda'],
                colsample_bytree= params['colsample_bytree'],
                is_unbalance= params['is_unbalance'],
                min_split_gain= params['min_split_gain'],
                subsample= params['boosting_type']['subsample'],
                objective= params['objective'],
                random_state= self.seed,
                silent=-1,
                verbose=-1
            )

            clf.fit(train_x, train_y, eval_set=[(train_x, train_y), (valid_x, valid_y)],
                    eval_metric='auc', verbose=400, early_stopping_rounds= self.early_stopping)

            oof_preds[valid_idx] = clf.predict_proba(valid_x, num_iteration=clf.best_iteration_)[:, 1]
            del clf, train_x, train_y, valid_x, valid_y
            gc.collect()

        result = roc_auc_score(self.train_df['TARGET'], oof_preds)
        self._iter += 1
        t0 = int(time.time() - t0)
        print("{}. CV AUC Score: {:.6f} - model run time: {}s".format(self._iter, result, t0))
        self._write_to_csv(hyperparameters, result, t0)
        return {'loss': 1 - result, 'hyperparameters': hyperparameters,
                'iteration': self._iter, 'train_time': t0, 'status': STATUS_OK}

    def _write_to_csv(self, hyperparameters, score, run_time):
        if not os.path.isfile(self.fn):
            self._create_csv()
        row = [score, run_time]
        hyperparameters['subsample'] = hyperparameters['boosting_type'].get('subsample', 1.0)
        hyperparameters['boosting_type'] = hyperparameters['boosting_type']['boosting_type']
        row.extend([hyperparameters[key] for key in self.grid_params_list])
        
        with open(self.fn, 'a') as csv_file:
            writer = csv.writer(csv_file)
            writer.writerow(row)

    def _create_csv(self):
        of_connection = open(self.fn, 'w')
        writer = csv.writer(of_connection)
        # Write column names
        headers = ['score', 'runtime']
        headers.extend(self.grid_params_list)
        writer.writerow(headers)
        of_connection.close()