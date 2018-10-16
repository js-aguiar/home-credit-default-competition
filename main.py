"""
Run the complete pipeline for Home Credit Default Risk Competition.

This model was part of the 7th place solution at this Kaggle competition,
being the full solution an ensemble of many models including gradient
boosting, neural networks and random forests with different features.
With this single model it is possible to.

This pipeline uses the LightGBM library with GOSS boosting as described
in "LightGBM: A Highly Efficient Gradient Boosting Decision Tree".
Configurations are all at the config.py file.

Author: js-aguiar
"""

import os
import gc
import sys
import pandas as pd
from pipeline import application_pipeline
from pipeline import bureau_pipeline
from pipeline import previous_pipeline
from pipeline import previous_balance_pipeline
from pipeline import other_features
import config
import model
import utils


def run_pipeline(use_pickled_features=False, debug=False):
    """Run the complete pipeline.

    Arguments:
        use_pickled_features: Use features saved as pickle 
        files (boolean, default: False).
        debug: Run pipeline with a subset of data (boolean, default: False)
    """
    num_rows = 30000 if debug else None  # Subset of data for debugging

    # Preprocess and extract features from each csv file
    with utils.timer("Application data"):
        if use_pickled_features:
            df = pd.read_pickle(os.path.join(config.PICKLED_DATA_DIRECTORY, 'application.pkl'))
        else:
            df = application_pipeline.get_train_test(config.DATA_DIRECTORY, num_rows=num_rows)
    with utils.timer("Bureau data"):
        if use_pickled_features:
            bureau_df = pd.read_pickle(os.path.join(config.PICKLED_DATA_DIRECTORY,
                                                    'bureau_and_balance.pkl'))
        else:
            bureau_df = bureau_pipeline.get_bureau(config.DATA_DIRECTORY, num_rows=num_rows)
        df = pd.merge(df, bureau_df, on='SK_ID_CURR', how='left')
        del bureau_df
        gc.collect()
    with utils.timer("Previous application data"):
        if use_pickled_features:
            prev_df = pd.read_pickle(os.path.join(config.PICKLED_DATA_DIRECTORY, 'previous.pkl'))
        else:
            prev_df = previous_pipeline.get_previous_applications(config.DATA_DIRECTORY, num_rows)
        df = pd.merge(df, prev_df, on='SK_ID_CURR', how='left')
        del prev_df
        gc.collect()
    with utils.timer("Previous balance data"):
        if use_pickled_features:
            pos = pd.read_pickle(os.path.join(config.PICKLED_DATA_DIRECTORY, 'pos_cash.pkl'))
        else:
            pos = previous_balance_pipeline.get_pos_cash(config.DATA_DIRECTORY, num_rows)
        df = pd.merge(df, pos, on='SK_ID_CURR', how='left')
        del pos
        gc.collect()
        if use_pickled_features:
            ins = pd.read_pickle(os.path.join(config.PICKLED_DATA_DIRECTORY, 'payments.pkl'))
        else:
            ins = previous_balance_pipeline.get_installment_payments(config.DATA_DIRECTORY, num_rows)
        df = pd.merge(df, ins, on='SK_ID_CURR', how='left')
        del ins
        gc.collect()
        if use_pickled_features:
            cc = pd.read_pickle(os.path.join(config.PICKLED_DATA_DIRECTORY, 'credit_card.pkl'))
        else:
            cc = previous_balance_pipeline.get_credit_card(config.DATA_DIRECTORY, num_rows)
        df = pd.merge(df, cc, on='SK_ID_CURR', how='left')
        del cc
        gc.collect()

    # Add ratios and groupby between different tables
    with utils.timer('Add extra features'):
        df = other_features.add_ratio_features(df)
        df = other_features.add_groupby_features(df)
    # Reduce memory usage
    df = utils.reduce_memory(df)
    # List categorical features for LightGBM partitioning mechanism (Fisher 1958)
    lgbm_categorical_feat = [
        'CODE_GENDER', 'FLAG_OWN_CAR', 'NAME_CONTRACT_TYPE',
        'NAME_EDUCATION_TYPE', 'NAME_FAMILY_STATUS', 'NAME_HOUSING_TYPE',
        'NAME_INCOME_TYPE', 'OCCUPATION_TYPE', 'ORGANIZATION_TYPE',
        'WEEKDAY_APPR_PROCESS_START'
    ]
    with utils.timer("Run LightGBM"):
        model.kfold_lightgbm_sklearn(df, lgbm_categorical_feat)


def _parse_command_line():
    """Parse the command line.

    If a argument is passed, a specific pipeline will be run and
    the frame saved as a pickle file. Otherwise, it will run the
    complete pipeline including the model module.
    """
    if '-application' in sys.argv:
        df = application_pipeline.get_train_test(config.DATA_DIRECTORY)
        _pickle_file(df, 'application.pkl')
    elif '-bureau' in sys.argv:
        df = bureau_pipeline.get_bureau(config.DATA_DIRECTORY)
        _pickle_file(df, 'bureau_and_balance.pkl')
    elif '-previous' in sys.argv:
        df = previous_pipeline.get_previous_applications(config.DATA_DIRECTORY)
        _pickle_file(df, 'previous.pkl')
    elif '-pos_cash' in sys.argv:
        df = previous_balance_pipeline.get_pos_cash(config.DATA_DIRECTORY)
        _pickle_file(df, 'pos_cash.pkl')
    elif '-payments' in sys.argv:
        df = previous_balance_pipeline.get_installment_payments(config.DATA_DIRECTORY)
        _pickle_file(df, 'payments.pkl')
    elif '-credit_card' in sys.argv:
        df = previous_balance_pipeline.get_credit_card(config.DATA_DIRECTORY)
        _pickle_file(df, 'credit_card.pkl')
    elif '-use_pickle' in sys.argv:
        # Run the pipeline with pickle files
        run_pipeline(use_pickled_features=True, debug=False)
    else:
        # Run the complete pipeline from raw data
        run_pipeline(use_pickled_features=False, debug=False)


def _pickle_file(df, file_name):
    df = utils.reduce_memory(df)
    df.to_pickle(os.path.join(config.PICKLED_DATA_DIRECTORY, file_name))
    print("Saved as {} - frame shape: {}".format(file_name, df.shape))


if __name__ == "__main__":
    with utils.timer("Pipeline runtime"):
        _parse_command_line()