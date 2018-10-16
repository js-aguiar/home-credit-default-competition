"""
home-credit-default-competition repository

Functions for preprocessing and extracting features
from application_train.csv and application_test.csv
"""

import os
import gc
import numpy as np
import pandas as pd
import utils
import config



def get_train_test(path, num_rows=None):
    """Preprocess and extract features from application train and test.

    Both files are combined in a single Dataframe for preprocessing,
    aggregation and feature engineering. 

    Arguments:
        path: Path to the folder where files are saved (string).
        num_rows: Number of rows to load; None to read all (int, default: None).

    Returns:
        df: DataFrame with processed data.
    """
    train = pd.read_csv(os.path.join(path, 'application_train.csv'), nrows=num_rows)
    test = pd.read_csv(os.path.join(path, 'application_test.csv'), nrows=num_rows)
    df = train.append(test)
    del train, test
    gc.collect()

    # Data cleaning
    df = df[df['CODE_GENDER'] != 'XNA']  # 4 people with XNA code gender
    df = df[df['AMT_INCOME_TOTAL'] < 20000000]  # Max income in test is 4M
    df['DAYS_EMPLOYED'].replace(365243, np.nan, inplace=True)
    df['DAYS_LAST_PHONE_CHANGE'].replace(0, np.nan, inplace=True)

    # Flag_document features - count and kurtosis
    docs = [f for f in df.columns if 'FLAG_DOC' in f]
    df['DOCUMENT_COUNT'] = df[docs].sum(axis=1)
    df['NEW_DOC_KURT'] = df[docs].kurtosis(axis=1)
    # Categorical age - based on target plot
    df['AGE_RANGE'] = df['DAYS_BIRTH'].apply(lambda x: _get_age_label(x, [27, 40, 50, 65, 99]))

    # New features based on External sources
    df['EXT_SOURCES_PROD'] = df['EXT_SOURCE_1'] * df['EXT_SOURCE_2'] * df['EXT_SOURCE_3']
    df['EXT_SOURCES_WEIGHTED'] = df.EXT_SOURCE_1 * 2 + df.EXT_SOURCE_2 * 1 + df.EXT_SOURCE_3 * 3
    np.warnings.filterwarnings('ignore', r'All-NaN (slice|axis) encountered')
    for function_name in ['min', 'max', 'mean', 'nanmedian', 'var']:
        feature_name = 'EXT_SOURCES_{}'.format(function_name.upper())
        df[feature_name] = eval('np.{}'.format(function_name))(
            df[['EXT_SOURCE_1', 'EXT_SOURCE_2', 'EXT_SOURCE_3']], axis=1)

    # Credit ratios
    df['CREDIT_TO_ANNUITY_RATIO'] = df['AMT_CREDIT'] / df['AMT_ANNUITY']
    df['CREDIT_TO_GOODS_RATIO'] = df['AMT_CREDIT'] / df['AMT_GOODS_PRICE']
    # Income ratios
    df['ANNUITY_TO_INCOME_RATIO'] = df['AMT_ANNUITY'] / df['AMT_INCOME_TOTAL']
    df['CREDIT_TO_INCOME_RATIO'] = df['AMT_CREDIT'] / df['AMT_INCOME_TOTAL']
    df['INCOME_TO_EMPLOYED_RATIO'] = df['AMT_INCOME_TOTAL'] / df['DAYS_EMPLOYED']
    df['INCOME_TO_BIRTH_RATIO'] = df['AMT_INCOME_TOTAL'] / df['DAYS_BIRTH']
    # Time ratios
    df['EMPLOYED_TO_BIRTH_RATIO'] = df['DAYS_EMPLOYED'] / df['DAYS_BIRTH']
    df['ID_TO_BIRTH_RATIO'] = df['DAYS_ID_PUBLISH'] / df['DAYS_BIRTH']
    df['CAR_TO_BIRTH_RATIO'] = df['OWN_CAR_AGE'] / df['DAYS_BIRTH']
    df['CAR_TO_EMPLOYED_RATIO'] = df['OWN_CAR_AGE'] / df['DAYS_EMPLOYED']

    # Groupby 1: Statistics for applications with the same education, occupation and age range
    group = ['NAME_EDUCATION_TYPE', 'OCCUPATION_TYPE', 'AGE_RANGE']
    df = utils.do_median(df, group, 'EXT_SOURCES_MEAN', 'GROUP1_EXT_SOURCES_MEDIAN')
    df = utils.do_std(df, group, 'EXT_SOURCES_MEAN', 'GROUP1_EXT_SOURCES_STD')
    df = utils.do_median(df, group, 'AMT_INCOME_TOTAL', 'GROUP1_INCOME_MEDIAN')
    df = utils.do_std(df, group, 'AMT_INCOME_TOTAL', 'GROUP1_INCOME_STD')
    df = utils.do_median(df, group, 'CREDIT_TO_ANNUITY_RATIO', 'GROUP1_CREDIT_TO_ANNUITY_MEDIAN')
    df = utils.do_std(df, group, 'CREDIT_TO_ANNUITY_RATIO', 'GROUP1_CREDIT_TO_ANNUITY_STD')
    df = utils.do_median(df, group, 'AMT_CREDIT', 'GROUP1_CREDIT_MEDIAN')
    df = utils.do_std(df, group, 'AMT_CREDIT', 'GROUP1_CREDIT_STD')
    df = utils.do_median(df, group, 'AMT_ANNUITY', 'GROUP1_ANNUITY_MEDIAN')
    df = utils.do_std(df, group, 'AMT_ANNUITY', 'GROUP1_ANNUITY_STD')

    # Groupby 2: Statistics for applications with the same credit duration, income type and education
    df['CREDIT_TO_ANNUITY_GROUP'] = df['CREDIT_TO_ANNUITY_RATIO'].apply(lambda x: _group_credit_to_annuity(x))
    group = ['CREDIT_TO_ANNUITY_GROUP', 'NAME_INCOME_TYPE', 'NAME_EDUCATION_TYPE']
    df = utils.do_median(df, group, 'EXT_SOURCES_MEAN', 'GROUP2_EXT_SOURCES_MEDIAN')
    df = utils.do_std(df, group, 'EXT_SOURCES_MEAN', 'GROUP2_EXT_SOURCES_STD')
    df = utils.do_median(df, group, 'AMT_INCOME_TOTAL', 'GROUP2_INCOME_MEDIAN')
    df = utils.do_std(df, group, 'AMT_INCOME_TOTAL', 'GROUP2_INCOME_STD')
    df = utils.do_median(df, group, 'CREDIT_TO_ANNUITY_RATIO', 'GROUP2_CREDIT_TO_ANNUITY_MEDIAN')
    df = utils.do_std(df, group, 'CREDIT_TO_ANNUITY_RATIO', 'GROUP2_CREDIT_TO_ANNUITY_STD')
    df = utils.do_median(df, group, 'AMT_CREDIT', 'GROUP2_CREDIT_MEDIAN')
    df = utils.do_std(df, group, 'AMT_CREDIT', 'GROUP2_CREDIT_STD')
    df = utils.do_median(df, group, 'AMT_ANNUITY', 'GROUP2_ANNUITY_MEDIAN')
    df = utils.do_std(df, group, 'AMT_ANNUITY', 'GROUP2_ANNUITY_STD')

    # Encode categorical features (LabelEncoder)
    df, _ = utils.label_encoder(df, None)
    # Drop some features
    df = _drop_application_columns(df)
    return df


def _drop_application_columns(df):
    """ Drop a few noise features. """
    drop_list = [
        'FLAG_EMP_PHONE', 'FLAG_MOBIL', 'FLAG_CONT_MOBILE',
        'LIVE_REGION_NOT_WORK_REGION', 'FLAG_EMAIL', 'FLAG_PHONE',
        'FLAG_OWN_REALTY', 'NAME_TYPE_SUITE',
        'AMT_REQ_CREDIT_BUREAU_HOUR', 'AMT_REQ_CREDIT_BUREAU_WEEK',
        'COMMONAREA_MODE', 'FLOORSMAX_MODE', 'FLOORSMIN_MODE',
        'LIVINGAPARTMENTS_MODE', 'LIVINGAREA_MODE', 'NONLIVINGAPARTMENTS_MODE',
        'NONLIVINGAREA_MODE', 'ELEVATORS_MEDI', 'EMERGENCYSTATE_MODE',
        'FONDKAPREMONT_MODE', 'HOUSETYPE_MODE', 'WALLSMATERIAL_MODE'
    ]
    # Drop most flag document columns
    for doc_num in [2,4,5,6,7,9,10,11,12,13,14,15,16,17,19,20,21]:
        drop_list.append('FLAG_DOCUMENT_{}'.format(doc_num))
    df.drop(drop_list, axis=1, inplace=True)
    return df


def _group_credit_to_annuity(x):
    """ Return the credit duration group label (int). """
    if x == np.nan: return 0
    elif x <= 6: return 1
    elif x <= 12: return 2
    elif x <= 18: return 3
    elif x <= 24: return 4
    elif x <= 30: return 5
    elif x <= 36: return 6
    else: return 7


def _get_age_label(days_birth, ranges):
    """ Return the age group label (int). """
    age_years = -days_birth / 365
    for label, max_age in enumerate(ranges):
        if age_years <= max_age:
            return label + 1
    else:
        return 0