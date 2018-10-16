"""
home-credit-default-competition repository

Simple functions to create ratios and group features between
different data files.
"""

import gc
import pandas as pd

def add_ratio_features(df):
    """Add division between features - highly effective for GBDT models.

    Arguments:
        df: pandas DataFrame with features from all csv files

    Returns:
        df: Same DataFrame with the new features
    """

    ratio_features = {
        # CREDIT TO INCOME RATIO
        'BUREAU_INCOME_CREDIT_RATIO':
            ['BUREAU_AMT_CREDIT_SUM_MEAN', 'AMT_INCOME_TOTAL'],
        'BUREAU_ACTIVE_CREDIT_TO_INCOME_RATIO':
            ['BUREAU_ACTIVE_AMT_CREDIT_SUM_SUM', 'AMT_INCOME_TOTAL'],
        # PREVIOUS TO CURRENT CREDIT RATIO
        'CURRENT_TO_APPROVED_CREDIT_MIN_RATIO':
            ['APPROVED_AMT_CREDIT_MIN', 'AMT_CREDIT'],
        'CURRENT_TO_APPROVED_CREDIT_MAX_RATIO':
            ['APPROVED_AMT_CREDIT_MAX', 'AMT_CREDIT'],
        'CURRENT_TO_APPROVED_CREDIT_MEAN_RATIO': 
            ['APPROVED_AMT_CREDIT_MEAN', 'AMT_CREDIT'],
        # PREVIOUS TO CURRENT ANNUITY RATIO
        'CURRENT_TO_APPROVED_ANNUITY_MAX_RATIO':
            ['APPROVED_AMT_ANNUITY_MAX', 'AMT_ANNUITY'],
        'CURRENT_TO_APPROVED_ANNUITY_MEAN_RATIO':
            ['APPROVED_AMT_ANNUITY_MEAN', 'AMT_ANNUITY'],
        'PAYMENT_MIN_TO_ANNUITY_RATIO':
            ['INS_AMT_PAYMENT_MIN', 'AMT_ANNUITY'],
        'PAYMENT_MAX_TO_ANNUITY_RATIO':
            ['INS_AMT_PAYMENT_MAX', 'AMT_ANNUITY'],
        'PAYMENT_MEAN_TO_ANNUITY_RATIO':
            ['INS_AMT_PAYMENT_MEAN', 'AMT_ANNUITY'],
        # BUREAU TO CURRENT ANNUITY RATIO
        'CURRENT_TO_BUREAU_ANNUITY_MEAN_RATIO':
            ['BUREAU_AMT_ANNUITY_MEAN', 'AMT_ANNUITY'],
        # PREVIOUS TO CURRENT CREDIT TO ANNUITY RATIO
        'CTA_CREDIT_TO_ANNUITY_MAX_RATIO':
            ['APPROVED_CREDIT_TO_ANNUITY_RATIO_MAX', 'CREDIT_TO_ANNUITY_RATIO'],
        'CTA_CREDIT_TO_ANNUITY_MEAN_RATIO':
            ['APPROVED_CREDIT_TO_ANNUITY_RATIO_MEAN', 'CREDIT_TO_ANNUITY_RATIO'],
        # DAYS AND TIME DIFFERENCES AND RATIOS
        'DAYS_DECISION_MEAN_TO_BIRTH':
            ['APPROVED_DAYS_DECISION_MEAN', 'DAYS_BIRTH'],
        'DAYS_CREDIT_MEAN_TO_BIRTH':
            ['BUREAU_DAYS_CREDIT_MEAN', 'DAYS_BIRTH'],
        'DAYS_DECISION_MEAN_TO_EMPLOYED':
            ['APPROVED_DAYS_DECISION_MEAN', 'DAYS_EMPLOYED'],
        'DAYS_CREDIT_MEAN_TO_EMPLOYED':
            ['BUREAU_DAYS_CREDIT_MEAN', 'DAYS_EMPLOYED'],
        'HOUR_APPR_RATIO':
            ['HOUR_APPR_PROCESS_START', 'PREV_HOUR_APPR_PROCESS_START_MEAN'],
    }
    for ratio_feature, features in ratio_features.items():
        df[ratio_feature] = df[features[0]] / df[features[1]]
    # Hour: difference from previous applications
    df['HOUR_APPR_DIFFERENCE'] = df['HOUR_APPR_PROCESS_START'] - df['PREV_HOUR_APPR_PROCESS_START_MEAN']
    return df


def add_groupby_features(df):
    """Group some features by duration (credit/annuity) and extract the mean, median and std.

    Arguments:
        df: pandas DataFrame with features from all csv files

    Returns:
        df: Same DataFrame with the new features
    """
    g = 'CREDIT_TO_ANNUITY_RATIO'
    feats = ['EXT_SOURCE_1', 'EXT_SOURCE_2', 'EXT_SOURCE_3', 'BUREAU_ACTIVE_DAYS_CREDIT_MEAN',
             'APPROVED_CNT_PAYMENT_MEAN', 'EXT_SOURCES_PROD', 'CREDIT_TO_GOODS_RATIO',
             'INS_DAYS_ENTRY_PAYMENT_MAX', 'EMPLOYED_TO_BIRTH_RATIO', 'EXT_SOURCES_MEAN',
             'DAYS_REGISTRATION', 'DAYS_EMPLOYED', 'DAYS_BIRTH']
    agg = df.groupby(g)[feats].agg(['mean', 'median', 'std'])
    agg.columns = pd.Index(['CTAR_' + e[0] + '_' + e[1].upper() for e in agg.columns.tolist()])
    df = df.join(agg, how='left', on=g)
    del agg
    gc.collect()
    return df