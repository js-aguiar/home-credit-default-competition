"""
home-credit-default-competition repository

Utility functions for aggregating, grouping, reducing
memory usage and others.
"""

import gc
import time
import numpy as np
import pandas as pd
import multiprocessing as mp
from contextlib import contextmanager
from sklearn.linear_model import LinearRegression
from scipy.stats import kurtosis, iqr, skew
import config

@contextmanager
def timer(name):
    """Decorator that print the elapsed time and argument."""
    t0 = time.time()
    yield
    print("{} - done in {:.0f}s".format(name, time.time() - t0))


def group(df_to_agg, prefix, aggregations, aggregate_by= 'SK_ID_CURR'):
    """Group DataFrame and perform aggregations.

    Arguments:
        df_to_agg: DataFrame to be grouped.
        prefix: New features name prefix
        aggregations: Dictionary or list of aggregations (see pandas aggregate)
        aggregate_by: Column to group DataFrame

    Returns:
        agg_df: DataFrame with new features
    """
    agg_df = df_to_agg.groupby(aggregate_by).agg(aggregations)
    agg_df.columns = pd.Index(['{}{}_{}'.format(prefix, e[0], e[1].upper())
                               for e in agg_df.columns.tolist()])
    return agg_df.reset_index()


def group_and_merge(df_to_agg, df_to_merge, prefix, aggregations, aggregate_by= 'SK_ID_CURR'):
    """Group DataFrame, perform aggregations and merge with the second one.

    Arguments:
        df_to_agg: DataFrame to be grouped.
        df_to_merge: DataFrame where agg will be merged.
        prefix: Prefix for new features names (string)
        aggregations: Dictionary or list of aggregations (see pandas aggregate)
        aggregate_by: Column name to group DataFrame  (string)

    Returns:
        df_to_merge: Second dataframe with the aggregated features from the first.
    """
    agg_df = group(df_to_agg, prefix, aggregations, aggregate_by= aggregate_by)
    return df_to_merge.merge(agg_df, how='left', on= aggregate_by)


def do_mean(df, group_cols, counted, agg_name):
    """Add the mean for each group for a given feature in a DataFrame.

    Arguments:
        df: DataFrame to group and add the mean feature.
        group_cols: List with column or columns names to group by.
        counted: Feature name to get the mean (string).
        agg_name: New feature name (string)

    Returns:
        df: Same DataFrame with the new feature
    """
    gp = df[group_cols + [counted]].groupby(group_cols)[counted].mean().reset_index().rename(
        columns={counted: agg_name})
    df = df.merge(gp, on=group_cols, how='left')
    del gp
    gc.collect()
    return df


def do_median(df, group_cols, counted, agg_name):
    """Add the median for each group for a given feature in a DataFrame.

    Arguments:
        df: DataFrame to group and add the median feature.
        group_cols: List with column or columns names to group by.
        counted: Feature name to get the median (string).
        agg_name: New feature name (string)

    Returns:
        df: Same DataFrame with the new feature
    """
    gp = df[group_cols + [counted]].groupby(group_cols)[counted].median().reset_index().rename(
        columns={counted: agg_name})
    df = df.merge(gp, on=group_cols, how='left')
    del gp
    gc.collect()
    return df


def do_std(df, group_cols, counted, agg_name):
    """Add the standard deviation for each group for a given feature.

    Arguments:
        df: DataFrame to group and add the standard deviation feature.
        group_cols: List with column or columns names to group by.
        counted: Feature name to get the standard deviation (string).
        agg_name: New feature name (string)

    Returns:
        df: Same DataFrame with the new feature
    """
    gp = df[group_cols + [counted]].groupby(group_cols)[counted].std().reset_index().rename(
        columns={counted: agg_name})
    df = df.merge(gp, on=group_cols, how='left')
    del gp
    gc.collect()
    return df


def do_sum(df, group_cols, counted, agg_name):
    """Add the sum for each group for a given feature in a DataFrame.

    Arguments:
        df: DataFrame to group and add the sum feature.
        group_cols: List with column or columns names to group by.
        counted: Feature name to get the sum (string).
        agg_name: New feature name (string)

    Returns:
        df: Same DataFrame with the new feature
    """
    gp = df[group_cols + [counted]].groupby(group_cols)[counted].sum().reset_index().rename(
        columns={counted: agg_name})
    df = df.merge(gp, on=group_cols, how='left')
    del gp
    gc.collect()
    return df


def one_hot_encoder(df, categorical_columns=None, nan_as_category=True):
    """Create a new column for each categorical value in categorical columns.

    Arguments:
        df: DataFrame.
        categorical_columns: List of column names;
        if None all columns with object datatype will be considered.
        nan_as_category: If True add column for missing values (boolean)

    Returns:
        df: Same DataFrame with the encoded features
        categorical_columns: List with the names of OHE columns.
    """
    original_columns = list(df.columns)
    if not categorical_columns:
        categorical_columns = [col for col in df.columns if df[col].dtype == 'object']
    df = pd.get_dummies(df, columns=categorical_columns, dummy_na=nan_as_category)
    categorical_columns = [c for c in df.columns if c not in original_columns]
    return df, categorical_columns


def label_encoder(df, categorical_columns=None):
    """Encode categorical values as integers (0,1,2,3...) with pandas.factorize.

    Arguments:
        df: DataFrame.
        categorical_columns: List of column names;
        if None all columns with object datatype will be considered.

    Returns:
        df: Same DataFrame with the encoded features
        categorical_columns: List with the names of OHE columns.
    """
    if not categorical_columns:
        categorical_columns = [col for col in df.columns if df[col].dtype == 'object']
    for col in categorical_columns:
        df[col], _ = pd.factorize(df[col])
    return df, categorical_columns


def reduce_memory(df):
    """Reduce memory usage of a dataframe by setting data types. """
    start_mem = df.memory_usage().sum() / 1024 ** 2
    print('Initial df memory usage is {:.2f} MB for {} columns'
          .format(start_mem, len(df.columns)))

    for col in df.columns:
        col_type = df[col].dtypes
        if col_type != object:
            cmin = df[col].min()
            cmax = df[col].max()
            if str(col_type)[:3] == 'int':
                # Can use unsigned int here too
                if cmin > np.iinfo(np.int8).min and cmax < np.iinfo(np.int8).max:
                    df[col] = df[col].astype(np.int8)
                elif cmin > np.iinfo(np.int16).min and cmax < np.iinfo(np.int16).max:
                    df[col] = df[col].astype(np.int16)
                elif cmin > np.iinfo(np.int32).min and cmax < np.iinfo(np.int32).max:
                    df[col] = df[col].astype(np.int32)
                elif cmin > np.iinfo(np.int64).min and cmax < np.iinfo(np.int64).max:
                    df[col] = df[col].astype(np.int64)
            else:
                if cmin > np.finfo(np.float16).min and cmax < np.finfo(np.float16).max:
                    df[col] = df[col].astype(np.float16)
                elif cmin > np.finfo(np.float32).min and cmax < np.finfo(np.float32).max:
                    df[col] = df[col].astype(np.float32)
                else:
                    df[col] = df[col].astype(np.float64)
    end_mem = df.memory_usage().sum() / 1024 ** 2
    memory_reduction = 100 * (start_mem - end_mem) / start_mem
    print('Final memory usage is: {:.2f} MB - decreased by {:.1f}%'.format(end_mem, memory_reduction))
    return df


def add_features_in_group(features, gr_, feature_name, aggs, prefix):
    """Generate all aggregation features in aggs list for the given feature.

    Arguments:
        features: DataFrame where features will be add.
        gr_: pandas Groupby object.
        feature_name: Feature used for aggregation (string).
        aggs: List of strings with pandas aggregations names.
        prefix: Prefix for the name of the new features (string).

    Returns:
        features: Same DataFrame from arguments with new features.
    """
    for agg in aggs:
        if agg == 'sum':
            features['{}{}_sum'.format(prefix, feature_name)] = gr_[feature_name].sum()
        elif agg == 'mean':
            features['{}{}_mean'.format(prefix, feature_name)] = gr_[feature_name].mean()
        elif agg == 'max':
            features['{}{}_max'.format(prefix, feature_name)] = gr_[feature_name].max()
        elif agg == 'min':
            features['{}{}_min'.format(prefix, feature_name)] = gr_[feature_name].min()
        elif agg == 'std':
            features['{}{}_std'.format(prefix, feature_name)] = gr_[feature_name].std()
        elif agg == 'count':
            features['{}{}_count'.format(prefix, feature_name)] = gr_[feature_name].count()
        elif agg == 'skew':
            features['{}{}_skew'.format(prefix, feature_name)] = skew(gr_[feature_name])
        elif agg == 'kurt':
            features['{}{}_kurt'.format(prefix, feature_name)] = kurtosis(gr_[feature_name])
        elif agg == 'iqr':
            features['{}{}_iqr'.format(prefix, feature_name)] = iqr(gr_[feature_name])
        elif agg == 'median':
            features['{}{}_median'.format(prefix, feature_name)] = gr_[feature_name].median()
    return features


def add_trend_feature(features, gr, feature_name, prefix):
    """Return linear regression parameters (linear trend) for given feature.

    Arguments:
        features: DataFrame where trends will be add.
        gr: pandas Groupby object.
        feature_name: Feature for calculating trends (string).
        prefix: Prefix for the name of the new features (string).

    Returns:
        features: Same DataFrame from arguments with new features.
    """
    y = gr[feature_name].values
    try:
        x = np.arange(0, len(y)).reshape(-1, 1)
        lr = LinearRegression()
        lr.fit(x, y)
        trend = lr.coef_[0]
    except:
        trend = np.nan
    features['{}{}'.format(prefix, feature_name)] = trend
    return features


def parallel_apply(groups, func, index_name='Index', num_workers=0, chunk_size=100000):
    """Apply the given function using multiprocessing (in parallel).

    Arguments:
        groups: pandas Groupby object.
        func: Function to apply.
        index_name: pandas Index (string).
        num_workers: Number of jobs (threads). If zero, config value will be used.

    Returns:
        features: Same DataFrame from arguments with new features.
    """
    if num_workers <= 0: num_workers = config.NUM_THREADS
    #n_chunks = np.ceil(1.0 * groups.ngroups / chunk_size)
    indeces, features = [], []
    for index_chunk, groups_chunk in chunk_groups(groups, chunk_size):
        with mp.pool.Pool(num_workers) as executor:
            features_chunk = executor.map(func, groups_chunk)
        features.extend(features_chunk)
        indeces.extend(index_chunk)

    features = pd.DataFrame(features)
    features.index = indeces
    features.index.name = index_name
    return features


def chunk_groups(groupby_object, chunk_size):
    """Iterator that yields chunks of data with chunk_size."""
    n_groups = groupby_object.ngroups
    group_chunk, index_chunk = [], []
    for i, (index, df) in enumerate(groupby_object):
        group_chunk.append(df)
        index_chunk.append(index)
        if (i + 1) % chunk_size == 0 or i + 1 == n_groups:
            group_chunk_, index_chunk_ = group_chunk.copy(), index_chunk.copy()
            group_chunk, index_chunk = [], []
            yield index_chunk_, group_chunk_