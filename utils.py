'''Collection of useful functions for data analysis
and machine learning modelling
'''
from typing import List
import re

import pandas as pd
import numpy as np
from scipy import stats
from pandas.api.types import is_datetime64_any_dtype as is_datetime
from pandas.api.types import is_categorical_dtype


def summarize_cats(df: pd.DataFrame) -> pd.DataFrame:
    '''Create table summarizing categorical variables'''

    print(f"Dataset Shape: {df.shape}")
    summary = pd.DataFrame(df.dtypes, columns=['dtypes'])
    summary = summary.reset_index()
    summary['Column Name'] = summary['index']
    summary = summary[['Column Name', 'dtypes']]
    summary['Missing'] = df.isnull().sum().values
    summary['Uniques'] = df.nunique().values

    for name in summary['Column Name'].value_counts().index:

        # List unique values
        list_uniques = [str(v) for v in df[name].unique()]
        summary.loc[summary['Column Name'] == name,
                    'Values'] = ' '.join(list_uniques)

        # Calculate entropy
        shares = df[name].value_counts(normalize=True)
        summary.loc[summary['Column Name'] == name, 'Entropy'] = round(
            stats.entropy(shares, base=2), 2)

    return summary


def reduce_mem_usage(
    df: pd.DataFrame, cols_exclude: List[str] = []
) -> pd.DataFrame:
    '''Iterate through all the columns of a dataframe and modify
    the data type to reduce memory usage.

    Original code from
    https://www.kaggle.com/gemartin/load-data-reduce-memory-usage by @gemartin
    '''

    start_mem = df.memory_usage().sum() / 1024**2

    cols = [c for c in df.columns if c not in cols_exclude]
    print(
        "Reducing memory for the following columns: ",
        cols,
        sep='\n'
    )

    for col in cols:
        if is_datetime(df[col]) or is_categorical_dtype(df[col]):
            continue

        print(f"Reducing memory for {col}")
        col_type = df[col].dtype

        if col_type != object:

            c_min = df[col].min()
            c_max = df[col].max()

            if str(col_type)[:3] == "int":
                if c_min > np.iinfo(np.int8).min \
                        and c_max < np.iinfo(np.int8).max:

                    df[col] = df[col].astype(np.int8)

                elif c_min > np.iinfo(np.int16).min \
                        and c_max < np.iinfo(np.int16).max:

                    df[col] = df[col].astype(np.int16)

                elif c_min > np.iinfo(np.int32).min \
                        and c_max < np.iinfo(np.int32).max:

                    df[col] = df[col].astype(np.int32)

                elif c_min > np.iinfo(np.int64).min \
                        and c_max < np.iinfo(np.int64).max:

                    df[col] = df[col].astype(np.int64)

            else:
                df[col] = df[col].astype(np.float32)
        else:
            df[col] = df[col].astype("category")

    end_mem = df.memory_usage().sum() / 1024**2
    print(
        f"Memory usage before: {start_mem:.2f} MB",
        f"Memory usage after: {end_mem:.2f} MB "
        f"({100 * (start_mem - end_mem) / start_mem:.1f}% decrease)",
        sep='\n'
    )

    return df


def print_full(df: pd.DataFrame, num_rows: int = 100) -> None:
    '''Print the first num_rows rows of dataframe in full

    Resets display options back to default after printing
    '''
    pd.set_option('display.max_rows', len(df))
    pd.set_option('display.max_columns', None)
    pd.set_option('display.width', 2000)
    pd.set_option('display.float_format', '{:20,.2f}'.format)
    pd.set_option('display.max_colwidth', -1)
    display(df.iloc[0:num_rows])
    pd.reset_option('display.max_rows')
    pd.reset_option('display.max_columns')
    pd.reset_option('display.width')
    pd.reset_option('display.float_format')
    pd.reset_option('display.max_colwidth')

    return None
