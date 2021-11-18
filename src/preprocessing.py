'''
Downloads all the dataset from minio, preprocess the data, split the data into train, test and validation set.
'''
import argparse
import glob
import io
import json
import os
import re
import sys
from datetime import datetime
from io import StringIO
from pathlib import Path

import numpy as np
import pandas as pd
from minio import Minio
from sklearn.compose import ColumnTransformer
from sklearn.model_selection import (GroupShuffleSplit, KFold,
                                     cross_val_predict, cross_val_score,
                                     train_test_split)
from sklearn.preprocessing import (LabelEncoder, MinMaxScaler, OneHotEncoder,
                                   StandardScaler)

import config as acm
import plot as pl


def clean_data(df):
    """
    Basic cleaning of the data using the following criterion:
    - dropping any column with more than 50% missing values
    - dropping columns with less than 3 unquie values

    Args:
        df: dataset to be cleaned

    Returns:
       clean dataframe
    """
    df = df.copy()

    # Drop any column with more than 50% missing values
    half_count = len(df) / 2
    df = df.dropna(thresh=half_count, axis=1)

    # Again, there may be some columns with more than one unique value, but one value that has insignificant frequency in the data set.
    FREQ_THRESHOLD = 3
    PROTECTED_COLS = ['energy_eui_additional_fuel_gj_per_m_sq',
                      'energy_eui_electricity_gj_per_m_sq',
                      'energy_eui_natural_gas_gj_per_m_sq']

    # Identify columns that don't match the frequency threshold
    meets_threshold = df.nunique() >= FREQ_THRESHOLD
    meets_threshold[PROTECTED_COLS] = True

    # Drop any columns that didn't meet the threshold
    df = df.loc[:, meets_threshold]

    return df


def read_output(path):
    """
    Used to read the building simulation I/O file

    Args:
        path: file path where data is to be read from in minio

    Returns:
       btap_df: Dataframe containing the clean building parameters file.
       floor_sq: the square foot of the building
    """
    btap_df = acm.access_minio(operation='read',
                               path=path,
                               data='')

    floor_sq = btap_df['bldg_conditioned_floor_area_m_sq'].unique()

    # dropping output features present in the output file and dropping columns with one unique value
    output_drop_list = ['Unnamed: 0', ':erv_package', ':template']
    for col in btap_df.columns:
        if ((':' not in col) and (col not in ['energy_eui_additional_fuel_gj_per_m_sq', 'energy_eui_electricity_gj_per_m_sq', 'energy_eui_natural_gas_gj_per_m_sq', 'net_site_eui_gj_per_m_sq'])):
            output_drop_list.append(col)
    btap_df = btap_df.drop(output_drop_list, axis=1)
    btap_df = clean_data(btap_df)
    btap_df['Total Energy'] = btap_df[['net_site_eui_gj_per_m_sq']].sum(axis=1)
    drop_list = ['energy_eui_additional_fuel_gj_per_m_sq', 'energy_eui_electricity_gj_per_m_sq', 'energy_eui_natural_gas_gj_per_m_sq', 'net_site_eui_gj_per_m_sq']
    btap_df = btap_df.drop(drop_list, axis=1)

    return btap_df, floor_sq


def read_weather(path: str) -> pd.DataFrame:
    """
    Used to read the weather epw file from minio

    Args:
        path: file path where weather file is to be read from in minio

    Returns:
       btap_df: Dataframe containing the clean weather file.
    """
    # Load the data from blob storage.
    s3 = acm.establish_s3_connection(acm.settings.MINIO_URL, acm.settings.MINIO_ACCESS_KEY, acm.settings.MINIO_SECRET_KEY)
    weather_df = pd.read_parquet(s3.open(acm.settings.NAMESPACE.joinpath(path).as_posix()))

    # Remove spurious columns.
    weather_df = clean_data(weather_df)

    # date_int is used later to join data together.
    weather_df["date_int"] = weather_df['rep_time'].dt.strftime("%m%d").astype(int)

    # Aggregate data by day to reduce the complexity, leaving date values as they are.
    min_cols = {'year': 'min','month':'min','day':'min','rep_time':'min','date_int':'min'}
    sum_cols = [c for c in weather_df.columns if c not in [min_cols.keys()]]
    agg_funcs = dict(zip(sum_cols,['sum'] * len(sum_cols)))
    agg_funcs.update(min_cols)
    weather_df = weather_df.groupby([weather_df['rep_time'].dt.day]).agg(agg_funcs)

    # Remove the rep_time column, since later stages don't know to expect it.
    weather_df = weather_df.drop('rep_time', axis='columns')

    return weather_df


def read_hour_energy(path, floor_sq):
    """
    Used to read the weather epw file from minio

    Args:
        path: file path where weather.csv file is to be read from in minio
        floor_sq: the square foot of the building
    Returns:
       energy_hour_melt: Dataframe containing the clean and transposed hourly energy file.
    """

    energy_hour_df = acm.access_minio(operation='read',
                               path=path,
                               data='')

    eletricity_hour_df = energy_hour_df[energy_hour_df['Name'] != "Electricity:Facility"].groupby(['datapoint_id']).sum()
    energy_df = eletricity_hour_df.agg(lambda x: x / (floor_sq * 1000000))
    energy_df = energy_df.drop(['KeyValue'], axis=1)

    energy_df = clean_data(energy_df)
    energy_hour_df = energy_df.reset_index()
    energy_hour_melt = energy_hour_df.melt(id_vars=['datapoint_id'], var_name='Timestamp', value_name='energy')
    energy_hour_melt["date_int"] = pd.to_datetime(df['timestamp']).dt.strftime("%m%d").astype(int)
    energy_hour_melt = energy_hour_melt.groupby(['datapoint_id', 'date_int'])['energy'].agg(lambda x: x.sum()).reset_index()

    return energy_hour_melt


def groupsplit(X, y, valsplit):
    """
    Used to split the dataset by datapoint_id into train and test sets.

    The data is split to ensure all datapoints for each datapoint_id occurs completely in the respective dataset split.

    Note that where there is validation set, data is split with 80% for training and 20% for test set.

    Otherwise, the test set is split further with 60% as test set and 40% as validation set.

    Args:
        X: data excluding the target_variable
        y: target variable with datapoint_id
        valsplit: flag to indicate if there is a dataframe for the validation set. Accepeted values are "yes" or "no"
    Returns:
       X_train: X trainset
       y_train: y trainset
       X_test: X testset
       y_test_complete: Dataframe containing the target variable with corresponding datapointid
    """
    if valsplit == 'yes':
        gs = GroupShuffleSplit(n_splits=2, train_size=.7, random_state=42)
    else:
        gs = GroupShuffleSplit(n_splits=2, test_size=.4, random_state=42)

    train_ix, test_ix = next(gs.split(X, y, groups=X.datapoint_id))

    X_train = X.loc[train_ix]
    y_train = y.loc[train_ix]
    X_test = X.loc[test_ix]
    y_test_complete = y.loc[test_ix]

    return X_train, y_train, X_test, y_test_complete


def train_test_split(energy_daily_df, val_df, valsplit):
    """
    Used to split the dataset by datapoint_id into train , test and validation sets.

    Args:
        energy_daily_df: the merged dataframe for simulation I/O, weather, and hourly energy file.
        val_df: the merged dataframe for simulation I/O, weather, and hourly energy file validation set. Where there is no validation set, its value is null
        valsplit: flag to indicate if there is a dataframe for the validation set. Accepeted values are "yes" or "no"
    Returns:
       X_train: X trainset
       y_train: y trainset
       X_test: X testset
       y_test_complete: Dataframe containing the target variable with corresponding datapointid
       X_validate: X validate set
       y_validate: y validate set
       y_validate_complete: Dataframe containing the target variable with corresponding datapointid for the validation set
    """
    drop_list = ['index', 'Dew Point Temperature', 'Horizontal Infrared Radiation Intensity', ':datapoint_id', 'level_0', 'index', 'date_int', ':datapoint_id', 'Year', 'Month', 'Day', 'Hour']

    # split to train and test datasets
    y = energy_daily_df[['energy', 'datapoint_id', 'Total Energy']]
    X = energy_daily_df.drop(['energy'], axis=1)
    X_train, y_train, X_test, y_test_complete = groupsplit(X, y, valsplit)
    y_test = y_test_complete[['energy', 'datapoint_id']]

    if valsplit == 'yes':
        y_val = val_df[['energy', 'datapoint_id', 'Total Energy']]
        X_val = val_df.drop(['energy'], axis=1)
        y_validate_complete = y_val
        X_validate = X_val.drop(drop_list, axis=1)
        X_validate = X_validate.drop(['datapoint_id', 'Total Energy'], axis=1)
        y_validate = y_validate_complete[['energy', 'datapoint_id']]
    else:
        X_test = X_test.reset_index(drop=True)
        y_test = y_test_complete.reset_index(drop=True)
        X_test, y_test, X_validate, y_validate_complete = groupsplit(X_test, y_test, valsplit)
        y_validate = y_validate_complete[['energy', 'datapoint_id']]
        X_validate = X_validate.drop(drop_list, axis=1)
        X_validate = X_validate.drop(['datapoint_id', 'Total Energy'], axis=1)

    energy_daily_df = energy_daily_df.drop(drop_list, axis=1)
    X_train = X_train.drop(drop_list, axis=1)
    X_test = X_test.drop(drop_list, axis=1)
    y_train = y_train['energy']
    X_train = X_train.drop(['datapoint_id', 'Total Energy'], axis=1)
    X_test = X_test.drop(['datapoint_id', 'Total Energy'], axis=1)

    return X_train, X_test, y_train, y_test, y_test_complete, X_validate, y_validate, y_validate_complete


def categorical_encode(x_train, x_test, x_validate):
    """
    Used to encode the categorical variables contained in the x_train, x_test and x_validate

    Note that the encoded data return creates additional columns equivalent to the unique categorical values in the each categorical column.

    Args:
         X_train: X trainset
         X_test:  X testset
         X_validate: X validate set
    Returns:
        X_train_oh: encoded X trainset
        X_test_oh: encoded X testset
        x_val_oh: encoded X validate set
        all_features: all features after encoding.
       y_validate_complete: Dataframe containing the target variable with corresponding datapointid for the validation set
    """
    # extracting the categorical columns
    cat_cols = x_train.select_dtypes(include=['object']).columns
    other_cols = x_train.drop(columns=cat_cols).columns

    # Create the encoder.
    encoder = OneHotEncoder(handle_unknown="ignore")
    ct = ColumnTransformer([('ohe', OneHotEncoder(sparse=False), cat_cols)], remainder=MinMaxScaler())
    # Apply the encoder.
    x_train_oh = ct.fit_transform(x_train)
    x_test_oh = ct.transform(x_test)
    x_val_oh = ct.transform(x_validate)
    encoded_cols = ct.named_transformers_.ohe.get_feature_names(cat_cols)
    all_features = np.concatenate([encoded_cols, other_cols])

    return x_train_oh, x_test_oh, x_val_oh, all_features


def process_data(args):
    """
    Used to encode the categorical variables contained in the x_train, x_test and x_validate

    Note that the encoded data return creates additional columns equivalent to the unique categorical values in the each categorical column.

    Args:
         args: arguements provided from the main

    Returns:
        the preprocessed dataset is uploaded to minio
    """
    btap_df, floor_sq = read_output(args.in_build_params)
    weather_df = read_weather(args.in_weather)
    energy_hour_df = read_hour_energy(args.in_hour, floor_sq)
    energy_hour_merge = pd.merge(energy_hour_df, btap_df, left_on=['datapoint_id'], right_on=[':datapoint_id'], how='left').reset_index()
    energy_daily_df = pd.merge(energy_hour_merge, weather_df, on='date_int', how='left').reset_index()

    if args.in_build_params_val:
        btap_df_val, floor_sq = read_output(args.in_build_params_val)
        energy_hour_df_val = read_hour_energy(args.in_hour_val, floor_sq)
        energy_hour_merge_val = pd.merge(energy_hour_df_val, btap_df_val, left_on=['datapoint_id'], right_on=[':datapoint_id'], how='left').reset_index()
        energy_daily_df_val = pd.merge(energy_hour_merge_val, weather_df, on='date_int', how='left').reset_index()
        X_train, X_test, y_train, y_test, y_test_complete, X_validate, y_validate, y_validate_complete = train_test_split(energy_daily_df, energy_daily_df_val, 'yes')
    else:
        energy_hour_df_val = ''
        btap_df_val = ''
        energy_daily_df_val = ''
        X_train, X_test, y_train, y_test, y_test_complete, X_validate, y_validate, y_validate_complete = train_test_split(energy_daily_df, energy_daily_df_val, 'no')

    pl.corr_plot(energy_daily_df)
    X_train_oh, X_test_oh, X_val_oh, all_features = categorical_encode(X_train, X_test, X_validate)

    # Creates `data` structure to save and share train and test datasets.
    data = {'features': all_features.tolist(),
            'y_train': y_train.to_json(orient="values"),
            'X_train': X_train_oh.tolist(),
            'X_test': X_test_oh.tolist(),
            'y_test': y_test.values.tolist(),
            'y_test_complete': y_test_complete.values.tolist(),
            'X_validate': X_val_oh.tolist(),
            'y_validate': y_validate.values.tolist(),
            'y_validate_complete': y_validate_complete.values.tolist(),
           }

    data_json = json.dumps(data).encode('utf-8')
    acm.access_minio(operation='copy',
                 path=args.output_path,
                 data=data_json)

    return


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    # Paths must be passed in, not hardcoded
    parser.add_argument('--in_hour', type=str, help='The minio bucket where the data is located in')
    parser.add_argument('--in_build_params', type=str, help='Name of data file to be read')
    parser.add_argument('--in_weather', type=str, help='Name of weather file to be read')
    parser.add_argument('--output_path', type=str, help='Path of the local file where the output file should be written.')
    parser.add_argument('--in_hour_val', type=str, help='The minio bucket where the data is located in')
    parser.add_argument('--in_build_params_val', type=str, help='Name of data file to be read')
    args = parser.parse_args()

    process_data(args)

    # to run the program use the command below
# python3 preprocessing.py --in_build_params input_data/output_2021-10-04.xlsx --in_hour input_data/total_hourly_res_2021-10-04.csv --in_weather weather/CAN_QC_Montreal-Trudeau.Intl.AP.716270_CWEC2016.epw.parquet --output_path output_data/preprocessing_out --in_build_params_val input_data/output.xlsx --in_hour_val input_data/total_hourly_res.csv
