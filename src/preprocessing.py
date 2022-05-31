"""
Preprocesses each dataset and splits the data into train, test and validation set.

CLI arguments match those defined by ``main()``.
"""
# TODO: Remove all deep copy calls, limiting number of copies in memory

import argparse
import copy
import glob
import io
import json
import logging
import os
import re
import sys
from datetime import datetime
from io import StringIO
from pathlib import Path

import joblib
import matplotlib
import numpy as np
import pandas as pd
import pyarrow
import typer
from minio import Minio
from sklearn.compose import ColumnTransformer
from sklearn.model_selection import (GroupShuffleSplit, KFold,
                                     cross_val_predict, cross_val_score,
                                     train_test_split)
from sklearn.preprocessing import (LabelEncoder, MinMaxScaler, OneHotEncoder,
                                   StandardScaler)

import config
import plot as pl

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def clean_data(df) -> pd.DataFrame:
    """
    Basic cleaning of the data using the following criterion:

    - dropping any column with more than 50% missing values
      The 50% threshold is a way to eliminate columns with too much missing values in the dataset.
      We cant use N/A as it will elimnate the entire row /datapoint_id. Giving the number of features we have to work it its better we eliminate
      columns with features that have too much missing values than to eliminate by rows, which is what N/A will do .
    - dropping columns with 1 unique value
      For columns with  1 unique values are dropped during data cleaning as they have low variance
      and hence have little or no significant contribution to the accuracy of the model.

    Args:
        df: dataset to be cleaned

    Returns:
       df: cleaned dataframe
    """
    df = df.copy()
    # Drop any column with more than 50% missing values
    half_count = len(df) / 2
    df = df.dropna(thresh=half_count, axis=1)

    # Again, there may be some columns with more than one unique value, but one value that has insignificant frequency in the data set.
    for col in df.columns:
        num = len(df[col].unique())
        # *If processing in a non-merged batch, need to specify single values which should be maintined
        if ((len(df[col].unique()) == 1) and (col not in ['energy_eui_additional_fuel_gj_per_m_sq','energy_eui_electricity_gj_per_m_sq','energy_eui_natural_gas_gj_per_m_sq'])):
            df.drop(col,inplace=True,axis=1)
    return df

def process_building_files(path_elec, path_gas, clean_dataframe=True):
    """
    Used to read the building simulation I/O file

    Args:
        path_elec: file path where data is to be read. This is a mandatory parameter and in the case where only one simulation I/O file is provided, the path to this file should be indicated here.
        path_gas: This would be path to the gas output file. This is optional, if there is no gas output file to the loaded, then a value of path_gas ='' should be used
        clean_dataframe: True if the loaded data should be cleaned, False if cleaning should be skipped

    Returns:
       btap_df: Dataframe containing the clean building parameters file.
       floor_sq: the square foot of the building
    """
    btap_df = None

    btap_df = pd.read_excel(path_elec)
    if path_gas:
        btap_df = pd.concat([btap_df, pd.read_excel(path_gas)], ignore_index=True)
    # Building meters squared
    floor_sq = btap_df['bldg_conditioned_floor_area_m_sq'].unique()

    # Dropping output features present in the output file and dropping columns with one unique value
    output_drop_list = ['Unnamed: 0', ':erv_package', ':template']
    for col in btap_df.columns:
        if ((':' not in col) and (col not in ['energy_eui_additional_fuel_gj_per_m_sq', 'energy_eui_electricity_gj_per_m_sq', 'energy_eui_natural_gas_gj_per_m_sq', 'net_site_eui_gj_per_m_sq'])):
            output_drop_list.append(col)
    btap_df = btap_df.drop(output_drop_list,axis=1)
    if clean_dataframe:
        btap_df = clean_data(btap_df)
    if 'net_site_eui_gj_per_m_sq' in btap_df:
        btap_df['Total Energy'] = btap_df[['net_site_eui_gj_per_m_sq']].sum(axis=1)
    drop_list=['energy_eui_additional_fuel_gj_per_m_sq','energy_eui_electricity_gj_per_m_sq','energy_eui_natural_gas_gj_per_m_sq','net_site_eui_gj_per_m_sq',':analysis_id']
    # Drop any remaining fields which exist, ignoring raised errors from any which do not exist or have ben removed
    btap_df = btap_df.drop(drop_list, axis=1, errors='ignore')

    return btap_df, floor_sq

def process_building_files_batch(directory: str, start_date: str, end_date: str) -> pd.DataFrame:
    """
    Given a directory of .xlsx building files, process and clean each file, combining
    them into one dataframe with entries for every day within a provided timespan.
    Each row will be assigned a custom identifier following the format:
    name_of_file/index_number_in_file
    The square foot of the building will be added in a column 'square_foot' which can be removed
    if unused.

    Args:
        directory: Directory containing one or more .xlsx building files (with no other .xlsx files present)
        start_date: Starting date of the specified timespan (in the form Month_number-Day_number)
        end_date: Ending date of the specified timespan (in the form Month_number-Day_number)

    Returns:
       buildings_df: Dataframe containing the clean building parameters files.
    """
    # TODO: Check if directory exists
    ...
    # Define the year to be used when generating the date range, since we skip leap years, 2022 is used
    TEMP_YEAR = '2022-'
    start_date = TEMP_YEAR + start_date
    end_date = TEMP_YEAR + end_date
    # Define the dataframe variable
    buildings_df = None
    # Find all files in the directory
    for filename in os.listdir(directory):
        # Only process .xlsx files in case other filetypes are in the directory
        if filename.endswith(".xlsx"):
            # Process the single building file
            building_df, floor_sq = process_building_files(directory + '/' + filename, False, False)
            # Update the received outputs include a custom identifier column of the form '<filename>/<row_index>'
            building_df["Prediction Identifier"] = building_df.apply(lambda r: filename + '/' + str(r.name), axis=1)
            # Add a column to track the floor_sq values as needed for the specific file
            building_df["floor_sq"] = building_df.apply(lambda r: floor_sq[0], axis=1)
            # Duplicate each row for every day between the specified start and end dates (ignoring the year)
            # Holds a set of dataframes which each have the specified date range for a specific row
            # Uses this approach since the dictionary must be edited before converted back into a dataframe
            date_dfs = []
            # For each row, create a dataframe for every day within the specified start and end dates
            for _, row in building_df.iterrows():
                # Convert the row into a dictionary to reconstruct a dataframe from
                dict_vals = row.to_dict()
                # Define a series of days for each day in the specified range (in the form Month_numberDay_number)
                dict_vals["date_int"] = pd.Series(pd.date_range(start_date, end_date).strftime("%m%d")).astype("int32")
                # Track the updated dataframe
                date_dfs.append(pd.DataFrame(dict_vals))
            # Combine all dataframes together to achieve a dataframe with each original row now containing an additional row
            # with each month and day within the specified range
            building_df = pd.concat(date_dfs)

            # Set/merge the output with the other processed building data
            if buildings_df is not None:
                buildings_df = pd.concat([buildings_df, building_df], ignore_index=True)
            else:
                buildings_df = building_df
    return buildings_df

def read_weather(path: str) -> pd.DataFrame:
    """
    Used to read the weather .parquet file

    Args:
        path: file path where weather file is to be read

    Returns:
       btap_df: Dataframe containing the clean weather file.
    """
    # Warn the user if they supply a weather file that looks like a CSV.
    if path.endswith('.csv'):
        logger.warn("Weather data must be in parquet format. Ensure %s is valid.", path)
    # Load the data.
    weather_df = None

    try:
        weather_df = pd.read_parquet(path)
        logger.debug("weather hours: %s", weather_df['hour'].unique())
    except pyarrow.lib.ArrowInvalid as err:
        logger.error("Invalid weather file format supplied. Is %s a parquet file?", path)
        sys.exit(1)

    # Remove spurious columns.
    weather_df = clean_data(weather_df)
    # date_int is used later to join data together.
    weather_df["date_int"] = weather_df['rep_time'].dt.strftime("%m%d").astype(int)

    # Remove the rep_time column, since later stages don't know to expect it.
    weather_df = weather_df.drop('rep_time', axis='columns')
    weather_df = weather_df.groupby(['date_int']).agg(lambda x: x.sum())
    logger.debug("weather data shape: %s", weather_df.shape)
    logger.debug("Weather data NA values:\n%s", weather_df.isna().any())
    return weather_df

def process_hourly_energy(path_elec, path_gas, floor_sq):
    """
    Used to read the hourly energy file(s)

    Args:
        path_elec: file path where the electric hourly energy consumed file is to be read. This is a mandatory parameter and in the case where only one hourly energy output file is provided, the path to this file should be indicated here.
        path_gas: This would be path to the gas output file. This is optional, if there is no gas output file to the loaded, then a value of path_gas ='' should be used
        floor_sq: the square foot of the building

    Returns:
       energy_df: Dataframe containing the clean and transposed hourly energy file.
    """
    energy_df = pd.read_csv(path_elec)
    if path_gas:
        energy_df = pd.concat([energy_df, pd.read_csv(path_gas)], ignore_index=True)
    # Adds all except Electricity:Facility
    energy_df = energy_df[energy_df['Name'] != "Electricity:Facility"].groupby(['datapoint_id']).sum()
    energy_df = energy_df.agg(lambda x: x / (floor_sq * 1000000))
    energy_df = energy_df.drop(['KeyValue'], axis=1)
    energy_df = clean_data(energy_df)

    energy_df = energy_df.reset_index()
    # TODO: Investigate if this can be optimized
    energy_df = energy_df.melt(id_vars=['datapoint_id'], var_name='Timestamp', value_name='energy')
    energy_df["date_int"] = energy_df['Timestamp'].apply(lambda r : datetime.strptime(r, '%Y-%m-%d %H:%M'))

    energy_df["date_int"] = energy_df["date_int"].apply(lambda r : r.strftime("%m%d"))
    energy_df["date_int"] = energy_df["date_int"].apply(lambda r : int(r))
    energy_df = energy_df.groupby(['datapoint_id','date_int'])['energy'].agg(lambda x: x.sum()).reset_index()

    return energy_df


def groupsplit(X, y, valsplit, random_seed=42):
    """
    Used to split the dataset by datapoint_id into train and test sets.
    The data is split to ensure all datapoints for each datapoint_id occurs completely in the respective dataset split.
    Note that where there is validation set, data is split with 80% for training and 20% for test set.
    Otherwise, the test set is split further with 60% as test set and 40% as validation set.

    Args:
        X: data excluding the target_variable
        y: target variable with datapoint_id
        valsplit: flag to indicate if there is a dataframe for the validation set. Accepeted values are "yes" or "no"
        random_seed: random seed to be passed for when splitting the data

    Returns:
       X_train: X trainset
       y_train: y trainset
       X_test: X testset
       y_test_complete: Dataframe containing the target variable with corresponding datapointid
    """
    logger.info("groupsplit with valsplit: %s", valsplit)
    if valsplit == 'yes':
        gs = GroupShuffleSplit(n_splits=2, train_size=.7, random_state=random_seed)
    else:
        gs = GroupShuffleSplit(n_splits=2, test_size=.4, random_state=random_seed)

    train_ix, test_ix = next(gs.split(X, y, groups=X.datapoint_id))

    X_train = X.loc[train_ix]
    y_train = y.loc[train_ix]
    X_test = X.loc[test_ix]
    y_test_complete = y.loc[test_ix]

    return X_train, y_train, X_test, y_test_complete


def create_dataset(energy_daily_df, val_df, valsplit, random_seed):
    """
    Used to split the dataset by datapoint_id into train , test and validation sets.

    Args:
        energy_daily_df: the merged dataframe for simulation I/O, weather, and hourly energy file.
        val_df: the merged dataframe for simulation I/O, weather, and hourly energy file validation set. Where there is no validation set, its value is null
        valsplit: flag to indicate if there is a dataframe for the validation set. Accepeted values are "yes" or "no"
        random_seed: random seed to be passed for when splitting the data

    Returns:
       X_train: X trainset
       y_train: y trainset
       X_test: X testset
       y_test_complete: Dataframe containing the target variable with corresponding datapointid
       X_validate: X validate set
       y_validate: y validate set
       y_validate_complete: Dataframe containing the target variable with corresponding datapointid for the validation set
    """
    logger.info("train_test_split with valsplit: %s", valsplit)
    drop_list= ['index', ':datapoint_id', 'level_0', 'index', 'date_int', ':datapoint_id']

    #split to train and test datasets
    y = energy_daily_df[['energy', 'datapoint_id', 'Total Energy']]
    X = energy_daily_df.drop(['energy'], axis = 1)
    X_train, y_train, X_test, y_test_complete = groupsplit(X, y, valsplit, random_seed)
    y_test = y_test_complete[['energy','datapoint_id']]

    if valsplit == 'yes' :
        y_validate_complete = val_df[['energy', 'datapoint_id', 'Total Energy']]
        X_val = val_df.drop(['energy'], axis=1)
        y_validate = y_validate_complete[['energy', 'datapoint_id']]
        X_validate = X_val.drop(drop_list, axis=1)
        X_validate = X_validate.drop(['datapoint_id', 'Total Energy'], axis=1)
    else:
        y_test = y_test_complete
        X_test = X_test.reset_index(drop=True)
        y_test = y_test.reset_index(drop=True)
        X_test, y_test, X_validate, y_validate_complete = groupsplit(X_test, y_test, valsplit, random_seed)
        y_test_complete = y_test
        y_validate = y_validate_complete[['energy', 'datapoint_id']]
        y_test = y_test[['energy', 'datapoint_id']]
        # TODO: Move to central call since many duplicate/uneeded lines
        X_validate = X_validate.drop(drop_list, axis=1)
        X_validate = X_validate.drop(['datapoint_id', 'Total Energy'], axis=1)
    # TODO: Remove since not needed
    energy_daily_df = energy_daily_df.drop(drop_list, axis=1)

    X_train = X_train.drop(drop_list, axis=1)
    X_test = X_test.drop(drop_list, axis=1)
    y_train = y_train['energy']
    X_train = X_train.drop(['datapoint_id', 'Total Energy'], axis=1)
    X_test = X_test.drop(['datapoint_id', 'Total Energy'], axis=1)

    return X_train, X_test, y_train, y_test, y_test_complete, X_validate, y_validate, y_validate_complete

def generate_only_samples(energy_daily_df, output_path, ohe_file):
    """
    Used to split the dataset by datapoint_id into a single dataset to then be used passed through to a stored model.

    Args:
        energy_daily_df: the merged dataframe for simulation I/O, weather, and hourly energy file.
        output_path: Where the output files should be placed.
        ohe_file: Location of an OHE file which has already been saved from training.

    Returns:
       X: dataset containing preprocessed samples without labels
    """
    logger.info("Generating dataset to get predictions for.")
    return categorical_encode(energy_daily_df, None, None, output_path, ohe_file)

def categorical_encode(x_train, x_test, x_validate, output_path, ohe_file=''):
    """
    Used to encode the categorical variables contained in the x_train, x_test and x_validate
    Note that the encoded data return creates additional columns equivalent to the unique categorical values in the each categorical column.

    Args:
         X_train: X trainset
         X_test:  X testset
         X_validate: X validation set
         output_path: Where the output files should be placed.
         ohe_file: Location of an OHE file which has already been saved from training.

    Returns:
        X_train_oh: encoded X trainset
        X_test_oh: encoded X testset
        x_val_oh: encoded X validation set
        all_features: all features after encoding.
    """
    logger.info("Encoding any categorical features.")
    # extracting the categorical columns
    cat_cols = x_train.select_dtypes(include=['object']).columns
    other_cols = x_train.drop(columns=cat_cols).columns
    logger.info("categorical encode: %s", cat_cols)
    # Create the encoder.
    ct = ColumnTransformer([('ohe', OneHotEncoder(sparse=False, handle_unknown="ignore"), cat_cols)], remainder=MinMaxScaler())
    # Load and apply an OHE
    if len(ohe_file) > 0:
        ct = joblib.load(ohe_file)
        x_train_oh = ct.transform(x_train)
    else:
        x_train_oh = ct.fit_transform(x_train)
    # Set default values in case only a set is passed under x_train
    x_test_oh, x_val_oh = np.array([]), np.array([])
    if x_test is not None:
        x_test_oh = ct.transform(x_test)
        x_val_oh = ct.transform(x_validate)
    encoded_cols = ct.named_transformers_.ohe.get_feature_names(cat_cols)
    all_features = np.concatenate([encoded_cols, other_cols])
    # Output the OHE file if a new OHE has been fit
    if len(ohe_file) < 1:
        joblib.dump(ct, output_path + "/" + config.Settings().APP_CONFIG.OHE_FILENAME)
    return x_train_oh, x_test_oh, x_val_oh, all_features

def main(config_file: str = typer.Argument(..., help="Location of the .yml config file (default name is input_config.yml)."),
         hourly_energy_electric_file: str = typer.Option("", help="Location and name of a electricity energy file to be used if the config file is not used."),
         building_params_electric_file: str = typer.Option("", help="Location and name of a electricity building parameters file to be used if the config file is not used."),
         weather_file: str = typer.Argument("", help="Location and name of a .parquet weather file to be used."),
         val_hourly_energy_file: str = typer.Option("", help="Location and name of a electricity energy validation file to be used if the config file is not used."),
         val_building_params_file: str = typer.Option("", help="Location and name of a electricity building parameters validation file to be used if the config file is not used."),
         hourly_energy_gas_file: str = typer.Option("", help="Location and name of a gas energy file to be used if the config file is not used."),
         building_params_gas_file: str = typer.Option("", help="Location and name of a gas building parameters file to be used if the config file is not used."),
         output_path: str = typer.Option("", help="The output path to be used. Note that this value should be empty unless this file is called from a pipeline."),
         preprocess_only_for_predictions: bool = typer.Option(False, help="True if the data to be preprocessed is to be used for prediction, not for training."),
         random_seed: int = typer.Option(42, help="The random seed to be used when splitting the data."),
         building_params_folder: str = typer.Option("", help="The folder location containing all building parameter files which will have predictions made on by the provided model. Only used preprocess_only_for_predictions is True."),
         start_date: str = typer.Option("1-1", help="The start date to specify the start of which weather data is attached to the building data. Expects the input to be in the form Month_number-Day_number. Only used preprocess_only_for_predictions is True."),
         end_date: str = typer.Option("12-31", help="The end date to specify the end of which weather data is attached to the building data. Expects the input to be in the form Month_number-Day_number. Only used preprocess_only_for_predictions is True."),
         ohe_file: str = typer.Option("", help="Location and name of a ohe.pkl OneHotEncoder file which was generated in the root of a training output folder. To be used when generating a dataset to obtain predictions for."),
         cleaned_columns_file: str = typer.Option("", help="Location and name of a cleaned_columns.json file which was generated in the root of a training output folder. To be used when generating a dataset to obtain predictions for."),
        ):
    """
    Used to encode the categorical variables contained in the x_train, x_test and x_validate
    Note that the encoded data return creates additional columns equivalent to the unique categorical values in the each categorical column.

    Args:
        config_file: Location of the .yml config file (default name is input_config.yml).
        hourly_energy_electric_file: Location and name of a electricity energy file to be used if the config file is not used.
        building_params_electric_file: Location and name of a electricity building parameters file to be used if the config file is not used.
        weather_file: Location and name of a .parquet weather file to be used.
        val_hourly_energy_file: Location and name of a electricity energy validation file to be used if the config file is not used.
        val_building_params_file: Location and name of a electricity building parameters validation file to be used if the config file is not used.
        hourly_energy_gas_file: Location and name of a gas energy file to be used if the config file is not used.
        building_params_gas_file: Location and name of a gas building parameters file to be used if the config file is not used.
        output_path: Where output data should be placed. Note that this value should be empty unless this file is called from a pipeline.
        preprocess_only_for_predictions: True if the data to be preprocessed is to be used for prediction, not for training.
        random_seed: The random seed to be used when splitting the data.
        building_params_folder: The folder location containing all building parameter files which will have predictions made on by the provided model. Only used preprocess_only_for_predictions is True.
        start_date: The start date to specify the start of which weather data is attached to the building data. Expects the input to be in the form Month_number-Day_number. Only used preprocess_only_for_predictions is True.
        end_date: The end date to specify the end of which weather data is attached to the building data. Expects the input to be in the form Month_number-Day_number. Only used preprocess_only_for_predictions is True.
        ohe_file: Location and name of a ohe.pkl OneHotEncoder file which was generated in the root of a training output folder. To be used when generating a dataset to obtain predictions for.
        cleaned_columns_file: Location and name of a cleaned_columns.json file which was generated in the root of a training output folder. To be used when generating a dataset to obtain predictions for.
    """
    logger.info("Beginning the weather, energy, and building preprocessing step.")
    DOCKER_INPUT_PATH = config.Settings().APP_CONFIG.DOCKER_INPUT_PATH
    # Load all content stored from the config file, if provided
    if len(config_file) > 0:
        # Load the specified config file
        cfg = config.get_config(DOCKER_INPUT_PATH + config_file)
        # Load the stored building parameter filenames for the train and val sets
        building_params = cfg.get(config.Settings().APP_CONFIG.BUILDING_PARAM_FILES)
        building_params_electric_file, building_params_gas_file = building_params[0], building_params[1]
        val_building_params_file = cfg.get(config.Settings().APP_CONFIG.VAL_BUILDING_PARAM_FILES)[0]
        # Load the stored hourly energy filenames for the train and val sets
        energy_params = cfg.get(config.Settings().APP_CONFIG.ENERGY_PARAM_FILES)
        hourly_energy_electric_file, hourly_energy_gas_file = energy_params[0], energy_params[1]
        val_hourly_energy_file = cfg.get(config.Settings().APP_CONFIG.VAL_ENERGY_PARAM_FILES)[0]
        # Load the folder directory of building files and the start/end timespans if they are used
        if preprocess_only_for_predictions:
            building_params_folder = cfg.get(config.Settings().APP_CONFIG.BUILDING_BATCH_PATH)
            start_date = cfg.get(config.Settings().APP_CONFIG.SIMULATION_START_DATE)
            end_date = cfg.get(config.Settings().APP_CONFIG.SIMULATION_END_DATE)
    # If the output path is blank, map to the docker output path
    if len(output_path) < 1:
        output_path = config.Settings().APP_CONFIG.DOCKER_OUTPUT_PATH

    # For each mandatory input file, append the docker input path
    hourly_energy_electric_file = DOCKER_INPUT_PATH + hourly_energy_electric_file
    building_params_electric_file = DOCKER_INPUT_PATH + building_params_electric_file
    # Since the weather file may already be a full path from a pipeline, check if the input path is needed
    if os.path.exists(DOCKER_INPUT_PATH + weather_file): weather_file = DOCKER_INPUT_PATH + weather_file
    # For each optional file, add the prefix only if it is not empty
    if len(building_params_folder) > 0: building_params_folder = DOCKER_INPUT_PATH + building_params_folder
    if len(val_hourly_energy_file) > 0: val_hourly_energy_file = DOCKER_INPUT_PATH + val_hourly_energy_file
    if len(val_building_params_file) > 0: val_building_params_file = DOCKER_INPUT_PATH + val_building_params_file
    if len(hourly_energy_gas_file) > 0: hourly_energy_gas_file = DOCKER_INPUT_PATH + hourly_energy_gas_file
    if len(building_params_gas_file) > 0: building_params_gas_file = DOCKER_INPUT_PATH + building_params_gas_file
    if len(ohe_file) > 0: ohe_file = DOCKER_INPUT_PATH + ohe_file

    # Load the weather data from the specified path
    logger.info("Loading and preparing the weather file %s.", weather_file)
    weather_df = read_weather(weather_file)
    # Building parameters (electric - mandatory, gas - optional)
    logger.info("Loading and preparing the building file(s).")
    # If the preprocessing is only being done for the building and weather files, begin preparing the output,
    # skipping the preprocessing steps used for training
    if preprocess_only_for_predictions:
        btap_df = process_building_files_batch(building_params_folder, start_date, end_date)
        btap_df_ids = btap_df[["Prediction Identifier", "date_int"]]
        btap_df = btap_df.drop(['floor_sq', ':datapoint_id', "Prediction Identifier", "Total Energy"], axis='columns', errors='ignore')
        # Keep only the columns used when training alongside the date_int column
        with open(cleaned_columns_file, 'r', encoding='UTF-8') as cleaned_columns_f:
            cleaned_columns = json.load(cleaned_columns_f)["columns"]
            cleaned_columns.append('date_int')
            btap_df = btap_df[cleaned_columns]
        #drop_list= ['index', ':datapoint_id', 'level_0', 'index', 'date_int', ':datapoint_id', 'Total Energy']
        btap_df = pd.merge(btap_df, weather_df, on='date_int', how='left').reset_index()
        btap_df = btap_df.drop(['date_int', 'index'], axis='columns')
        X, _, _, all_features = generate_only_samples(btap_df, output_path, ohe_file)
        return X, btap_df_ids, all_features
    # Otherwise, load the file(s) to be used for training
    else:
        btap_df, floor_sq = process_building_files(building_params_electric_file, building_params_gas_file)
        # Save the column names from the cleaning process
        column_path = Path(output_path).joinpath(config.Settings().APP_CONFIG.CLEANED_COLUMNS_FILENAME)
        with open(str(column_path), 'w', encoding='utf8') as json_output:
            json.dump({'columns': [col for col in btap_df.columns if col not in ["Total Energy", ":datapoint_id"]]}, json_output)
        # Hourly energy consumption (electric - mandatory, gas - optional)
        logger.info("Loading and preparing the energy file(s).")
        # Merge the building parameters with the hourly energy consuption
        btap_df = pd.merge(process_hourly_energy(hourly_energy_electric_file, hourly_energy_gas_file, floor_sq),
                                     btap_df,
                                     left_on=['datapoint_id'],
                                     right_on=[':datapoint_id'],
                                     how='left').reset_index()
        logger.info("NA values in btap_df after merging with energy:\n%s", btap_df.isna().any())
        # Derive a daily consumption dataframe between the weather and hourly energy consumption
        btap_df = pd.merge(btap_df, weather_df, on='date_int', how='left').reset_index()
    logger.info("NA values in btap_df after merging with weather:\n%s", btap_df.isna().any())
    # Proceed normally to construct the train/test/val sets only if the data will be used for training
    if val_building_params_file:
        logger.info("Loading files to be used for the validation set.")
        btap_df_val, floor_sq = process_building_files(val_building_params_file, '')
        btap_df_val = pd.merge(process_hourly_energy(val_hourly_energy_file, '', floor_sq),
                               btap_df_val,
                               left_on=['datapoint_id'],
                               right_on=[':datapoint_id'],
                               how='left').reset_index()
        logger.info("NA values in btap_df_val after merging with energy:\n%s", btap_df_val.isna().any())
        btap_df_val = pd.merge(btap_df_val, weather_df, on='date_int', how='left').reset_index()
        logger.info("NA values in btap_df_val after merging with weather:\n%s", btap_df_val.isna().any())
        X_train, X_test, y_train, y_test, y_test_complete, X_validate, y_validate, y_validate_complete = create_dataset(btap_df, btap_df_val, 'yes', random_seed)
    else:
        energy_hour_df_val= '' ; btap_df_val =''; energy_daily_df_val=''
        X_train, X_test, y_train, y_test, y_test_complete, X_validate, y_validate, y_validate_complete = create_dataset(btap_df, btap_df_val, 'no', random_seed)
    # Otherwise generate the samples to get predictions for
    # Encode any categorical features
    X_train_oh, X_test_oh, X_val_oh, all_features = categorical_encode(X_train, X_test, X_validate, output_path)

    logger.info("Preparing json file of the train/test/validation sets.")
    #Creates `data` structure to save and share train and test datasets.
    data = {
            'features': all_features.tolist(),
            'y_train': y_train.to_json(orient='values'),
            'X_train': X_train_oh.tolist(),
            'X_test': X_test_oh.tolist(),
            'y_test': y_test.values.tolist(),
            'y_test_complete': y_test_complete.values.tolist(),
            'X_validate': X_val_oh.tolist(),
            'y_validate': y_validate.values.tolist(),
            'y_validate_complete': y_validate_complete.values.tolist()}

    # Before saving, ensure that the directory exists
    # Bucket used to store preprocessing data.
    preprocessing_path = Path(output_path).joinpath(config.Settings().APP_CONFIG.PREPROCESSING_BUCKET_NAME)

    # Make sure the bucket for preprocessing data exists to avoid write errors
    logger.info("Creating directory %s.", str(preprocessing_path))
    config.create_directory(str(preprocessing_path))

    # Save the json file
    output_file = str(preprocessing_path.joinpath(config.Settings().APP_CONFIG.PREPROCESSING_FILENAME + ".json"))
    logger.info("Saving json file %s.", output_file)
    with open(output_file, 'w', encoding='utf8') as json_output:
        json.dump(data, json_output)

    logger.info("Preprocessing file has been saved as %s.", output_file)
    return output_file

    try:
        pl.target_plot(y_train, y_test)
        pl.corr_plot(btap_df)
    except ValueError as ve:
        logger.error("Unable to produce plots. Plotting threw an exception: %s", ve)
    except matplotlib.units.ConversionError as ce:
        logger.error("Unable to produce plots. matplotlib conversion error: %s", ce)

if __name__ == '__main__':
    # Load settings from the environment
    settings = config.Settings()
    # Run the CLI interface
    typer.run(main)
