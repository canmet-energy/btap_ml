"""
Select features that are used to build the surrogate mode.

CLI arguments match those defined by ``main()``.
"""
import argparse
import json
import logging
import os
from pathlib import Path

import numpy as np
import pandas as pd
import s3fs
import typer
import xgboost as xgb
from sklearn import linear_model
from sklearn.ensemble import RandomForestRegressor
from sklearn.feature_selection import RFECV
from sklearn.linear_model import ElasticNetCV, Lasso, LassoCV, LinearRegression
from sklearn.model_selection import KFold
from sklearn.preprocessing import MinMaxScaler, RobustScaler, StandardScaler

import config

############################################################
# feature selection
############################################################

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def select_features(config_file, preprocessed_data_file, estimator_type, output_path):
    """
    Select the feature which contribute most to the prediction for the total energy consumed.
    Default estimator_type used for feature selection is 'LassoCV'

    Args:
        config_file: Location of the .yml config file (default name is input_config.yml).
        preprocessed_data_file: Location and name of a .json preprocessing file to be used if the preprocessing is skipped.
        estimator_type: The type of feature selection to be performed. The default is lasso, which will be used if nothing is passed.
                        The other options are 'linear', 'elasticnet', and 'xgb'.
        output_path: Folder location where output files should be placed.
    """
    with open(preprocessed_data_file, 'r', encoding='UTF-8') as preprocessing_file:
        preprocessed_dataset = json.load(preprocessing_file)

    features = preprocessed_dataset["features"]
    X_train = pd.DataFrame(preprocessed_dataset["X_train"], columns=features)
    X_test = pd.DataFrame(preprocessed_dataset["X_test"], columns=features)

    print(X_train.head(10))

    #standardize
    scalerx= RobustScaler()
    scalery= RobustScaler()
    X_train = scalerx.fit_transform(preprocessed_dataset["X_train"])
    y_train = pd.read_json(preprocessed_dataset["y_train"], orient='values').values.ravel()

    logger.info("Run estimator: %s", estimator_type)
    if estimator_type == "linear":
        estimator = LinearRegression()
        rfecv = RFECV(estimator=estimator, step=1, cv=KFold(10), scoring='neg_mean_squared_error')
        fit = rfecv.fit(X_train, y_train)
        rank_features_nun = pd.DataFrame(rfecv.ranking_, columns=["rank"], index=preprocessed_dataset["features"])
        selected_features = rank_features_nun.loc[rank_features_nun["rank"] == 1].index.tolist()
    elif estimator_type == "elasticnet":
        reg = ElasticNetCV(n_jobs=-1, cv=10)
        fit = reg.fit(X_train, y_train)
        rank_features_nun = pd.DataFrame(rfecv.ranking_, columns=["rank"], index=preprocessed_dataset["features"])
        selected_features = rank_features_nun.loc[rank_features_nun["rank"] == 1].index.tolist()
        score = rfecv.score(X_train, y_train)
        rank_features_nun = pd.DataFrame(reg.coef_, columns=["rank"], index=preprocessed_dataset["features"])
        selected_features = rank_features_nun.loc[abs(rank_features_nun["rank"]) > 0].index.tolist()
    elif estimator_type == "xgb":
        estimator = xgb.XGBRegressor(n_jobs=-1)
        rfecv = RFECV(estimator=estimator, step=1, cv=KFold(10), scoring='neg_mean_squared_error')
        fit = rfecv.fit(X_train, y_train)
        rank_features_nun = pd.DataFrame(rfecv.ranking_, columns=["rank"], index=preprocessed_dataset["features"])
        selected_features = rank_features_nun.loc[rank_features_nun["rank"] == 1].index.tolist()
    else:
        # Use LassoCV
        reg = linear_model.LassoCV(cv=10,n_jobs=-1,n_alphas=100,tol=600)
        fit = reg.fit(X_train,y_train)
        score = reg.score(X_train,y_train)
        rank_features_nun = pd.DataFrame(reg.coef_, columns=["rank"], index = preprocessed_dataset["features"])
        selected_features = rank_features_nun.loc[abs(rank_features_nun["rank"])>0].index.tolist()
        print(score)
        print(len(selected_features))
        print(selected_features)

    # Create the root directory in the mounted drive
    features_bucket_path = Path(output_path).joinpath(config.Settings().APP_CONFIG.FEATURE_SELECTION_BUCKET_NAME)
    output_filename = config.Settings().APP_CONFIG.FEATURE_SELECTION_FILENAME + estimator_type + ".json"
    # Write the data to s3
    file_path = features_bucket_path.joinpath(output_filename)

    # Create the root directory in the mounted drive
    logger.info("Creating directory %s.", str(features_bucket_path))
    config.create_directory(str(features_bucket_path))

    features_json = {'features': selected_features}

    with open(str(file_path), 'w', encoding='utf8') as json_output:
        json.dump(features_json, json_output)

    return str(file_path)

def main(config_file: str = typer.Argument(..., help="Location of the .yml config file (default name is input_config.yml)."),
         preprocessed_data_file: str = typer.Argument(..., help="Location and name of a .json preprocessing file to be used."),
         estimator_type: str = typer.Option("", help="The type of feature selection to be performed. The default is lasso, which will be used if nothing is passed. The other options are 'linear', 'elasticnet', and 'xgb'."),
         output_path: str = typer.Option("", help="The output path to be used. Note that this value should be empty unless this file is called from a pipeline."),
         ) -> str:
    """
    Select the feature which contribute most to the prediction for the total energy consumed.
    Default estimator_type used for feature selection is 'LassoCV'

    Args:
        config_file: Location of the .yml config file (default name is input_config.yml).
        preprocessed_data_file: Location and name of a .json preprocessing file to be used.
        estimator_type: The type of feature selection to be performed. The default is lasso, which will be used if nothing is passed.
                        The other options are 'linear', 'elasticnet', and 'xgb'.
        output_path: Where output data should be placed. Note that this value should be empty unless this file is called from a pipeline.
    """
    DOCKER_INPUT_PATH = config.Settings().APP_CONFIG.DOCKER_INPUT_PATH
    # Load all content stored from the config file, if provided
    if len(config_file) > 0:
        # Load the specified config file
        cfg = config.get_config(DOCKER_INPUT_PATH + config_file)
        estimator_type = cfg.get(config.Settings().APP_CONFIG.ESTIMATOR_TYPE)
    # If the output path is blank, map to the docker output path
    if len(output_path) < 1:
        output_path = config.Settings().APP_CONFIG.DOCKER_OUTPUT_PATH
    # Since the preprocessing file may already be a full path from a pipeline, check if the input path is needed
    if os.path.exists(DOCKER_INPUT_PATH + preprocessed_data_file): preprocessed_data_file = DOCKER_INPUT_PATH + preprocessed_data_file
    # Perform the feature selection
    features_filepath = select_features(config_file, preprocessed_data_file, estimator_type, output_path)
    return features_filepath

if __name__ == '__main__':
    # Load settings from the environment
    settings = config.Settings()
    # Run the CLI interface
    typer.run(main)
