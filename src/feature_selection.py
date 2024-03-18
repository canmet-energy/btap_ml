"""
Select features that are used to build the surrogate mode.
"""
import json
import logging
import os
from pathlib import Path

import numpy as np
import pandas as pd
import typer
import xgboost as xgb
from sklearn import linear_model
from sklearn.ensemble import RandomForestRegressor
from sklearn.feature_selection import RFECV
from sklearn.linear_model import ElasticNetCV, Lasso, LassoCV, LinearRegression
from sklearn.model_selection import KFold
from sklearn.preprocessing import (MaxAbsScaler, MinMaxScaler, Normalizer,
                                   PowerTransformer, QuantileTransformer,
                                   RobustScaler, StandardScaler, minmax_scale)

import config
from models.feature_selection_model import FeatureSelectionModel

############################################################
# feature selection
############################################################

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def select_features(preprocessed_data_file, estimator_type, output_path):
    """
    Select the feature which contribute most to the prediction for the energy and costing values.

    Args:
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

    # Scale the inputs before selecting the featueres
    scalerx = StandardScaler()
    #scalerx = QuantileTransformer(output_distribution="normal", random_state=42)#RobustScaler()#QuantileTransformer(output_distribution="normal", random_state=42)#StandardScaler()#MinMaxScaler()#RobustScaler()
    X_train = scalerx.fit_transform(preprocessed_dataset["X_train"])
    y_train = pd.read_json(preprocessed_dataset["y_train"], orient='values').values#.ravel()
    # Ignore the total value within the loaded file, just use the individual outputs
    y_train = y_train[:, 1:]

    # For multiple outputs, only the MultiTaskLassoCV is used, this can be adjusted if more time
    # is available to test other approaches.
    """
    logger.info("Run estimator: %s", estimator_type)
    if estimator_type.lower() == "linear":
        estimator = LinearRegression()
        rfecv = RFECV(estimator=estimator, step=1, cv=KFold(10), scoring='neg_mean_squared_error')
        fit = rfecv.fit(X_train, y_train)
        rank_features_nun = pd.DataFrame(rfecv.ranking_, columns=["rank"], index=preprocessed_dataset["features"])
        selected_features = rank_features_nun.loc[rank_features_nun["rank"] == 1].index.tolist()
    elif estimator_type.lower() == "elasticnet":
        # Takes significant processing time
        reg = ElasticNetCV(n_jobs=-1)
        rfecv = RFECV(estimator=reg, step=1, cv=KFold(10), scoring='neg_mean_squared_error')
        fit = rfecv.fit(X_train, y_train)
        rank_features_nun = pd.DataFrame(rfecv.ranking_, columns=["rank"], index=preprocessed_dataset["features"])
        selected_features = rank_features_nun.loc[rank_features_nun["rank"] == 1].index.tolist()
    elif estimator_type.lower() == "xgb":
        # Takes significant processing time
        estimator = xgb.XGBRegressor(n_jobs=-1)
        rfecv = RFECV(estimator=estimator, step=1, cv=KFold(10), scoring='neg_mean_squared_error')
        fit = rfecv.fit(X_train, y_train)
        rank_features_nun = pd.DataFrame(rfecv.ranking_, columns=["rank"], index=preprocessed_dataset["features"])
        selected_features = rank_features_nun.loc[rank_features_nun["rank"] == 1].index.tolist()
    else:
    """
    # Use LassoCV
    reg = linear_model.MultiTaskLassoCV(cv=10, n_jobs=-1, n_alphas=100, tol=600)
    fit = reg.fit(X_train,y_train)
    score = reg.score(X_train,y_train)
    selected_features = np.array(features)[np.abs(sum(reg.coef_) / len(reg.coef_))  > 0]

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

    features_json = {'features': selected_features.tolist()}

    with open(str(file_path), 'w', encoding='utf8') as json_output:
        json.dump(features_json, json_output)

    return str(file_path)

def main(config_file: str = typer.Argument(..., help="Location of the .yml config file (default name is input_config.yml)."),
         preprocessed_data_file: str = typer.Argument(..., help="Location and name of a .json preprocessing file to be used."),
         estimator_type: str = typer.Option("", help="The type of feature selection to be performed. The default is lasso, which will be used if nothing is passed. The other options are 'linear', 'elasticnet', and 'xgb'."),
         output_path: str = typer.Option("", help="The output path to be used. Note that this value should be empty unless this file is called from a pipeline."),
         ) -> str:
    """
    Select the feature which contribute most to the prediction for the energy or costing values.

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
        if estimator_type == "":
            estimator_type = cfg.get(config.Settings().APP_CONFIG.ESTIMATOR_TYPE)
    # If the output path is blank, map to the docker output path
    if len(output_path) < 1:
        output_path = config.Settings().APP_CONFIG.DOCKER_OUTPUT_PATH
    # Validate all inputs
    input_model = FeatureSelectionModel(input_prefix=DOCKER_INPUT_PATH,
                                        preprocessed_data_file=preprocessed_data_file,
                                        estimator_type=estimator_type)
    # Perform the feature selection
    features_filepath = select_features(input_model.preprocessed_data_file, input_model.estimator_type, output_path)
    return features_filepath

if __name__ == '__main__':
    # Load settings from the environment
    settings = config.Settings()
    # Run the CLI interface
    typer.run(main)
