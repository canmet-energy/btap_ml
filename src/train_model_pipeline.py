"""
Given all data files, preprocess the data and train a model with the preprocessed data

CLI arguments match those defined by ``main()``.
"""
import logging
import os
import shutil
from datetime import datetime
from pathlib import Path

import typer
import yaml

import config
import feature_selection
import predict
import prepare_weather
import preprocessing

# Get a log handler
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def load_and_validate_config(config_file: str):
    """
    TODO: Decide on how this will work
    Loads the specified yaml config file for the program and validates that each component has been loaded
    """
    # Load the config
    with open(config_file, 'r') as f:
        contents = yaml.safe_load(f)
        weather_key = settings.APP_CONFIG.WEATHER_KEY
        epw_files = contents.get(weather_key)
        # If default is empty string, change, otherwise keep?
    return

def main(config_file: str = typer.Argument(..., help="Location of the .yml config file (default name is input_config.yml)."),
         random_seed: int = typer.Option(7, help="The random seed to be used when training."),
        # Weather
         weather_file: str = typer.Option("", help="Location and name of a .parquet weather file to be used if weather generation is skipped."),
         skip_weather_generation: bool = typer.Option(False, help="True if the .parquet weather file generation should be skipped, where the weather_file input is used, False if the weather file generation should be performed."),
        # Preprocessing
         hourly_energy_electric_file: str = typer.Option("", help="Location and name of a electricity energy file to be used if the config file is not used."),
         building_params_electric_file: str = typer.Option("", help="Location and name of a electricity building parameters file to be used if the config file is not used."),
         val_hourly_energy_file: str = typer.Option("", help="Location and name of a electricity energy validation file to be used if the config file is not used."),
         val_building_params_file: str = typer.Option("", help="Location and name of a electricity building parameters validation file to be used if the config file is not used."),
         hourly_energy_gas_file: str = typer.Option("", help="Location and name of a gas energy file to be used if the config file is not used."),
         building_params_gas_file: str = typer.Option("", help="Location and name of a gas building parameters file to be used if the config file is not used."),
         skip_file_preprocessing: bool = typer.Option(False, help="True if the .json preprocessing file generation should be skipped, where the preprocessed_data_file input is used, False if the preprocessing file generation should be performed."),
        # Feature selection
         preprocessed_data_file: str = typer.Option("", help="Location and name of a .json preprocessing file to be used if the preprocessing is skipped."),
         estimator_type: str = typer.Option("", help="The type of feature selection to be performed. The default is lasso, which will be used if nothing is passed. The other options are 'linear', 'elasticnet', and 'xgb'."),
         skip_feature_selection: bool = typer.Option(False, help="True if the .json feature selection file generation should be skipped, where the selected_features_file input is used, False if the feature selection file generation should be performed."),
        # Training
         selected_features_file: str = typer.Option("", help="Location and name of a .json feature selection file to be used if the feature selection is skipped."),
         perform_param_search: str = typer.Option("no", help="'yes' if hyperparameter tuning should be performed (increases runtime), 'no' if the default hyperparameters should be used."),
         skip_model_training: bool = typer.Option(False, help="True if the model training should be skipped. Useful if only the preprocessing steps should be performed."),
         ) -> None:
    """
    Run through the entire training pipeline to train a surrogate Machine Learning model which predicts energy output.
    All outputs will be placed within a folder created in the specified output path which uniquely uses the datetime for naming the folder.
    First, the specified weather file will be retrieved and converted into a .parquet file.
    Second, the provided input building and energy files will be preprocessed and split into train/test/validation sets.
    Third, the best features from the input data will be selected to be used for training.
    Finally, a Machine Learning model will be instantiated and trained with the preprocessed data and selected features.
    Note that all inputs except for the config file are optional, however the arguements can be set from the command line
    if an empty string is passed as input for the config .yml file.

    Args:
        config_file: Location of the .yml config file (default name is input_config.yml).
        random_seed: Random seed to be used when training.
        weather_file: Location and name of a .parquet weather file to be used if weather generation is skipped.
        skip_weather_generation: True if the .parquet weather file generation should be skipped,
                                 where the weather_file input is used, False if the weather file generation should be performed.
        hourly_energy_electric_file: Location and name of a electricity energy file to be used if the config file is not used.
        building_params_electric_file: Location and name of a electricity building parameters file to be used if the config file is not used.
        val_hourly_energy_file: Location and name of a electricity energy validation file to be used if the config file is not used.
        val_building_params_file: Location and name of a electricity building parameters validation file to be used if the config file is not used.
        hourly_energy_gas_file: Location and name of a gas energy file to be used if the config file is not used.
        building_params_gas_file: Location and name of a gas building parameters file to be used if the config file is not used.
        skip_file_preprocessing: True if the .json preprocessing file generation should be skipped,
                                 where the preprocessed_data_file input is used, False if the preprocessing file generation should be performed.
        preprocessed_data_file: Location and name of a .json preprocessing file to be used if the preprocessing is skipped.
        estimator_type: The type of feature selection to be performed. The default is lasso, which will be used if nothing is passed.
                        The other options are 'linear', 'elasticnet', and 'xgb'.
        skip_feature_selection: True if the .json feature selection file generation should be skipped,
                                where the selected_features_file input is used, False if the feature selection file generation should be performed.
        selected_features_file: Location and name of a .json feature selection file to be used if the feature selection is skipped.
        perform_param_search: 'yes' if hyperparameter tuning should be performed (increases runtime), 'no' if the default hyperparameters should be used.
        skip_model_training: True if the model training should be skipped. Useful if only the preprocessing steps should be performed.
    """
    logger.info("Beginning the training process.")
    DOCKER_INPUT_PATH = config.Settings().APP_CONFIG.DOCKER_INPUT_PATH
    # Load the settings
    settings = config.Settings()
    # Load and validate anything from the config file
    if len(config_file) > 0:
        logger.info("Loading provided configuration file.")
        load_and_validate_config(DOCKER_INPUT_PATH + config_file)
        cfg = config.get_config(DOCKER_INPUT_PATH + config_file)
    # Create directory to hold all data for the run (datetime/...)
    # If used, copy the config file within the directory to log the input values
    output_path = config.Settings().APP_CONFIG.DOCKER_OUTPUT_PATH
    output_path = Path(output_path).joinpath(settings.APP_CONFIG.TRAIN_BUCKET_NAME + str(datetime.now()))
    # Create the root directory in the mounted drive
    logger.info("Creating output directory %s.", str(output_path))
    config.create_directory(str(output_path))
    # If the config file is used, copy it into the output folder
    logger.info("Copying config file into %s.", str(output_path))
    if len(config_file) > 0:
        shutil.copy(DOCKER_INPUT_PATH + config_file, str(output_path.joinpath("input_config.yml")))
    output_path = str(output_path)
    # Prepare weather (perhaps can allow a .csv to require processing while .parquet skips processing)
    if not skip_weather_generation:
        weather_file = prepare_weather.main(config_file, output_path=output_path)
    # Preprocess the data (generates json with train, test, validate)
    if not skip_file_preprocessing:
        preprocessed_data_file = preprocessing.main(config_file=config_file, hourly_energy_electric_file=hourly_energy_electric_file,
                                                    building_params_electric_file=building_params_electric_file,
                                                    weather_file=weather_file, val_hourly_energy_file=val_hourly_energy_file,
                                                    val_building_params_file=val_building_params_file, hourly_energy_gas_file=hourly_energy_gas_file,
                                                    building_params_gas_file=building_params_gas_file, output_path=output_path,
                                                    preprocess_only_for_predictions=False, random_seed=random_seed,
                                                    building_params_folder='', start_date='', end_date='', ohe_file='', cleaned_columns_file='')
    # Perform feature selection (retrieve the features to be used)
    if not skip_feature_selection:
       selected_features_file = feature_selection.main(config_file, preprocessed_data_file, estimator_type, output_path)
    # Perform the training and output the model
    if not skip_model_training:
        model_path, train_results = predict.main(config_file, preprocessed_data_file, selected_features_file, perform_param_search, output_path, random_seed)
    # Provide any additional outputs/plots
    ...
    logger.info("Training process has been completed")
    return

if __name__ == '__main__':
    # Load settings from the environment
    settings = config.Settings()
    # Run the CLI interface
    typer.run(main)
