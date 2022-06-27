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
from models.training_model import TrainingModel

# Get a log handler
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def main(config_file: str = typer.Argument(..., help="Location of the .yml config file (default name is input_config.yml)."),
         random_seed: int = typer.Option(-1, help="The random seed to be used when training. Should not be -1 when used through the CLI."),
        # Weather
         weather_key: str = typer.Option("", help="The epw file key to be used (only the key, not the full GitHub repository link)."),
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
        weather_key: The epw file key to be used (only the key, not the full GitHub repository link).
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
    INPUT_CONFIG_FILENAME = "input_config.yml"
    # Load the settings
    settings = config.Settings()
    # Begin by loading the config file, if passed, to overwrite
    # blank argument values
    if len(config_file) > 0:
        cfg = config.get_config(DOCKER_INPUT_PATH + config_file)
        if random_seed < 0: random_seed = cfg.get(config.Settings().APP_CONFIG.RANDOM_SEED)
        if weather_key == "": weather_key = cfg.get(config.Settings().APP_CONFIG.WEATHER_KEY)
        # If the energy or building electricity files are not provided, load the files
        if hourly_energy_electric_file == "":
            hourly_energy_electric_file = cfg.get(config.Settings().APP_CONFIG.ENERGY_PARAM_FILES)[0]
            building_params_electric_file = cfg.get(config.Settings().APP_CONFIG.BUILDING_PARAM_FILES)[0]
        if hourly_energy_gas_file == "":
            hourly_energy_gas_file = cfg.get(config.Settings().APP_CONFIG.ENERGY_PARAM_FILES)[1]
            building_params_gas_file = cfg.get(config.Settings().APP_CONFIG.BUILDING_PARAM_FILES)[1]
        if val_hourly_energy_file == "": val_hourly_energy_file = cfg.get(config.Settings().APP_CONFIG.VAL_ENERGY_PARAM_FILE)
        if val_building_params_file == "": val_building_params_file = cfg.get(config.Settings().APP_CONFIG.VAL_BUILDING_PARAM_FILE)
        if estimator_type == "": estimator_type = cfg.get(config.Settings().APP_CONFIG.ESTIMATOR_TYPE)
        if perform_param_search == "": perform_param_search = cfg.get(config.Settings().APP_CONFIG.PARAM_SEARCH)
    # Since the weather key should be a list, set to be a list
    # if it is only a string
    if isinstance(weather_key, str):
        weather_key = [weather_key]
    # Validate all input arguments before continuing
    # Program will output an error if validation fails
    input_model = TrainingModel(input_prefix=DOCKER_INPUT_PATH,
                                config_file=config_file,
                                random_seed=random_seed,
                                epw_file=weather_key,
                                weather_file=weather_file,
                                skip_weather_generation=skip_weather_generation,
                                building_param_files=[building_params_electric_file,
                                                      building_params_gas_file],
                                energy_param_files=[hourly_energy_electric_file,
                                                    hourly_energy_gas_file],
                                val_hourly_energy_file=val_hourly_energy_file,
                                val_building_params_file=val_building_params_file,
                                skip_file_preprocessing=skip_file_preprocessing,
                                preprocessed_data_file=preprocessed_data_file,
                                estimator_type=estimator_type,
                                skip_feature_selection=skip_feature_selection,
                                selected_features_file=selected_features_file,
                                perform_param_search=perform_param_search,
                                skip_model_training=skip_model_training)

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
        shutil.copy(DOCKER_INPUT_PATH + config_file, str(output_path.joinpath(INPUT_CONFIG_FILENAME)))
    output_path = str(output_path)
    # Prepare weather (perhaps can allow a .csv to require processing while .parquet skips processing)
    if not skip_weather_generation:
        input_model.weather_file = prepare_weather.main(config_file=input_model.config_file,
                                                        epw_file=input_model.epw_file,
                                                        output_path=output_path)
    # Preprocess the data (generates json with train, test, validate)
    if not skip_file_preprocessing:
        input_model.preprocessed_data_file = preprocessing.main(config_file=input_model.config_file,
                                                                hourly_energy_electric_file=input_model.energy_param_files[0],
                                                                building_params_electric_file=input_model.building_param_files[0],
                                                                weather_file=input_model.weather_file,
                                                                val_hourly_energy_file=input_model.val_hourly_energy_file,
                                                                val_building_params_file=input_model.val_building_params_file,
                                                                hourly_energy_gas_file=input_model.energy_param_files[1],
                                                                building_params_gas_file=input_model.building_param_files[1],
                                                                output_path=output_path,
                                                                preprocess_only_for_predictions=False,
                                                                random_seed=input_model.random_seed,
                                                                building_params_folder='',
                                                                start_date='',
                                                                end_date='',
                                                                ohe_file='',
                                                                cleaned_columns_file='')
    # Perform feature selection (retrieve the features to be used)
    if not skip_feature_selection:
        input_model.selected_features_file = feature_selection.main(input_model.config_file,
                                                                    input_model.preprocessed_data_file,
                                                                    input_model.estimator_type,
                                                                    output_path)
    # Perform the training and output the model
    if not skip_model_training:
        model_path, train_results = predict.main(input_model.config_file,
                                                 input_model.preprocessed_data_file,
                                                 input_model.selected_features_file,
                                                 input_model.perform_param_search,
                                                 output_path,
                                                 input_model.random_seed)
    # Provide any additional outputs/plots as needed
    #...
    logger.info("Training process has been completed.")
    return

if __name__ == '__main__':
    # Load settings from the environment
    settings = config.Settings()
    # Run the CLI interface
    typer.run(main)
