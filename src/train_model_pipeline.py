"""
Given all data files, preprocess the data and train an energy model and a costing model with the preprocessed data
"""
import logging
import os
import shutil
from datetime import datetime
from pathlib import Path

import time

import typer

import config
import feature_selection
import predict
import preprocessing
from models.training_model import TrainingModel

# Get a log handler
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Ensure reproduability when using a GPU
os.environ["TF_CUDNN_DETERMINISTIC"] = "true"
os.environ["TF_DETERMINISTIC_OPS"] = "true"

def main(config_file: str = typer.Argument(..., help="Location of the .yml config file (default name is input_config.yml)."),
         random_seed: int = typer.Option(-1, help="The random seed to be used when training. Should not be -1 when used through the CLI."),
        # Preprocessing
         hourly_energy_electric_file: str = typer.Option("", help="Location and name of a electricity energy file to be used if the config file is not used."),
         building_params_electric_file: str = typer.Option("", help="Location and name of a electricity building parameters file to be used if the config file is not used."),
         val_hourly_energy_file: str = typer.Option("", help="Location and name of an energy validation file to be used if the config file is not used."),
         val_building_params_file: str = typer.Option("", help="Location and name of a building parameters validation file to be used if the config file is not used."),
         hourly_energy_gas_file: str = typer.Option("", help="Location and name of a gas energy file to be used if the config file is not used."),
         building_params_gas_file: str = typer.Option("", help="Location and name of a gas building parameters file to be used if the config file is not used."),
         skip_file_preprocessing: bool = typer.Option(False, help="True if the .json preprocessing file generation should be skipped, where the preprocessed_data_file input is used, False if the preprocessing file generation should be performed."),
         delete_preprocessing_file: bool = typer.Option(True, help="True if the preprocessing output file should be removed after use, False if the preprocessing file should be kept (for analysis or to use in later training runs)."),
        # Feature selection
         preprocessed_data_file: str = typer.Option("", help="Location and name of a .json preprocessing file to be used if the preprocessing is skipped."),
         estimator_type: str = typer.Option("", help="The type of feature selection to be performed. The default is lasso, which will be used if nothing is passed. The other options are 'linear', 'elasticnet', and 'xgb'."),
         skip_feature_selection: bool = typer.Option(False, help="True if the .json feature selection file generation should be skipped, where the selected_features_file input is used, False if the feature selection file generation should be performed."),
        # Training
         selected_features_file: str = typer.Option("", help="Location and name of a .json feature selection file to be used if the feature selection is skipped."),
         skip_model_training: bool = typer.Option(False, help="True if the model training should be skipped. Useful if only the preprocessing steps should be performed."),
         use_updated_model: bool = typer.Option(True, help="True if the larger model architecture should be used for energy training."),
         use_dropout: bool = typer.Option(True, help="True if the regularization technique should be used (on by default). False if tests are desired without dropout. Note that not using dropout may cause bias to learned when training."),
         selected_model_type: str = typer.Option("", help="Type of model selected. can either be 'mlp' or 'rf'"),
         ) -> None:
    """
    Run through the entire training pipeline to train two surrogate Machine Learning models, one to predict energy and one to predict costing.
    The process below is repeated for both the energy and costing training processes.
    First, the energy model will be trained, then the costing model will be trained.
    All outputs will be placed within a folder created in the specified output path which uniquely uses the datetime for naming the folder.
    First, the provided input building files will be loaded.
    Second, for energy training, the energy files will be preprocessed. The weather data will also be collected and processed for energy training.
    Third, the dataset is split into train/test/validation sets.
    Fourth, the best features from the input data will be selected to be used for training.
    Finally, a Machine Learning model will be instantiated and trained with the preprocessed data and selected features.
    Note that all inputs except for the config file are optional, however the arguements can be set from the command line
    if an empty string is passed as input for the config .yml file.

    Args:
        config_file: Location of the .yml config file (default name is input_config.yml).
        random_seed: Random seed to be used when training.
        hourly_energy_electric_file: Location and name of a electricity energy file to be used if the config file is not used.
        building_params_electric_file: Location and name of a electricity building parameters file to be used if the config file is not used.
        val_hourly_energy_file: Location and name of an energy validation file to be used if the config file is not used.
        val_building_params_file: Location and name of a building parameters validation file to be used if the config file is not used.
        hourly_energy_gas_file: Location and name of a gas energy file to be used if the config file is not used.
        building_params_gas_file: Location and name of a gas building parameters file to be used if the config file is not used.
        skip_file_preprocessing: True if the .json preprocessing file generation should be skipped,
                                 where the preprocessed_data_file input is used, False if the preprocessing file generation should be performed.
        delete_preprocessing_file: True if the preprocessing output file should be removed after use, False if the preprocessing file should be
                                   kept (for analysis or to use in later training runs).
        preprocessed_data_file: Location and name of a .json preprocessing file to be used if the preprocessing is skipped.
        estimator_type: The type of feature selection to be performed. The default is lasso, which will be used if nothing is passed.
                        The other options are 'linear', 'elasticnet', and 'xgb'.
        skip_feature_selection: True if the .json feature selection file generation should be skipped,
                                where the selected_features_file input is used, False if the feature selection file generation should be performed.
        selected_features_file: Location and name of a .json feature selection file to be used if the feature selection is skipped.
        perform_param_search: 'yes' if hyperparameter tuning should be performed (increases runtime), 'no' if the default hyperparameters should be used.
        skip_model_training: True if the model training should be skipped. Useful if only the preprocessing steps should be performed.
        use_updated_model: True if the larger model architecture should be used for training for energy training.
        use_dropout: True if the regularization technique should be used (on by default). False if tests are desired without dropout. Note that not using dropout may cause bias to learned when training.
        selected_model_type: Type of model selected. can either be 'mlp' for Multilayer Perceptron or 'rf' for Random Forest
    """
    logger.info("Beginning the training process.")
    DOCKER_INPUT_PATH = config.Settings().APP_CONFIG.DOCKER_INPUT_PATH
    INPUT_CONFIG_FILENAME = "input_config.yml"

    # Load the settings
    settings = config.Settings()

    # Set the perform_param_search parameter to 'yes' to tune the models.
    # Otherwise set it to 'no'.
    perform_param_search = 'no'

    # Begin by loading the config file, if passed, to overwrite
    # blank argument values
    if len(config_file) > 0:

        cfg = config.get_config(DOCKER_INPUT_PATH + config_file)
        if random_seed < 0: random_seed = cfg.get(config.Settings().APP_CONFIG.RANDOM_SEED)
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
        if selected_model_type == "": selected_model_type = cfg.get(config.Settings().APP_CONFIG.SELECTED_MODEL_TYPE)

    # Identify the training processes to be taken and whether the updated model should
    # be used for the specified training (energy and/or costing)
    TRAINING_PROCESSES = [[config.Settings().APP_CONFIG.ENERGY, use_updated_model],
                          [config.Settings().APP_CONFIG.COSTING, use_updated_model]]

    # Create directory to hold all data for the run (datetime/...)
    # If used, copy the config file within the directory to log the input values
    output_path_root = config.Settings().APP_CONFIG.DOCKER_OUTPUT_PATH
    # With Windows, the colon may cause issues depending on how the
    # dependencies work with them, thus they are removed

    output_path_root = Path(output_path_root).joinpath(settings.APP_CONFIG.TRAIN_BUCKET_NAME + str(datetime.now()).replace(":", "-")).joinpath(cfg.get(config.Settings().APP_CONFIG.SELECTED_MODEL_TYPE))

    # Create the root directory
    logger.info("Creating output directory %s.", str(output_path_root))
    config.create_directory(str(output_path_root))



    # If the config file is used, copy it into the output folder
    logger.info("Copying config file into %s.", str(output_path_root))
    if len(config_file) > 0:
        shutil.copy(DOCKER_INPUT_PATH + config_file, str(output_path_root.joinpath(INPUT_CONFIG_FILENAME)))

    # Perform all specified training processes
    for idx, training_process_params in enumerate(TRAINING_PROCESSES):
        training_process = training_process_params[0]
        train_with_updated_model = training_process_params[1]
        # Validate all input arguments before continuing
        # Program will output an error if validation fails
        input_model = TrainingModel(input_prefix=DOCKER_INPUT_PATH,
                                    config_file=config_file,
                                    random_seed=random_seed,
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
        # Define the output path for the current training process
        output_path = output_path_root.joinpath(training_process)

        # Create the root directory in the mounted drive
        logger.info("Creating output directory %s.", str(output_path))
        config.create_directory(str(output_path))

        output_path = str(output_path)
        # Preprocess the data (generates json with train, test, validate)
        if not skip_file_preprocessing:
            start_time = time.time()
            input_model.preprocessed_data_file = preprocessing.main(config_file=input_model.config_file,
                                                                    process_type=training_process,
                                                                    hourly_energy_electric_file=input_model.energy_param_files[0],
                                                                    building_params_electric_file=input_model.building_param_files[0],
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
            print(time.time() - start_time)
        # Perform feature selection (retrieve the features to be used)
        if not skip_feature_selection:
            input_model.selected_features_file = feature_selection.main(input_model.config_file,
                                                                        input_model.preprocessed_data_file,
                                                                        input_model.estimator_type,
                                                                        output_path)
        # Perform the training and output the model
        if not skip_model_training:
            model_path, train_results = predict.main(input_model.config_file,
                                                     training_process,
                                                     input_model.preprocessed_data_file,
                                                     input_model.selected_features_file,
                                                     selected_model_type,
                                                     input_model.perform_param_search,
                                                     output_path,
                                                     input_model.random_seed,
                                                     input_model.building_param_files[0],
                                                     input_model.building_param_files[1],
                                                     input_model.val_building_params_file,
                                                     train_with_updated_model,
                                                     use_dropout,
                                                     idx)

        # If requested, delete the preprocessing file after completion
        if delete_preprocessing_file:
            try:
                os.remove(input_model.preprocessed_data_file)
            except OSError:
                pass
    # Provide any additional outputs/plots as needed
    #...
    logger.info("Training process has been completed.")
    return

if __name__ == '__main__':
    # Load settings from the environment
    settings = config.Settings()
    # Run the CLI interface
    typer.run(main)
