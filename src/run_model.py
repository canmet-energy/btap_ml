"""
Given a specific model and files, output the model's predictions for the files.

CLI arguments match those defined by ``main()``.
"""
import json
import logging
import os
import shutil
from datetime import datetime
from pathlib import Path

import joblib
import pandas as pd
import typer
import yaml
from sklearn.preprocessing import RobustScaler
from tensorflow import keras

import config
import prepare_weather
import preprocessing

# Get a log handler
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def main(config_file: str = typer.Argument(..., help="Location of the .yml config file (default name is input_config.yml)."),
         model_file: str = typer.Option("", help="Location and name of a .h5 trained keras model to be used for training."),
         ohe_file: str = typer.Option("", help="Location and name of a ohe.pkl OneHotEncoder file which was generated in the root of a training output folder."),
         scaler_X_file: str = typer.Option("", help="Location and name of a scaler_X.pkl fit scaler file which was generated with the trained model_file."),
         scaler_y_file: str = typer.Option("", help="Location and name of a scaler_y.pkl fit scaler file which was generated with the trained model_file."),
         weather_file: str = typer.Option("", help="Location and name of a .parquet weather file to be used if weather generation is skipped."),
         skip_weather_generation: bool = typer.Option(False, help="True if the .parquet weather file generation should be skipped, where the weather_file input is used, False if the weather file generation should be performed."),
         building_params_folder: str = typer.Option("", help="The folder location containing all building parameter files which will have predictions made on by the provided model."),
         start_date: str = typer.Option("1-1", help="The start date to specify the start of which weather data is attached to the building data. Expects the input to be in the form Month_number-Day_number."),
         end_date: str = typer.Option("12-31", help="The end date to specify the end of which weather data is attached to the building data. Expects the input to be in the form Month_number-Day_number."),
         selected_features_file: str = typer.Option("", help="Location and name of a .json feature selection file to be used if the feature selection is skipped."),
         ) -> None:
    """
    Preprocess a set of input building files and a weather file to obtain a dataset to obtain daily energy predictions for.
    The feature selection file that has been used with the trained model must be included to appropriately preprocess the data.
    The start/end dates to be spanned are specified within the provided config_file or through the CLI, but it is assumed that
    each day within an arbitrary year will receive predictions.
    A trained Keras model must be provided as input to perform the predictions on the data. These predictions will be output into a
    .csv file which follows the format of the energy files which are used to train the models. The outputs will be for daily energy
    values rather than hourly energy values, where outputs represent the total energy output observed from generated energy files
    from rows without the Electricity:Facility Name.

    Args:
        config_file: Location of the .yml config file (default name is input_config.yml).
        model_file: Location and name of a .h5 trained keras model to be used for training.
        ohe_file: Location and name of a ohe.pkl OneHotEncoder file which was generated in the root of a training output folder.
        scaler_X_file: Location and name of a scaler_X.pkl fit scaler file which was generated with the trained model_file.
        scaler_y_file: Location and name of a scaler_y.pkl fit scaler file which was generated with the trained model_file.
        weather_file: Location and name of a .parquet weather file to be used if weather generation is skipped.
        skip_weather_generation: True if the .parquet weather file generation should be skipped, where the weather_file input is used, False if the weather file generation should be performed.
        building_params_folder: The folder location containing all building parameter files which will have predictions made on by the provided model.
        selected_features_file: Location and name of a .json feature selection file to be used if the feature selection is skipped.
        start_date: The start date to specify the start of which weather data is attached to the building data. Expects the input to be in the form Month_number-Day_number.
        end_date: The end date to specify the end of which weather data is attached to the building data. Expects the input to be in the form Month_number-Day_number.
    """
    settings = config.Settings()
    DOCKER_INPUT_PATH = config.Settings().APP_CONFIG.DOCKER_INPUT_PATH
    TEMP_YEAR = "2022-"
    if len(config_file) > 0:
        #load_and_validate_config(config_file)
        cfg = config.get_config(DOCKER_INPUT_PATH + config_file)
        selected_features_file = cfg.get(settings.APP_CONFIG.FEATURES_FILE)
        model_file = cfg.get(settings.APP_CONFIG.TRAINED_MODEL_FILE)
        scaler_X_file = cfg.get(settings.APP_CONFIG.SCALER_X_FILE)
        scaler_y_file = cfg.get(settings.APP_CONFIG.SCALER_Y_FILE)
        # If the OHE file is empty, the preprocessing will fit a new OneHotEncoder to be used
        ohe_file = cfg.get(settings.APP_CONFIG.OHE_FILE)
        start_date = cfg.get(config.Settings().APP_CONFIG.SIMULATION_START_DATE)
        end_date = cfg.get(config.Settings().APP_CONFIG.SIMULATION_END_DATE)
    # Create directory to hold all data for the run (datetime/...)
    # If used, copy the config file within the directory to log the input values
    output_path = config.Settings().APP_CONFIG.DOCKER_OUTPUT_PATH
    output_path = Path(output_path).joinpath(settings.APP_CONFIG.RUN_BUCKET_NAME + str(datetime.now()))
    # Create the root directory in the mounted drive
    logger.info("Creating output directory %s.", str(output_path))
    config.create_directory(str(output_path))
    # If the config file is used, copy it into the output folder
    logger.info("Copying config file into %s.", str(output_path))
    if len(config_file) > 0:
        shutil.copy(DOCKER_INPUT_PATH + config_file, str(output_path.joinpath("input_config.yml")))
    output_path = str(output_path)

    # Note: To avoid changing formatting, presently the data will be saved as train/test sets, but then be combined before passed through the model
    # Prepare weather .parquet file if not skipped
    if not skip_weather_generation:
        weather_file = prepare_weather.main(config_file, output_path=output_path)
    # Preprocess the data (generates json with train, test, validate)
    X, X_ids, all_features = preprocessing.main(config_file=config_file, hourly_energy_electric_file=None, building_params_electric_file=building_params_folder,
                                                weather_file=weather_file, val_hourly_energy_file=None, val_building_params_file=None,
                                                hourly_energy_gas_file=None, building_params_gas_file=None, output_path=output_path,
                                                preprocess_only_for_predictions=True, ohe_file=ohe_file)
    logger.info("Updating dataset to only use selected features.")
    # Load the selected_features file
    with open(DOCKER_INPUT_PATH + selected_features_file, 'r', encoding='UTF-8') as feature_selection_file:
        features_json = json.load(feature_selection_file)
    # Load the data into a dataframe, only keeping the required features
    X_df = pd.DataFrame(X, columns=all_features)
    X_df = X_df[features_json["features"]]
    # Load the scalers to be used for scaling the input data and predictions
    scaler_X = joblib.load(DOCKER_INPUT_PATH + scaler_X_file)
    scaler_y = joblib.load(DOCKER_INPUT_PATH + scaler_y_file)
    logger.info("Transforming the input data with provided scaler files.")
    # Scale the input data
    X = scaler_X.transform(X_df)
    logger.info("Loading the specified keras model.")
    # Load the keras model
    model = keras.models.load_model(DOCKER_INPUT_PATH + model_file, compile=False)
    logger.info("Getting the predictions for the input data.")
    # Get the megajoule predictions (or call the predict.evaluate function!)
    X_ids["Predicted Daily Energy Total (Megajoules per square meter)"] = model.predict(X)
    logger.info("Scaling the predictions to their appropriate form.")
    # Transform the outputs back into their expected megajoule form
    X_ids["Predicted Daily Energy Total (Megajoules per square meter)"] = scaler_y.inverse_transform(X_ids["Predicted Daily Energy Total (Megajoules per square meter)"].values.reshape(-1,1))
    logger.info("Preparing output file format for daily Megajoules per square meter.")
    # Convert the int date values into a standard representation, without the year
    X_ids["Date"] = X_ids["date_int"].apply(lambda r: '0' + str(r) if len(str(r)) == 3 else str(r))
    X_ids["Date"] = pd.to_datetime(X_ids["Date"], format="%m%d")
    # Replace the year with a placeholder value to clearly identify that it is unused
    X_ids["Date"] = X_ids["Date"].apply(lambda r: r.strftime('%m/%d') + '/YYYY')
    X_ids = X_ids.drop('date_int', axis=1)
    logger.info("Preparing aggregated output in Gigajoules per square meter over the specified date range.")
    # From the daily total, generate a total for the entire start-end date in gigajoules
    X_aggregated = X_ids.drop("Date", axis=1).groupby(['Prediction Identifier']).sum()
    total_days = len(pd.date_range(TEMP_YEAR + start_date, TEMP_YEAR + end_date))
    #X_aggregated['Predicted Energy Total (Gigajoules per square meter)'] = X_aggregated['Predicted Daily Energy Total (Megajoules per square meter)'].apply(lambda r: float(r / total_days))
    X_aggregated["Predicted Energy Total (Gigajoules per square meter)"] = X_aggregated["Predicted Daily Energy Total (Megajoules per square meter)"].apply(lambda r : float((r*1.0)/1000))
    X_aggregated = X_aggregated.drop("Predicted Daily Energy Total (Megajoules per square meter)", axis=1)
    # Output the predictions alongside any relevant information
    logger.info("Outputting predictions to %s.", str(output_path))
    X_ids.to_csv(output_path + '/' + settings.APP_CONFIG.RUNNING_DAILY_RESULTS_FILENAME)
    X_aggregated.to_csv(output_path + '/' + settings.APP_CONFIG.RUNNING_AGGREGATED_RESULTS_FILENAME)
    return

if __name__ == '__main__':
    # Load settings from the environment
    settings = config.Settings()
    # Run the CLI interface
    typer.run(main)
