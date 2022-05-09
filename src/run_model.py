"""
Given a specific model and files, output the model's predictions for the files.

CLI arguments match those defined by ``main()``.
"""
import os
from datetime import datetime
from pathlib import Path

import typer
import yaml
from tensorflow import keras

import config
import prepare_weather
import preprocessing


def main(config_file: str = typer.Argument(..., help="Path to the configuration YAML file for BTAP CLI."),
         model_file: str = typer.Option("", help="The location of the trained model to be used."),
         weather_file: str = typer.Option("", help="The weather file which will be processed."),
         building_params_file: str = typer.Option("", help="The building parameters file which will have predictions made on by the provided model."),
         hourly_energy_params_file: str = typer.Option("", help="The hourly energy file which will have predictions made on by the provided model."),
         features_file: str = typer.Option("", help="The selected features file which will have predictions made on by the provided model."),
         output_path: str = typer.Option("", help="The minio location and filename where the output file should be written."),
         skip_weather_generation: bool = typer.Option(False, help="True to skip the weather processing (i.e. use the specified .parquet file from the weather_file option), False to perform the weather processing."),
         skip_file_preprocessing: bool = typer.Option(False, help="True to skip the preprocessing (i.e. use the specified preprocessing output .json, under the preprocessed_data_file option, file from a previous run), False to perform the preprocessing."),
         ) -> None:
    """
    TODO: Complete description

    Args:
        ...
    """
    settings = config.Settings()
    if len(config_file) > 0:
        #load_and_validate_config(config_file)
        cfg = config.get_config(config_file)
        features_file = cfg.get(settings.APP_CONFIG.FEATURES_FILE)
        model_file = cfg.get(settings.APP_CONFIG.TRAINED_MODEL_FILE)

    # Create directory to hold all data for the run (datetime/...)
    # Create the root directory in the mounted drive
    if not os.path.isdir(str(output_path)):
        os.mkdir(output_path)

    # Need preprocessing -> predict
    # Either output as json or call directly
    # Will need weather file if called
    # Weather file -> parquet
    # Building and energy (gas and electric) -> json output\
    # Note: To avoid changing formatting, presently the data will be saved as train/test sets, but then be combined before passed through the model
    # Prepare weather (perhaps can allow a .csv to require processing while .parquet skips processing)
    # TODO: Verify that y is not needed for these files within preprocessing.py (since there would just be the X values)
    if not skip_weather_generation:
        weather_file = prepare_weather.main(config_file)
    # Preprocess the data (generates json with train, test, validate)
    # TODO: Make a function in the file for ONLY generating a complete test set (use config values)
    if not skip_file_preprocessing:
        preprocessed_data_file = preprocessing.main(config_file=config_file, hourly_energy_electric_file=hourly_energy_params_file, building_params_electric_file=building_params_file,
                                                    weather_file=weather_file, val_hourly_energy_file=None, val_building_params_file=None,
                                                    hourly_energy_gas_file=None, building_params_gas_file=None, output_path=output_path, preprocess_only_samples=True)
    # Load the preprocessed data and the mode
    preprocessed_data = ...
    model = keras.models.load_model(model_file)
    # Get the predictions (or call the predict.evaluate function!)
    predictions = model.predict(preprocessed_data)
    # Output the predictions alongside any relevant information
    ...
    return

if __name__ == '__main__':
    # Load settings from the environment
    settings = config.Settings()
    # Run the CLI interface
    typer.run(main)
