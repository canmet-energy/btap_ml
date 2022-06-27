#!/usr/bin/env python3
"""Prepare the weather file(s) specified in the YAML configuration to be used by the pipeline.

CLI arguments match those defined by ``main()``.
"""
import logging
import os
from pathlib import Path

import pandas as pd
import typer
import yaml

import config

# Get a log handler
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def get_weather_df(filename: str) -> pd.DataFrame:
    """Fetch weather data from the main storage location.
    Loads weather files from the NREL/openstudio-standards GitHub repository where they are managed.

    Args:
        filename: The filename of a weather file (.epw)

    Returns:
        A dataframe of weather data summarized by day.
    """
    epw_file_store = config.Settings().APP_CONFIG.WEATHER_DATA_STORE

    # Number of rows in the file that are just providing metadata
    meta_row_count = 8
    logger.debug('Skipping first %s rows of EPW data', meta_row_count)

    # Columns in the file as defined in Auxiliary Programs documentation
    epw_columns = ['year', 'month', 'day', 'hour', 'minute', 'datasource', 'drybulb', 'dewpoint', 'relhum', 'atmos_pressure',
                   'exthorrad', 'extdirrad', 'horirsky', 'glohorrad', 'dirnorrad', 'difhorrad', 'glohorillum', 'dirnorillum',
                   'difhorillum', 'zenlum', 'winddir', 'windspd', 'totskycvr', 'opaqskycvr', 'visibility', 'ceiling_hgt',
                   'presweathobs', 'presweathcodes', 'precip_wtr', 'aerosol_opt_depth', 'snowdepth', 'days_last_snow',
                   'Albedo', 'liq_precip_depth', 'liq_precip_rate']

    file_url = f"{epw_file_store}/{filename}"
    logger.info("Reading EPW file from %s", file_url)
    df = pd.read_csv(file_url, names=epw_columns, skiprows=meta_row_count)
    logger.debug("Data shape after fetch: %s", df.shape)

    return df


def save_epw(df: pd.DataFrame, filename: str, output_path: str) -> None:
    """Save preprared EPW data out to blob storage.
    The filename of the original file is used, with the extension replaced with .parquet.

    Args:
        df: Pandas DataFrame to be saved out.
        filename: Filename of the source file used to produce the DataFrame.
    """
    logger.debug("Data shape being saved: %s", df.shape)
    # Add a parquet extension to the file name
    if not filename.endswith(".parquet"):
        filename = filename.replace(".epw", "")
        filename = f"{filename}.parquet"

    # Bucket used to store weather data.
    #weather_bucket_path = config.Settings().NAMESPACE.joinpath(config.Settings().APP_CONFIG.WEATHER_BUCKET_NAME)
    weather_bucket_path = output_path.joinpath(config.Settings().APP_CONFIG.WEATHER_BUCKET_NAME)
    logger.info("Weather data will be placed under '%s'", weather_bucket_path)

    file_path = str(weather_bucket_path.joinpath(filename))
    # Create the weather directory in the mounted drive
    logger.info("Creating directory %s.", str(weather_bucket_path))
    config.create_directory(str(weather_bucket_path))
    # Write the weather file as a parquet file
    with open(file_path, 'wb') as outfile:
        df.to_parquet(outfile)
    return file_path


def adjust_hour(df: pd.DataFrame, colname: str = 'hour'):
    """Adjust all hourly values to align with Python representation.

    Args:
        df: Raw weather data, with hours from 1 to 24 hours.
        colname: The name of the column with hour indicators.

    Returns:
        df: A pd.DataFrame object where the hour column has been adjusted to 0 - 23 hours.

    """
    df = df.copy()
    logger.info("Adjusting %s column by -1", colname)
    df[colname] = df[colname] - 1
    return df


def process_weather_file(filename: str):
    """Process a weather file and return the dataframe.

    Args:
        filename: The name of the weather file to load, as defined in the building config file.

    Returns:
        A pd.DataFrame object with the ready to use weather information.
    """
    logger.info("Processing weather file %s", filename)
    # Columns not used by EnergyPlus
    weather_drop_list = ['minute', 'datasource', 'exthorrad', 'extdirrad', 'glohorrad', 'glohorillum', 'dirnorrad',
                         'difhorrad', 'zenlum', 'totskycvr', 'opaqskycvr', 'visibility', 'ceiling_hgt', 'precip_wtr',
                         'aerosol_opt_depth', 'days_last_snow', 'Albedo', 'liq_precip_rate', 'presweathcodes']
    logger.debug("Dropping %s unused columns from weather data", len(weather_drop_list))

    df = (get_weather_df(filename)
          .drop(weather_drop_list, axis=1)
          .pipe(adjust_hour)
          .assign(rep_time=lambda x: pd.to_datetime(x[['year', 'month', 'day', 'hour']])))
    logger.debug("Data shape in processing: %s", df.shape)
    if df['hour'].loc[df['hour'] > 23].any():
        logger.warn("Hour values greater than 23 found. Date parsing will likely return values coded to the following days.")
    return df


def main(config_file: str = typer.Argument(..., help="Path to configuration YAML file."),
         epw_file: str = typer.Option("", help="The epw key to be used if the config file is not used."),
         output_path: str = typer.Option("", help="The output path to be used. Note that this value should be empty unless this file is called from a pipeline."),
         ) -> str:
    """Take raw EPW files as defined in BTAP YAML configuration and prepare it for use by the model.
    Uses the EnergyPlus configuration to identify and process weather data in EPW format. The weather data is then
    saved to blob storage for use in later processing stages.

    Args:
        config_file: Path to the configuration file used for EnergyPlus.
        epw_file: The epw key to be used if the config file is not use.
        output_path: Where output data should be placed. Note that this value should be empty unless this file is called from a pipeline.
    """
    logger.info("Beginning the weather processing step.")
    DOCKER_INPUT_PATH = config.Settings().APP_CONFIG.DOCKER_INPUT_PATH
    # Load the information from the config file, if provided
    if len(config_file) > 0:
        # Load the specified config file
        cfg = config.get_config(DOCKER_INPUT_PATH + config_file)
        # No validation is needed beyond loading the specified epw key
        if isinstance(epw_file, str) and epw_file == "": epw_file = cfg.get(config.Settings().APP_CONFIG.WEATHER_KEY)

    # If the output path is blank, map to the docker output path
    if len(output_path) < 1:
        output_path = config.Settings().APP_CONFIG.DOCKER_OUTPUT_PATH

    # Data could be a single file or a list of files
    # The application assumes that a list will have a maximum size of one
    if not isinstance(epw_file, str):
        epw_file = epw_file[0]

    logger.info("Beginning the loading process for the weather file with the key %s.", epw_file)
    data = process_weather_file(epw_file)
    logger.debug("Data shape after processing: %s", data.shape)
    output_filepath = save_epw(data, epw_file, Path(output_path))
    logger.info("Weather file has been saved as %s.", output_filepath)
    return output_filepath


if __name__ == '__main__':
    # Load settings from the environment.
    settings = config.Settings()
    # Run the CLI interface.
    typer.run(main)
