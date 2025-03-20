import logging
import os
from pathlib import Path
import pandas as pd
import typer
import yaml
import config
import zipfile
from io import BytesIO
import requests

# Get a log handler
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def get_weather_df_from_zip(epw_filename: str) -> pd.DataFrame:
    """Fetch weather data from a zip file stored at the main storage location.

    Args:
        epw_filename: The filename of the EPW file within the zip archive.

    Returns:
        A dataframe of weather data summarized by day.
    """
    epw_file_store = config.Settings().APP_CONFIG.WEATHER_DATA_STORE

    epw_filename = epw_filename.rstrip('.epw')

    # Number of rows in the file that are just providing metadata
    meta_row_count = 8
    logger.debug('Skipping first %s rows of EPW data', meta_row_count)

    # Columns in the file as defined in Auxiliary Programs documentation
    epw_columns = ['year', 'month', 'day', 'hour', 'minute', 'datasource', 'drybulb', 'dewpoint', 'relhum', 'atmos_pressure',
                   'exthorrad', 'extdirrad', 'horirsky', 'glohorrad', 'dirnorrad', 'difhorrad', 'glohorillum', 'dirnorillum',
                   'difhorillum', 'zenlum', 'winddir', 'windspd', 'totskycvr', 'opaqskycvr', 'visibility', 'ceiling_hgt',
                   'presweathobs', 'presweathcodes', 'precip_wtr', 'aerosol_opt_depth', 'snowdepth', 'days_last_snow',
                   'Albedo', 'liq_precip_depth', 'liq_precip_rate']

    zip_url = f"{epw_file_store}/{epw_filename}.zip"
    logger.info("Downloading zip file from %s", zip_url)
    response = requests.get(zip_url, verify=False)
    response.raise_for_status()

    with zipfile.ZipFile(BytesIO(response.content)) as z:
        expected_filename = epw_filename + ".epw"
        if expected_filename not in z.namelist():
            raise FileNotFoundError(f"{expected_filename} not found in the zip archive.")
        with z.open(expected_filename) as epw_file:
            logger.info("Reading EPW file from zip archive")
            df = pd.read_csv(epw_file, names=epw_columns, skiprows=meta_row_count)

    logger.debug("Data shape after fetch: %s", df.shape)
    return df

def save_epw(df: pd.DataFrame, filename: str, output_path: str) -> None:
    """Save prepared EPW data out to blob storage.

    Args:
        df: Pandas DataFrame to be saved out.
        filename: Filename of the source file used to produce the DataFrame.
    """
    logger.debug("Data shape being saved: %s", df.shape)
    if not filename.endswith(".parquet"):
        filename = filename.replace(".epw", "") + ".parquet"

    weather_bucket_path = output_path.joinpath(config.Settings().APP_CONFIG.WEATHER_BUCKET_NAME)
    logger.info("Weather data will be placed under '%s'", weather_bucket_path)

    file_path = str(weather_bucket_path.joinpath(filename))
    logger.info("Creating directory %s.", str(weather_bucket_path))
    config.create_directory(str(weather_bucket_path))

    df.to_parquet(file_path, index=False)
    logger.info("Weather file saved as %s.", file_path)

def adjust_hour(df: pd.DataFrame, colname: str = 'hour') -> pd.DataFrame:
    """Adjust all hourly values to align with Python representation.

    Args:
        df: Raw weather data, with hours from 1 to 24 hours.
        colname: The name of the column with hour indicators.

    Returns:
        A pd.DataFrame object where the hour column has been adjusted to 0 - 23 hours.
    """
    df = df.copy()
    logger.info("Adjusting %s column by -1", colname)
    df[colname] = df[colname] - 1
    return df

def process_weather_file(epw_filename: str):
    """Process a single weather file and return the dataframe.

    Args:
        epw_filename: The filename of the EPW file within the zip archive.

    Returns:
        A pd.DataFrame object with the ready-to-use weather information.
    """
    logger.info("Processing weather file %s", epw_filename)

    weather_drop_list = ['minute', 'datasource', 'exthorrad', 'extdirrad', 'glohorrad', 'glohorillum', 'dirnorrad',
                         'difhorrad', 'zenlum', 'totskycvr', 'opaqskycvr', 'visibility', 'ceiling_hgt', 'precip_wtr',
                         'aerosol_opt_depth', 'days_last_snow', 'Albedo', 'liq_precip_rate', 'presweathcodes']
    logger.debug("Dropping %s unused columns from weather data", len(weather_drop_list))

    df = (get_weather_df_from_zip(epw_filename)
          .drop(weather_drop_list, axis=1)
          .pipe(adjust_hour)
          .assign(rep_time=lambda x: pd.to_datetime(x[['year', 'month', 'day', 'hour']]))
          )

    logger.debug("Data shape in processing: %s", df.shape)
    if df['hour'].loc[df['hour'] > 23].any():
        logger.warning("Hour values greater than 23 found. Date parsing may result in shifted days.")
    return df

def process_weather_files(filenames: list):
    """
    Process a batch of weather files and return the combined dataframe.

    Args:
        filenames: List of EPW filenames within their respective zip archives.

    Returns:
        A pd.DataFrame object with the combined weather information.
    """
    combined_weather_df = None
    for filename in filenames:
        weather_df = process_weather_file(filename)
        weather_df[':epw_file'] = filename
        if combined_weather_df is not None:
            combined_weather_df = pd.concat([combined_weather_df, weather_df], ignore_index=True)
        else:
            combined_weather_df = weather_df
    return combined_weather_df

def main(config_file: str = typer.Argument(..., help="Path to configuration YAML file."),
         epw_file: str = typer.Option("", help="The EPW file name to be used."),
         output_path: str = typer.Option("", help="The output path to be used. Defaults to pipeline output path.")) -> str:
    """Take raw EPW files from a zip archive, process them, and save them as parquet files.

    Args:
        config_file: Path to the configuration file used for EnergyPlus.
        epw_file: The EPW file name to be used.
        output_path: Where output data should be placed.
    """
    logger.info("Starting weather file processing.")
    DOCKER_INPUT_PATH = config.Settings().APP_CONFIG.DOCKER_INPUT_PATH
    if len(config_file) > 0:
        cfg = config.get_config(DOCKER_INPUT_PATH + config_file)
        if not epw_file:
            epw_file = cfg.get(config.Settings().APP_CONFIG.WEATHER_KEY)

    if len(output_path) < 1:
        output_path = config.Settings().APP_CONFIG.DOCKER_OUTPUT_PATH

    if isinstance(epw_file, str):
        data = process_weather_file(epw_file)
    else:
        data = process_weather_files(epw_file)

    output_filepath = save_epw(data, epw_file if isinstance(epw_file, str) else "combined", Path(output_path))
    logger.info("Weather file processing complete. File saved at %s", output_filepath)
    return output_filepath

if __name__ == '__main__':
    settings = config.Settings()
    typer.run(main)
