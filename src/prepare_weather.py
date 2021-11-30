"""Prepare the weather file(s) specified in the YAML configuration to be used by the pipeline."""
import logging

import pandas as pd
import typer
import yaml

import config

# Get a log handler
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def get_config(config_file: str):
    """Load the specified configuration file from blob storage.

    Args:
        config_file: Path to the config file relative to the default bucket.

    Returns:
        Dictionary of configuration information.
    """
    logger.info("Establishing connection to S3 store at %s", settings.MINIO_URL)
    s3 = config.establish_s3_connection(settings.MINIO_URL,
                                        settings.MINIO_ACCESS_KEY,
                                        settings.MINIO_SECRET_KEY)

    # Create a path to the config from the namespace
    config_file_path = settings.NAMESPACE.joinpath(config_file).as_posix()
    logger.info("Reading config from file %s", config_file_path)
    contents = yaml.safe_load(s3.open(config_file_path, mode='rb'))
    return contents


def get_weather_df(filename: str) -> pd.DataFrame:
    """Fetch weather data from the main storage location.
    Loads weather files from the NREL/openstudio-standards GitHub repository where they are managed.

    Args:
        filename: The filename of a weather file (.epw)

    Returns:
        A dataframe of weather data summarized by day.
    """
    epw_file_store = settings.APP_CONFIG.WEATHER_DATA_STORE

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


def save_epw(df: pd.DataFrame, filename: str) -> None:
    """Save preprared EPW data out to blob storage.
    The filename of the original file is used, with the extension replaced with .parquet.

    Args:
        df: Pandas DataFrame to be saved out.
        filename: Filename of the source file used to produce the DataFrame.
    """
    logger.debug("Data shape being saved: %s", df.shape)
    # Add a parquet extension to the file name
    if not filename.endswith(".parquet"):
        filename = f"{filename}.parquet"

    # Bucket used to store weather data.
    weather_bucket_name = settings.APP_CONFIG.WEATHER_BUCKET_NAME
    logger.info("Weather data will be placed under '%s'", weather_bucket_name)

    # Establish a connection to the blob store.
    s3 = config.establish_s3_connection(settings.MINIO_URL,
                                        settings.MINIO_ACCESS_KEY,
                                        settings.MINIO_SECRET_KEY)

    # Make sure the bucket for weather data exists to avoid write errors
    existing_items = s3.ls(settings.NAMESPACE.as_posix())
    if weather_bucket_name not in existing_items:
        bucket_path = settings.NAMESPACE.joinpath(weather_bucket_name).as_posix()
        logger.info("Weather bucket not found, creating %s", bucket_path)
        s3.mkdir(bucket_path)

    # Write the data to s3
    file_path = settings.NAMESPACE.joinpath(weather_bucket_name, filename).as_posix()
    logger.info("Saving weather data to %s", file_path)
    with s3.open(file_path, 'wb') as outfile:
        df.to_parquet(outfile)


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
          .assign(rep_time=lambda x: pd.to_datetime(x[['year', 'month', 'day', 'hour']])))
    logger.debug("Data shape in processing: %s", df.shape)
    return df


def main(config_file: str = typer.Argument(..., help="Path to configuration YAML file for BTAP CLI."),
         epw_file_key: str = typer.Option(':epw_file', help="Key used to store EPW file names.")) -> None:
    """Take raw EPW files as defined in BTAP YAML configuration and prepare it for use by the model.
    Uses the EnergyPlus configuration to identify and process weather data in EPW format. The weather data is then
    saved to blob storage for use in later processing stages.

    Args:
        config_file: Path to the configuration file used for EnergyPlus.
        epw_file_key: The key in the configuration file that has the weather file name(s). Default is ``:epw_file``.
    """
    cfg = get_config(config_file)
    building_opts_key = settings.APP_CONFIG.BUILDING_OPTS_KEY

    # Guard against incorrect or undefined file key.
    # EPW information should be under the building options.
    logger.debug("Looking for %s in %s", epw_file_key, building_opts_key)
    if epw_file_key not in cfg.get(building_opts_key):
        raise AttributeError(f"EPW file specification not found under {building_opts_key}.")

    epw_files = cfg.get(building_opts_key).get(epw_file_key)
    logger.info("Found %s weather files to process", len(epw_files))

    # Data could be a single file or a list of files
    if isinstance(epw_files, str):
        data = process_weather_file(epw_files)
        logger.debug("Data shape after processin: %s", data.shape)
        save_epw(data, epw_files)
    else:
        for name in epw_files:
            data = process_weather_file(name)
            logger.debug("Data shape after processing: %s", data.shape)
            save_epw(data, name)


if __name__ == '__main__':
    # Load settings from the environment.
    settings = config.Settings()

    # Run the CLI interface.
    typer.run(main)
