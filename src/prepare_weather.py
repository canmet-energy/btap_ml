import typer
import config
import yaml
import pandas as pd


# Weather file master repo
_epw_file_store = 'https://raw.githubusercontent.com/NREL/openstudio-standards/nrcan/data/weather/'

def get_config(config_file: str):
    """Load the specified configuration file from blob storage."""
    # TODO: implement loading config from blob storage
    contents = yaml.safe_load(config_file)
    return contents


def get_weather_df(filename: str):
    """Fetch weather data from the main storage location."""
    global _epw_file_store

    # Number of rows in the file that are just providing metadata
    meta_row_count = 8

    # Columns in the file as defined in Auxiliary Programs documentation
    epw_columns = ['year','month','day','hour','minute','datasource','drybulb','dewpoint','relhum','atmos_pressure',
                   'exthorrad','extdirrad','horirsky','glohorrad','dirnorrad','difhorrad','glohorillum','dirnorillum',
                   'difhorillum','zenlum','winddir','windspd','totskycvr','opaqskycvr','visibility','ceiling_hgt',
                   'presweathobs','presweathcodes','precip_wtr','aerosol_opt_depth','snowdepth','days_last_snow',
                   'Albedo','liq_precip_depth','liq_precip_rate']
    
    file_url = _epw_file_store + filename
    df = pd.read_csv(file_url, names=epw_columns, skiprows=meta_row_count)

    return df


def save_epw(df: pd.DataFrame):
    """Save preprared EPW data out to blob storage."""
    # TODO: Implement saving data back to storage
    pass


def main(config_file: str, epw_file_key: str=':epw_file'):
    """Take raw EPW files as defined in BTAP YAML configuration and prepare it for use by the model."""
    cfg = get_config(config_file)

    # Guard against incorrect or undefined file key
    if epw_file_key not in cfg:
        raise AttributeError("EPW file specification missing.")
    
    epw_files = cfg[epw_file_key]

    # Columns not used by EnergyPlus
    weather_drop_list = ['minute', 'datasource', 'exthorrad', 'extdirrad', 'glohorrad', 'glohorillum', 'dirnorrad', 
                         'difhorrad', 'zenlum', 'totskycvr', 'opaqskycvr', 'visibility', 'ceiling_hgt', 'precip_wtr', 
                         'aerosol_opt_depth', 'days_last_snow', 'Albedo', 'liq_precip_rate', 'presweathcodes']
    
    # Data could be a single file or a list of files
    if isinstance(epw_files, list):
        for name in epw_files:
            data = (get_weather_df(name)
                    .drop(weather_drop_list, axis=1)
                    .assign(rep_time=lambda x: pd.to_datetime(x['year','month','day','hour'])))
            save_epw(data)
    else:
        data = (get_weather_df(epw_files)
                .drop(weather_drop_list, axis=1)
                .assign(rep_time=lambda x: pd.to_datetime(x['year','month','day','hour'])))
        save_epw(data)


if __name__ == '__main__':
    typer.run(main)
