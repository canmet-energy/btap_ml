"""
For both the energy and costing training outputs, use the specified models and files to output the model predictions
for the specified batch of building files.
"""
import json
import logging
import os
import shutil
import time
from datetime import datetime
from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import shap
import typer
from lightgbm import LGBMRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.neural_network import MLPRegressor
from tensorflow import keras
from xgboost import XGBRegressor

import config
import preprocessing
from models.running_model import RunningModel

# Get a log handler
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)



def initialize_run_process(docker_input_path, config_file, model_file, ohe_file, cleaned_columns_file,
                           scaler_X_file, scaler_y_file, building_params_folder, start_date, end_date,
                           selected_features_file):
    """
    Initializes the running model for either costing or energy runs.

    Args:
        docker_input_path: The input prefix to be used for all files provided.
        config_file: Location of the .yml config file (default name is input_config.yml).
        model_file: Location and name of a .h5 trained keras model to be used for training.
        ohe_file: Location and name of a ohe.pkl OneHotEncoder file which was generated in the root of a training output folder.
        cleaned_columns_file: Location and name of a cleaned_columns.json file which was generated in the root of a training output folder.
        scaler_X_file: Location and name of a scaler_X.pkl fit scaler file which was generated with the trained model_file.
        scaler_y_file: Location and name of a scaler_y.pkl fit scaler file which was generated with the trained model_file.
        building_params_folder: The folder location containing all building parameter files which will have predictions made on by the provided model.
        start_date: The start date to specify the start of which weather data is attached to the building data. Expects the input to be in the form Month_number-Day_number.
        end_date: The end date to specify the end of which weather data is attached to the building data. Expects the input to be in the form Month_number-Day_number.
        selected_features_file: Location and name of a .json feature selection file to be used if the feature selection is skipped.
    """
    return RunningModel(input_prefix=docker_input_path,
                        config_file=config_file,
                        model_file=model_file,
                        ohe_file=ohe_file,
                        cleaned_columns_file=cleaned_columns_file,
                        scaler_X_file=scaler_X_file,
                        scaler_y_file=scaler_y_file,
                        building_params_folder=building_params_folder,
                        start_date=start_date,
                        end_date=end_date,
                        selected_features_file=selected_features_file)

def main(config_file: str = typer.Argument(..., help="Location of the .yml config file (default name is input_config.yml)."),
         energy_model_file: str = typer.Option("", help="Location and name of a .h5 trained keras model to be used for training. From the energy training."),
         energy_ohe_file: str = typer.Option("", help="Location and name of a ohe.pkl OneHotEncoder file which was generated in the root of a training output folder. From the energy training."),
         energy_cleaned_columns_file: str = typer.Option("", help="Location and name of a cleaned_columns.json file which was generated in the root of a training output folder. From the energy training."),
         energy_scaler_X_file: str = typer.Option("", help="Location and name of a scaler_X.pkl fit scaler file which was generated with the trained model_file. From the energy training."),
         energy_scaler_y_file: str = typer.Option("", help="Location and name of a scaler_y.pkl fit scaler file which was generated with the trained model_file. From the energy training."),
         energy_selected_features_file: str = typer.Option("", help="Location and name of a .json feature selection file to be used if the feature selection is skipped. From the energy training."),

         costing_model_file: str = typer.Option("", help="Location and name of a .h5 trained keras model to be used for training. From the costing training."),
         costing_ohe_file: str = typer.Option("", help="Location and name of a ohe.pkl OneHotEncoder file which was generated in the root of a training output folder. From the costing training."),
         costing_cleaned_columns_file: str = typer.Option("", help="Location and name of a cleaned_columns.json file which was generated in the root of a training output folder. From the costing training."),
         costing_scaler_X_file: str = typer.Option("", help="Location and name of a scaler_X.pkl fit scaler file which was generated with the trained model_file. From the costing training."),
         costing_scaler_y_file: str = typer.Option("", help="Location and name of a scaler_y.pkl fit scaler file which was generated with the trained model_file. From the costing training."),
         costing_selected_features_file: str = typer.Option("", help="Location and name of a .json feature selection file to be used if the feature selection is skipped. From the costing training."),

         building_params_folder: str = typer.Option("", help="The folder location containing all building parameter files which will have predictions made on by the provided model."),
         start_date: str = typer.Option("", help="The start date to specify the start of which weather data is attached to the building data. Expects the input to be in the form Month_number-Day_number (ex: 1-1)."),
         end_date: str = typer.Option("", help="The end date to specify the end of which weather data is attached to the building data. Expects the input to be in the form Month_number-Day_number (ex: 12-31)."),
         selected_model_type: str = typer.Option("", help="Type of model selected. can either be 'mlp' or 'rf'")
         ) -> None:
    """
    Preprocess a set of input building files to obtain a dataset to obtain daily energy and total costing predictions for.
    The feature selection file that has been used with the trained model must be included to appropriately preprocess the data.
    The start/end dates to be spanned are specified within the provided config_file or through the CLI, but it is assumed that
    each day within an arbitrary year will receive predictions.
    A trained Keras model must be provided as input to perform the predictions on the data. These predictions will be output into a
    .csv file which follows the format of the input files which are used to train the models. The energy outputs will be for daily energy
    values rather than hourly energy values, where outputs represent the total energy output observed from generated energy files
    from rows without the Electricity:Facility Name.

    Args:
        config_file: Location of the .yml config file (default name is input_config.yml).
        energy_model_file: Location and name of a .h5 trained keras model to be used for training. From the energy training.
        energy_ohe_file: Location and name of a ohe.pkl OneHotEncoder file which was generated in the root of a training output folder. From the energy training.
        energy_cleaned_columns_file: Location and name of a cleaned_columns.json file which was generated in the root of a training output folder. From the energy training.
        energy_scaler_X_file: Location and name of a scaler_X.pkl fit scaler file which was generated with the trained model_file. From the energy training.
        energy_scaler_y_file: Location and name of a scaler_y.pkl fit scaler file which was generated with the trained model_file. From the energy training.
        energy_selected_features_file: Location and name of a .json feature selection file to be used if the feature selection is skipped. From the energy training.
        costing_model_file: Location and name of a .h5 trained keras model to be used for training. From the costing training.
        costing_ohe_file: Location and name of a ohe.pkl OneHotEncoder file which was generated in the root of a training output folder. From the costing training.
        costing_cleaned_columns_file: Location and name of a cleaned_columns.json file which was generated in the root of a training output folder. From the costing training.
        costing_scaler_X_file: Location and name of a scaler_X.pkl fit scaler file which was generated with the trained model_file. From the costing training.
        costing_scaler_y_file: Location and name of a scaler_y.pkl fit scaler file which was generated with the trained model_file. From the costing training.
        costing_selected_features_file: Location and name of a .json feature selection file to be used if the feature selection is skipped. From the costing training.
        building_params_folder: The folder location containing all building parameter files which will have predictions made on by the provided model.
        start_date: The start date to specify the start of which weather data is attached to the building data. Expects the input to be in the form Month_number-Day_number.
        end_date: The end date to specify the end of which weather data is attached to the building data. Expects the input to be in the form Month_number-Day_number.
        selected_model_type: Type of model selected. can either be 'mlp' for Multilayer Perceptron or 'rf' for Random Forest
    """
    start_time = time.time()

    settings = config.Settings()
    DOCKER_INPUT_PATH = config.Settings().APP_CONFIG.DOCKER_INPUT_PATH
    # Define the year to be used for any date operations (a non leap year)
    # This value does not need to be adjusted for the current year
    TEMP_YEAR = "2022-"
    # Define the output column names to be used
    COL_NAME_DAILY_MEGAJOULES_ELEC = "Predicted Daily Electricity Energy Total (Megajoules per square meter)"
    COL_NAME_DAILY_MEGAJOULES_GAS = "Predicted Daily Gas Energy Total (Megajoules per square meter)"
    COL_NAME_AGGREGATED_GIGAJOULES_ELEC = "Predicted Electricity Energy Total (Gigajoules per square meter)"
    COL_NAME_AGGREGATED_GIGAJOULES_GAS = "Predicted Gas Energy Total (Gigajoules per square meter)"
    COL_NAME_TOTAL_COSTING = "Predicted Total Costing (per square meter)"

    if len(config_file) > 0:
        #load_and_validate_config(config_file)
        cfg = config.get_config(DOCKER_INPUT_PATH + config_file)
        # Load the energy training files
        if energy_model_file == "": energy_model_file = cfg.get(settings.APP_CONFIG.ENERGY_PREFIX + settings.APP_CONFIG.TRAINED_MODEL_FILE)
        if energy_ohe_file == "": energy_ohe_file = cfg.get(settings.APP_CONFIG.ENERGY_PREFIX + settings.APP_CONFIG.OHE_FILE)
        if energy_cleaned_columns_file == "": energy_cleaned_columns_file = cfg.get(settings.APP_CONFIG.ENERGY_PREFIX + settings.APP_CONFIG.CLEANED_COLUMNS_FILE)
        if energy_scaler_X_file == "": energy_scaler_X_file = cfg.get(settings.APP_CONFIG.ENERGY_PREFIX + settings.APP_CONFIG.SCALER_X_FILE)
        if energy_scaler_y_file == "": energy_scaler_y_file = cfg.get(settings.APP_CONFIG.ENERGY_PREFIX + settings.APP_CONFIG.SCALER_Y_FILE)
        if energy_selected_features_file == "": energy_selected_features_file = cfg.get(settings.APP_CONFIG.ENERGY_PREFIX + settings.APP_CONFIG.FEATURES_FILE)
        # Load the costing training files
        if costing_model_file == "": costing_model_file = cfg.get(settings.APP_CONFIG.COSTING_PREFIX + settings.APP_CONFIG.TRAINED_MODEL_FILE)
        if costing_ohe_file == "": costing_ohe_file = cfg.get(settings.APP_CONFIG.COSTING_PREFIX + settings.APP_CONFIG.OHE_FILE)
        if costing_cleaned_columns_file == "": costing_cleaned_columns_file = cfg.get(settings.APP_CONFIG.COSTING_PREFIX + settings.APP_CONFIG.CLEANED_COLUMNS_FILE)
        if costing_scaler_X_file == "": costing_scaler_X_file = cfg.get(settings.APP_CONFIG.COSTING_PREFIX + settings.APP_CONFIG.SCALER_X_FILE)
        if costing_scaler_y_file == "": costing_scaler_y_file = cfg.get(settings.APP_CONFIG.COSTING_PREFIX + settings.APP_CONFIG.SCALER_Y_FILE)
        if costing_selected_features_file == "": costing_selected_features_file = cfg.get(settings.APP_CONFIG.COSTING_PREFIX + settings.APP_CONFIG.FEATURES_FILE)
        # Load the remaining run information
        if building_params_folder == "": building_params_folder = cfg.get(config.Settings().APP_CONFIG.BUILDING_BATCH_PATH)
        if start_date == "": start_date = cfg.get(config.Settings().APP_CONFIG.SIMULATION_START_DATE)
        if end_date == "": end_date = cfg.get(config.Settings().APP_CONFIG.SIMULATION_END_DATE)
        if selected_model_type == "": selected_model_type = cfg.get(config.Settings().APP_CONFIG.SELECTED_MODEL_TYPE)

    # Identify the training processes to be taken (energy and/or costing)
    RUNNING_PROCESSES = [config.Settings().APP_CONFIG.ENERGY,
                         config.Settings().APP_CONFIG.COSTING]

    # Create directory to hold all data for the run (datetime/...)
    # If used, copy the config file within the directory to log the input values
    output_path = config.Settings().APP_CONFIG.DOCKER_OUTPUT_PATH
    output_path = Path(output_path).joinpath(settings.APP_CONFIG.RUN_BUCKET_NAME + str(datetime.now()).replace(":", "-")).joinpath(cfg.get(config.Settings().APP_CONFIG.SELECTED_MODEL_TYPE))
    # Create the root directory in the mounted drive
    logger.info("Creating output directory %s.", str(output_path))
    config.create_directory(str(output_path))

    # If the config file is used, copy it into the output folder
    logger.info("Copying config file into %s.", str(output_path))
    if len(config_file) > 0:
        shutil.copy(DOCKER_INPUT_PATH + config_file, str(output_path.joinpath("input_config.yml")))

    # Store the building data outside the loop such that it can be reused
    buildings_df = None
    output_path = str(output_path)
    # Perform all specified training processes
    for running_process in RUNNING_PROCESSES:
        # Validate all input arguments before continuing
        # Load the energy model if selected
        if running_process.lower() == config.Settings().APP_CONFIG.ENERGY:
            input_model = initialize_run_process(DOCKER_INPUT_PATH, config_file, energy_model_file, energy_ohe_file,
                                                 energy_cleaned_columns_file, energy_scaler_X_file, energy_scaler_y_file,
                                                 building_params_folder, start_date, end_date, energy_selected_features_file)
        # Load the costing model if selected
        elif running_process.lower() == config.Settings().APP_CONFIG.COSTING:
            input_model = initialize_run_process(DOCKER_INPUT_PATH, config_file, costing_model_file, costing_ohe_file,
                                                 costing_cleaned_columns_file, costing_scaler_X_file, costing_scaler_y_file,
                                                 building_params_folder, start_date, end_date, costing_selected_features_file)
        # If neither are selected, terminat ethe program
        else:
            return "The selected running type", str(running_process), "is invalid. Program is terminating."

        # Note: To avoid changing formatting, presently the data will be saved as train/test sets, but then be combined before passed through the model
        # Preprocess the data (generates json with train, test, validate)
        X, X_ids, all_features = preprocessing.main(config_file=input_model.config_file,
                                                    process_type=running_process,
                                                    hourly_energy_electric_file=None,
                                                    building_params_electric_file=None,
                                                    val_hourly_energy_file=None,
                                                    val_building_params_file=None,
                                                    hourly_energy_gas_file=None,
                                                    building_params_gas_file=None,
                                                    output_path=output_path,
                                                    preprocess_only_for_predictions=True,
                                                    building_params_folder=input_model.building_params_folder,
                                                    random_seed=-1,
                                                    start_date=input_model.start_date,
                                                    end_date=input_model.end_date,
                                                    ohe_file=input_model.ohe_file,
                                                    cleaned_columns_file=input_model.cleaned_columns_file)
        logger.info("Updating dataset to only use selected features.")
        # Load the selected_features file
        with open(input_model.selected_features_file, 'r', encoding='UTF-8') as feature_selection_file:
            features_json = json.load(feature_selection_file)
        # Load the data into a dataframe, only keeping the required features
        X_df = pd.DataFrame(X, columns=all_features)
        # X_shap = X_df
        X_df = X_df[features_json["features"]]
        X_shap = X_df.copy()
        # Load the scalers to be used for scaling the input data and predictions
        scaler_X = joblib.load(input_model.scaler_X_file)
        scaler_y = joblib.load(input_model.scaler_y_file)
        logger.info("Transforming the input data with provided scaler files.")
        # Scale the input data
        X = scaler_X.transform(X_df)
        logger.info("Loading the specified keras model.")
        # Load the keras model
        if selected_model_type.lower() == config.Settings().APP_CONFIG.MULTILAYER_PERCEPTRON:
            model = keras.models.load_model(input_model.model_file, compile=False)
        else:
            model = joblib.load(input_model.model_file)

        logger.info("Getting the predictions for the input data.")

        predictions = scaler_y.inverse_transform(model.predict(X))

        """
        SHAP Analysis - Start
        """

        def get_explainer(model, background, model_type=None):
            if model_type is None:
                model_type = auto_detect_model_type(model)

            if model_type == "tree":
                return shap.Explainer(model, background)
            elif model_type == "mlp":
                return shap.KernelExplainer(lambda x: model.predict(x), background.to_numpy())
            else:
                return shap.KernelExplainer(lambda x: model.predict(x), background.to_numpy())

        def auto_detect_model_type(model):
            if hasattr(model, "predict_proba"):
                return "classifier"
            elif isinstance(model, (RandomForestRegressor, XGBRegressor, LGBMRegressor)):
                return "tree"
            elif isinstance(model, MLPRegressor):
                return "mlp"
            else:
                return "kernel"

        def load_feature_rename_dict():
            return {
            ':airloop_economizer_type_DifferentialDryBulb': 'Economizer: Differential Dry Bulb',
            ':airloop_economizer_type_DifferentialEnthalpy': 'Economizer: Differential Enthalpy',
            ':airloop_economizer_type_NECB_Default': 'Economizer: NECB Default',

            ':boiler_eff_NECB 88%% Efficient Condensing Boiler': 'Boiler: 88%% Efficient Condensing',
            ':boiler_eff_NECB_Default': 'Boiler: NECB Default',
            ':boiler_eff_Viessmann Vitocrossal 300 CT3-17 96.2%% Efficient Condensing Gas Boiler': 'Boiler: Viessmann 96.2%% Efficient',

            ':dcv_type_CO2_based_DCV': 'DCV: CO₂-Based',
            ':dcv_type_No_DCV': 'DCV: None',
            ':dcv_type_Occupancy_based_DCV': 'DCV: Occupancy-Based',

            ':ecm_system_name_HS08_CCASHP_VRF': 'System: CCASHP + VRF',
            ':ecm_system_name_HS09_CCASHP_Baseboard': 'System: CCASHP + Baseboard',
            ':ecm_system_name_HS11_ASHP_PTHP': 'System: ASHP + PTHP',
            ':ecm_system_name_HS12_ASHP_Baseboard': 'System: ASHP + Baseboard',
            ':ecm_system_name_HS13_ASHP_VRF': 'System: ASHP + VRF',
            ':ecm_system_name_NECB_Default': 'System: NECB Default',

            ':erv_package_NECB_Default_All': 'ERV: NECB Default All',
            ':erv_package_Plate-All': 'ERV: Plate All',
            ':erv_package_Plate-Existing': 'ERV: Plate Existing',
            ':erv_package_Rotary-All': 'ERV: Rotary All',
            ':erv_package_Rotary-Existing': 'ERV: Rotary Existing',

            ':furnace_eff_NECB 85%% Efficient Condensing Gas Furnace': 'Furnace: 85%% Efficient Condensing',
            ':furnace_eff_NECB_Default': 'Furnace: NECB Default',

            ':nv_type_NECB_Default': 'Natural Ventilation: NECB Default',
            ':nv_type_add_nv': 'Natural Ventilation: Additional',

            ':shw_eff_NECB_Default': 'SHW: NECB Default',
            ':shw_eff_Natural Gas Direct Vent with Electric Ignition': 'SHW: NG Direct Vent + Elec Ignition',
            ':shw_eff_Natural Gas Power Vent with Electric Ignition': 'SHW: NG Power Vent + Elec Ignition',

            ':building_type_MidriseApartment': 'Building Type: Midrise Apartment',

            ':primary_heating_fuel_Electricity': 'Primary Heating Fuel: Electricity',
            ':primary_heating_fuel_ElectricityHPElecBackup': 'Primary Heating Fuel: Elec + HP Elec Backup',
            ':primary_heating_fuel_NaturalGas': 'Primary Heating Fuel: Natural Gas',
            ':primary_heating_fuel_NaturalGasHPGasBackup': 'Primary Heating Fuel: NG + HP Gas Backup',

            ':ext_roof_cond': 'Exterior Roof Conductance (W/m2.K)',
            ':ext_wall_cond': 'Exterior Wall Conductance (W/m2.K)',
            ':fdwr_set': 'Fenestration and Door-to-Wall Ratio',
            ':fixed_wind_solar_trans': 'Fixed Window Solar Transmittance',
            ':fixed_window_cond': 'Fixed Window Conductance (W/m2.K)',
            ':rotation_degrees': 'Rotation (Degrees)',
            ':srr_set': 'Skylight-to-Roof Ratio (SRR)',

            'year': 'Year',
            'month': 'Month',
            'day': 'Day',
            'hour': 'Hour',

            'drybulb': 'Outdoor Dry Bulb Temp (°C)',
            'dewpoint': 'Dew Point Temp (°C)',
            'relhum': 'Relative Humidity (%)',
            'atmos_pressure': 'Atmospheric Pressure (Pa)',
            'horirsky': 'Horizontal Infrared Radiation from Sky (W/m²)',
            'dirnorillum': 'Direct Normal Illuminance (lux)',
            'difhorillum': 'Diffuse Horizontal Illuminance (lux)',
            'winddir': 'Wind Direction (°)',
            'windspd': 'Wind Speed (m/s)',
            'presweathobs': 'Present Weather Observation',
            'snowdepth': 'Snow Depth (cm)',
            'liq_precip_depth': 'Liquid Precipitation Depth (mm)'
        }

        def run_shap_analysis(model, X_df, running_process, output_dir="D:/btap_ml/SHAP"):
            feature_rename_dict = load_feature_rename_dict()
            is_energy = running_process.lower() == config.Settings().APP_CONFIG.ENERGY
            output_type = "energy" if is_energy else "costing"
            plot_title = "Mean Absolute SHAP Values by Feature (Average Energy)" if is_energy else "Mean Absolute SHAP Values by Feature (Average Costing)"

            os.makedirs(output_dir, exist_ok=True)
            X_shap = X_df.copy()
            feature_names = X_shap.columns.tolist()
            background = shap.utils.sample(X_shap, 20, random_state=42)

            explainer = get_explainer(model, background)
            shap_values = explainer(X_shap.to_numpy())
            joblib.dump(shap_values, f"shap_values_{output_type}_avg.pkl")

            if len(shap_values.values.shape) != 3:
                raise ValueError("Expected multi-output SHAP values (3D array).")

            avg_shap_values = shap_values.values.mean(axis=2)
            avg_base_values = shap_values.base_values.mean(axis=1)

            avg_shap_explanation = shap.Explanation(
                values=avg_shap_values,
                base_values=avg_base_values,
                data=shap_values.data,
                feature_names=feature_names
            )

            renamed_feature_names = [feature_rename_dict.get(name, name) for name in feature_names]

            # SHAP summary plot
            shap.summary_plot(avg_shap_explanation, X_shap, feature_names=renamed_feature_names, show=False)
            summary_path = os.path.join(output_dir, f"shap_summary_plot_{output_type}_avg.png")
            plt.savefig(summary_path, dpi=300, bbox_inches='tight')
            plt.close()

            shap_values_df = pd.DataFrame(avg_shap_explanation.values, columns=renamed_feature_names)
            mean_abs_shap = shap_values_df.abs().mean().sort_values(ascending=False).head(20)
            plt.figure(figsize=(6, 0.3 * len(mean_abs_shap)))
            ax = mean_abs_shap.plot(kind='barh', color='royalblue', edgecolor='black')
            ax.set_yticklabels(ax.get_yticklabels(), fontsize=12)
            ax.set_xlabel("Mean |SHAP value|", fontsize=14)
            ax.set_ylabel("Feature", fontsize=14)
            plt.title(plot_title, fontsize=14)
            plt.grid(axis='x', linestyle='--', alpha=0.7)
            bar_chart_path = os.path.join(output_dir, f"shap_feature_importance_bar_chart_{output_type}_avg.png")
            plt.savefig(bar_chart_path, dpi=300, bbox_inches='tight')
            plt.close()

            print(f"SHAP summary saved: {summary_path}")
            print(f"SHAP bar chart saved: {bar_chart_path}")

        run_shap_analysis(model, X_df, running_process)


        """
        SHAP Analysis - End
        """


        if running_process.lower() == config.Settings().APP_CONFIG.ENERGY:
            X_ids[COL_NAME_DAILY_MEGAJOULES_ELEC] = [elem[0] for elem in predictions]
            X_ids[COL_NAME_DAILY_MEGAJOULES_GAS] = [elem[1] for elem in predictions]
        elif running_process.lower() == config.Settings().APP_CONFIG.COSTING:
            X_ids["Predicted cost_equipment_envelope_total_cost_per_m_sq"] = [elem[0] for elem in predictions]
            X_ids["Predicted cost_equipment_heating_and_cooling_total_cost_per_m_sq"] = [elem[1] for elem in predictions]
            X_ids["Predicted cost_equipment_lighting_total_cost_per_m_sq"] = [elem[2] for elem in predictions]
            X_ids["Predicted cost_equipment_ventilation_total_cost_per_m_sq"] = [elem[3] for elem in predictions]
            X_ids["Predicted cost_equipment_renewables_total_cost_per_m_sq"] = [elem[4] for elem in predictions]
            X_ids["Predicted cost_equipment_shw_total_cost_per_m_sq"] = [elem[5] for elem in predictions]
        """
        # Get the megajoule predictions (or call the predict.evaluate function!)
        X_ids[COL_NAME_DAILY_MEGAJOULES] = model.predict(X)
        logger.info("Scaling the predictions to their appropriate form.")
        # Transform the outputs back into their expected megajoule form
        X_ids[COL_NAME_DAILY_MEGAJOULES] = scaler_y.inverse_transform(X_ids[COL_NAME_DAILY_MEGAJOULES].values.reshape(-1, 1))
        """
        # Set X_aggregated to be X_ids such that costing can reuse code to link with buildings
        X_aggregated = X_ids
        if running_process.lower() == config.Settings().APP_CONFIG.ENERGY:
            logger.info("Preparing output file format for daily Megajoules per square meter.")
            # Convert the int date values into a standard representation, without the year
            X_ids["Date"] = X_ids["date_int"].astype(str).str.zfill(4)

            X_ids["Date"] = pd.to_datetime(X_ids["Date"], format="%m%d")
            # Replace the year with a placeholder value to clearly identify that it is unused
            X_ids["Date"] = X_ids["Date"].dt.strftime('%m/%d') + '/YYYY'

            X_ids = X_ids.drop('date_int', axis=1)
            logger.info("Preparing aggregated output in Gigajoules per square meter over the specified date range.")
            # From the daily total, generate a total for the entire start-end date in gigajoules
            X_aggregated = X_ids.drop("Date", axis=1).groupby(['Prediction Identifier'], sort=False, as_index=False).sum()
            total_days = len(pd.date_range(TEMP_YEAR + start_date, TEMP_YEAR + end_date))
            # If the averaged energy use is needed, the line below can be used
            # ... = X_aggregated[COL_NAME_DAILY_MEGAJOULES].apply(lambda r: float(r / total_days))
            X_aggregated[COL_NAME_AGGREGATED_GIGAJOULES_ELEC] = X_aggregated[COL_NAME_DAILY_MEGAJOULES_ELEC].astype('float64') / 1000.0
            X_aggregated[COL_NAME_AGGREGATED_GIGAJOULES_GAS] = X_aggregated[COL_NAME_DAILY_MEGAJOULES_GAS].astype('float64') / 1000.0

            X_aggregated = X_aggregated.drop([COL_NAME_DAILY_MEGAJOULES_ELEC, COL_NAME_DAILY_MEGAJOULES_GAS], axis=1)
        # Merge the processed building data used for training with the preprocessed building data
        if buildings_df is None:
            buildings_df, _ = preprocessing.process_building_files_batch(input_model.building_params_folder, "", "", running_process.lower() == config.Settings().APP_CONFIG.ENERGY)
        X_aggregated = pd.merge(X_aggregated, buildings_df, on='Prediction Identifier', how='left')
        # Output the predictions alongside any relevant information
        logger.info("Outputting predictions to %s.", str(output_path))
        aggregated_filename = settings.APP_CONFIG.RUNNING_COSTING_RESULTS_FILENAME
        if running_process.lower() == config.Settings().APP_CONFIG.ENERGY:
            aggregated_filename = settings.APP_CONFIG.RUNNING_AGGREGATED_RESULTS_FILENAME
            X_ids.to_csv(output_path + '/' + settings.APP_CONFIG.RUNNING_DAILY_RESULTS_FILENAME)
        X_aggregated.to_csv(output_path + '/' + aggregated_filename)

    time_taken = (time.time() - start_time)

    print(time_taken)

    return

if __name__ == '__main__':
    # Load settings from the environment
    settings = config.Settings()
    # Run the CLI interface
    typer.run(main)
