from typing import Optional

import pydantic

from . import shared_functions


class PreprocessingModel(pydantic.BaseModel):
    """
    Args:
        input_prefix: The input prefix to be used for all files provided.
        building_param_files: List of two building files [electricity, gas (optional)].
        energy_param_files: List of two energy files [electricity, gas (optional)].
        val_hourly_energy_file: Location and name of an energy validation file to be used if the config file is not used.
        val_building_params_file: Location and name of a building parameters validation file to be used if the config file is not used.
        hourly_energy_gas_file: Location and name of a gas energy file to be used if the config file is not used.
        building_params_gas_file: Location and name of a gas building parameters file to be used if the config file is not used.
        output_path: Where output data should be placed.
        preprocess_only_for_predictions: True if the data to be preprocessed is to be used for prediction, not for training.
        random_seed: The random seed to be used when splitting the data.
        building_params_folder: The folder location containing all building parameter files which will have predictions made on by the provided model. Only used preprocess_only_for_predictions is True.
        start_date: The start date to specify the start of which weather data is attached to the building data. Expects the input to be in the form Month_number-Day_number. Only used preprocess_only_for_predictions is True.
        end_date: The end date to specify the end of which weather data is attached to the building data. Expects the input to be in the form Month_number-Day_number. Only used preprocess_only_for_predictions is True.
        ohe_file: Location and name of a ohe.pkl OneHotEncoder file which was generated in the root of a training output folder. To be used when generating a dataset to obtain predictions for.
        cleaned_columns_file: Location and name of a cleaned_columns.json file which was generated in the root of a training output folder. To be used when generating a dataset to obtain predictions for.
    """
    input_prefix: str
    building_param_files: Optional[list]
    energy_param_files: Optional[list]
    val_hourly_energy_file: Optional[str]
    val_building_params_file: Optional[str]
    hourly_energy_gas_file: Optional[str]
    building_params_gas_file: Optional[str]
    output_path: str
    preprocess_only_for_predictions: bool
    random_seed: Optional[int]
    building_params_folder: Optional[str]
    start_date: Optional[str]
    end_date: Optional[str]
    ohe_file: Optional[str]
    cleaned_columns_file: Optional[str]

    @pydantic.validator("building_param_files", "energy_param_files", "val_hourly_energy_file", "val_building_params_file", "ohe_file", "cleaned_columns_file")
    @classmethod
    def validate_files_exist(cls, value, values):
        """
        For a filepath, add an input prefix if needed and validate
        that the file(s) exist. This function validates single string paths
        and a list of two paths, where only the first is required.

        Args:
         value: Either a single string or list of two strings which are filepaths.
         values: Other values within the class, used to get the input prefix.

        Returns:
            value: One or two string values with an added prefix, if needed.
        """
        return shared_functions.validate_files_exist(value, values)

    @pydantic.validator("energy_param_files")
    @classmethod
    def validate_building_energy_files(cls, value, values):
        """
        Ensure that any energy/building file pair for electricity or
        gas data both exist in the provided path. Gas files are optional
        and thus if any path is provided for gas files, both must be valid.
        NOTE: Only energy_parram_files is validated since the building files
        will have already been loaded and thus can be compared with the energy
        files.

        Args:
         value: A list of two strings containing the energy files provided.
         values: Other values within the class, used to get the building files.

        Returns:
            value: The validated energy file paths.
        """
        return shared_functions.validate_building_energy_files(value, values)

    @pydantic.validator("building_params_folder")
    @classmethod
    def add_input_prefix_for_batch_input(cls, value, values):
        """
        Validates that there is at least one .xlsx file
        within the specified directory and that the directory exists.

        Args:
         value: A list of a directory where the building files are located.

        Returns:
            value: The validated energy file paths.
        """
        if value is None or value == "": return value
        value = value.replace("\\", "/")
        return shared_functions.add_input_prefix_for_batch_input(values.get("input_prefix"), value)

    @pydantic.validator("start_date", "end_date")
    @classmethod
    def validate_date_strings(cls, value):
        """
        Given a date string, validate that it is in the form
        Month_number-Day_number.
        Args:
            value: A string containing a date.

        Returns:
            date_value: A verified date string.
        """
        if value == "": return value
        return shared_functions.validate_date_strings(value)
