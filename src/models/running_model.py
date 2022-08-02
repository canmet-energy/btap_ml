from typing import Optional

import pydantic

from . import shared_functions


class RunningModel(pydantic.BaseModel):
    """
    Args:
        input_prefix: The input prefix to be used for all files provided.
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
    input_prefix: str
    config_file: Optional[str]
    model_file: str
    ohe_file: str
    cleaned_columns_file: str
    scaler_X_file: str
    scaler_y_file: str
    building_params_folder: str
    start_date: str
    end_date: str
    selected_features_file: str

    @pydantic.validator("model_file", "ohe_file", "cleaned_columns_file", "scaler_X_file", "scaler_y_file", "selected_features_file")
    @classmethod
    def validate_files_exist(cls, value, values):
        return shared_functions.validate_files_exist(value, values)

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
        return shared_functions.validate_date_strings(value)
