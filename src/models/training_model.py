from typing import Optional

import pydantic

from . import shared_functions


class TrainingModel(pydantic.BaseModel):
    """
    Args:
        input_prefix: The input prefix to be used for all files provided.
        config_file: Location of the .yml config file (default name is input_config.yml).
        random_seed: Random seed to be used when training.
        building_param_files: List of two building files [electricity, gas (optional)].
        energy_param_files: List of two energy files [electricity, gas (optional)].
        val_hourly_energy_file: Location and name of an energy validation file to be used if the config file is not used.
        val_building_params_file: Location and name of a building parameters validation file to be used if the config file is not used.
        skip_file_preprocessing: True if the .json preprocessing file generation should be skipped,
                                 where the preprocessed_data_file input is used, False if the preprocessing file generation should be performed.
        preprocessed_data_file: Location and name of a .json preprocessing file to be used if the preprocessing is skipped.
        estimator_type: The type of feature selection to be performed. The default is lasso, which will be used if nothing is passed.
                        The other options are 'linear', 'elasticnet', and 'xgb'.
        skip_feature_selection: True if the .json feature selection file generation should be skipped,
                                where the selected_features_file input is used, False if the feature selection file generation should be performed.
        selected_features_file: Location and name of a .json feature selection file to be used if the feature selection is skipped.
        perform_param_search: 'yes' if hyperparameter tuning should be performed (increases runtime), 'no' if the default hyperparameters should be used.
        skip_model_training: True if the model training should be skipped. Useful if only the preprocessing steps should be performed.
    """
    input_prefix: str
    config_file: Optional[str]
    random_seed: int
    building_param_files: list
    energy_param_files: list
    val_hourly_energy_file: Optional[str]
    val_building_params_file: Optional[str]
    skip_file_preprocessing: bool
    preprocessed_data_file: Optional[str]
    estimator_type: Optional[str] = ""
    skip_feature_selection: bool
    selected_features_file: Optional[str]
    perform_param_search: str
    skip_model_training: bool

    @pydantic.validator("building_param_files", "energy_param_files", "val_hourly_energy_file", "val_building_params_file", "preprocessed_data_file", "selected_features_file")
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
