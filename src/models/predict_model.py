import pydantic

from . import shared_functions


class PredictModel(pydantic.BaseModel):
    """
    Args:
        input_prefix: The input prefix to be used for all files provided.
        preprocessed_data_file: Location and name of a .json preprocessing file to be used.
        selected_features_file: Location and name of a .json feature selection file to be used.
        perform_param_search: 'yes' if hyperparameter tuning should be performed (increases runtime), 'no' if the default hyperparameters should be used.
        random_seed: Random seed to be used when training. Should not be -1 when used through the CLI.
    """
    input_prefix: str
    preprocessed_data_file: str
    selected_features_file: str
    perform_param_search: str
    random_seed: int

    @pydantic.validator("preprocessed_data_file", "selected_features_file")
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
