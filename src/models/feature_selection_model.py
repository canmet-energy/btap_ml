from typing import Optional

import pydantic

from . import shared_functions


class FeatureSelectionModel(pydantic.BaseModel):
    """
    Args:
        input_prefix: The input prefix to be used for all files provided.
        preprocessed_data_file: Location and name of a .json preprocessing file to be used if the preprocessing is skipped.
        estimator_type: The type of feature selection to be performed. The default is lasso, which will be used if nothing is passed.
                        The other options are 'linear', 'elasticnet', and 'xgb'.
    """
    input_prefix: str
    preprocessed_data_file: str
    estimator_type: Optional[str]

    @pydantic.validator("preprocessed_data_file")
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
