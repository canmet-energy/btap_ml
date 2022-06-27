"""
A file containing functions which are shared by the pydantic
models.
"""
import os
import re

from .invalid_input_exception import InvalidInputException


def validate_files_exist(value, values):
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
    ERROR_MESSAGE = "Specified filepath is not valid. Check that the file exists and that the proper permissions are set."
    # If an optional value is empty, ignore it
    if value is None or value == "" or (isinstance(value, list) and value[0] is None):
        return value
    # Add the input prefix to all values as needed
    value = add_file_prefix_if_needed(values.get("input_prefix"), value)
    # For a single string, validate if it exists
    if isinstance(value, str) and not os.path.isfile(value):
        raise InvalidInputException(value=value, message=ERROR_MESSAGE)
    # For a list of two strings, validate that the first exists and the others exist
    # if a value is provided for them
    elif isinstance(value, list) and len(value) > 1 and (not os.path.isfile(value[0]) or not os.path.isfile(value[1])):
        for i in range(len(value)):
            if not os.path.isfile(value[i]) and (i == 0 or len(value[i]) > 0):
                raise InvalidInputException(value=value, message=ERROR_MESSAGE)
    return value

def validate_building_energy_files(value, values):
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
    if value[0] is None:
        return value
    value[0], value[1] = value[0].replace("\\", "/"), value[1].replace("\\", "/")
    # Check that the required electricity files are both present and valid
    if value[0] == "" or values.get("building_param_files")[0] == "":
        raise InvalidInputException(value=value, message="Specified electricity file is missing.")
    # Ensure that if any gas file is provided that both are provided and exist
    if (len(value[1]) > 0 and len(values.get("building_param_files")[1]) == 0) or (len(value[1]) == 0 and len(values.get("building_param_files")[1]) > 0):
        raise InvalidInputException(value=value, message="Both gas files must be provided, not only one.")
    return value

def add_file_prefix_if_needed(input_prefix: str, value):
    """
    Given a prefix and a string or list of strings, check
    if the string or list of strings should have the prefix
    added to the element(s).

    Args:
     input_prefix: A string prefix to be added to filepaths.
     value: Either a single string or list of strings which are filepaths.

    Returns:
        value: One or more string values with an added prefix, if needed.
    """
    # If the path is a string, check if the prefix should be added
    if isinstance(value, str) and os.path.isfile(input_prefix + value.replace("\\", "/")):
        value = input_prefix + value.replace("\\", "/")
    # If a list of strings is passed, check if any need the prefix added
    if isinstance(value, list):
        for i in range(len(value)):
            if os.path.isfile(input_prefix + value[i].replace("\\", "/")):
                value[i] = input_prefix + value[i].replace("\\", "/")
    return value

def validate_date_strings(value):
    """
    Given a date string, validate that it is in the form
    Month_number-Day_number.
    Args:
        value: A string containing a date.

    Returns:
        date_value: A verified date string.
    """
    date_value = re.search("^[1-9]{1,2}-[0-9]{1,2}$", value)
    if not date_value:
        raise InvalidInputException(value=value, message="Invalid date string provided, must be of the form Month-Day.")
    return date_value[0]

def add_input_prefix_for_batch_input(input_prefix, path):
    """
    Verify that a directory has .xlsx files AND add the prefix if needed.

    Args:
        input_prefix: A string prefix to be added to filepaths.
        path: A directory containing one or more building files.

    Returns:
        path: A verified directory containing one or more .xlsx file
    """
    if os.path.isdir(input_prefix + path):
        path = input_prefix + path
    elif not os.path.isdir(path):
        raise InvalidInputException(value=value, message="The specified building directory does not exist or cannot be accessed.")
    contains_xlsx_files = False
    for filename in os.listdir(path):
        # Verify that there is at least one .xlsx file to be used
        # NOTE: Only the file extension is validated, not the contents
        if filename.endswith(".xlsx"):
            contains_xlsx_files = True
    if not contains_xlsx_files:
        raise InvalidInputException(value=value, message="The specified building directory contains no .xlsx files.")
    return path
