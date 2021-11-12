Preprocessing
=============

When running manually, you need to trigger the preprocessing step to prepare all the input data. Several 
inputs are required to prepare for model training:

* the simulation I/O output file used in buiding and testing the model
* the hourly energy file associated with the in_build_params file
* the simulation I/O output file used in validating the model
* the hourly energy file associated with the in_build_params_val file
* the weather data for your location of interest

# TODO: Explain how validation data was generated

The following command will perform the required preprocessing on the given inputs::

    python3 preprocessing.py --tenant standard --bucket nrcan-btap --in_build_params input_data/output_2021-10-04.xlsx --in_hour input_data/total_hourly_res_2021-10-04.csv --in_weather input_data/montreal_epw.csv --output_path output_data/preprocessing_out --in_build_params_val input_data/output.xlsx --in_hour_val input_data/total_hourly_res.csv
