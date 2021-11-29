Preprocessing
=============

When running manually, you need to trigger the preprocessing step to prepare all the input data. Several 
inputs are required to prepare for model training:

* in_hour: The minio location and filename for the hourly energy consumption file is located. This would be the path for the electric hourly file if it exist.
* in_build_params:The minio location and filename the building simulation I/O file. This would be the path for the electric hourly file if it exist.
* in_weather: The minio location and filename for the converted  weather file to be read.
* in_hour_val: The minio location and filename for the hourly energy consumption file for the validation set, if it exist.
* in_build_params_val: The minio location and filename for the building simulation I/O file for the validation set, if it exist.
* in_hour_gas: The minio location and filename for the hourly energy consumption file is located. This would be the path for the gas hourly file if it exist.
* in_build_params_gas: The minio location and filename the building simulation I/O file. This would be the path for the gas hourly file if it exist.

The following command will perform the required preprocessing on the given inputs::

    python3 preprocessing.py --in_build_params input_data/output_elec_2021-11-05.xlsx --in_hour input_data/total_hourly_res_elec_2021-11-05.csv --in_weather input_data/montreal_epw.csv --output_path output_data/preprocessing_out --in_build_params_gas input_data/output_gas_2021-11-05.xlsx --in_hour_gas input_data/total_hourly_res_gas_2021-11-05.csv


During preprocessing, the hourly energy consumption file is transposed such that each datapoint_id has 8760 rows (365 * 24). Hence, for a simulation run containing 5000 datapoint_id, there would be 5000 * 8760 rows which would be 43.8 million rows. In order to avoid the preprocessing stage to become computationally expensive due to large datapoints created, the transposed hourly energy is is aggregated to daily energy for each datapoint_id. Similarly, the weather information is aggregated from hourly to daily, so that it can be merged with the hourly energy file. In essensece, 
the simulation I/O file(s), weather file and the hourly energy consumption file(s) are all merged to one dataframe which then splitted for training and testing purposes. 

.. Note:: The total daily energy computed for each datapoint_id is converted to Mega Joules per meter square (mj_per_m_sq), which is derived by converting the total energy used provided in the simulation I/O file to Mega Joules and then dividing the result by the building floor area. 

In the case where the validation set is not provided, the dataset provided is splitted into 70% training set, 20% test set and 10% validation set. However, when validation set is provided, the dataset is splitted only into 80% training set and 20% test set. It should be noted that the data splitting is performed using GroupShuffleSplit function which ensures that all data instances for a datapoint_id are included in the respective data set they are split into which implies a datapoint_id used for training would not have any instance in the test set.  


