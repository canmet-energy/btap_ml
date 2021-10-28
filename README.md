# BTAP Surrogate Model
Advances in clean technologies and building practices can make new buildings “net-zero energy”, meaning they require so little energy they could potentially rely on their own renewable energy supplies for all of their energy needs. 

Through research and development, technology costs continue to fall, and government and industry efforts and investments will accelerate that trend. These advances, supported by a model “net-zero energy ready” building code, will enable all builders to adopt these practices and lower lifecycle costs for homeowners.’

The goal is to develop surrogate models that generate data to inform in the design of near-net-zero energy buildings in Canada.


## About this Repo

This repo contains the python Notebooks and codes for the entire surrogate modelling for a full service restaurant. 

## Requirements

* Python 3
* Access to Minio
* Docker installed and running on your computer
* A git client
* Install the requirement packages
    ```
    pip3 install -r requirements.txt
    ```


 ## How to run this app

We suggest you create a virtual environment for running the surrogate model with Python and Clone this repository. 

```
cd c:\users\bukola
git clone https://github.com/canmet-energy/btap_ml.git
cd btap_ml
```

## Structure
- Block Storage Guide.ipynb: sample notebook of how to access minio using s3f3. 
- tensorboard.ipynb: use this notebook to start the tensorboard dashboard for metrics visualization and scrutinization of the surrogate model.
- src: Contain all source code used in building the surrogate model. 
    - preprocessing.py: downloads all the dataset from minio, preprocess the data, split the data into train, test and validation set. 
    - feature_selection.py: use the output from preprocoessing to extract the features that would be used in building the surrogate model
    - predict.py: builds the surrogate model using the preprocessed data and the selected features described above
    - plot.py: contains functions used to create plots. 
    
## how to use

Option 1: Run the surrogate model without using the kubeflow pipeline.

```
cd src
```

- **step 1)** Preprocessing
    - You need to type the command below into a terminal
    * Notes: check to ensure the file paths specified for the arguements below exist in minio
        - in_build_params: the simulation I/O output file used in buiding and testing the model.
        - in_hour: the hourly energy file associated with the in_build_params file.
        - in_build_params_val: the simulation I/O output file used in validating the model.
        - in_hour_val: the hourly energy file associated with the in_build_params_val file.
        - in_weather: the epw file converted to csv. 
    ```
    python3 preprocessing.py --tenant standard --bucket nrcan-btap --in_build_params input_data/output_2021-10-04.xlsx --in_hour input_data/total_hourly_res_2021-10-04.csv --in_weather input_data/montreal_epw.csv --output_path output_data/preprocessing_out --in_build_params_val input_data/output.xlsx --in_hour_val input_data/total_hourly_res.csv
    ```

- **step 2)** Feature Selection
    - Run the command below from a terminal
     ```
    python3 feature_selection.py --tenant standard --bucket nrcan-btap --in_obj_name output_data/preprocessing_out --output_path output_data/feature_out --estimator_type elasticnet
    ```
   
- **step 3)** Building the Surrogate Model:
    Option A: No hyperparameter search.
    - The surrogate model can be built usind default parameters by setting the value of --param_search as no
    ```
    python3 feature_selection.py --tenant standard --bucket nrcan-btap --in_obj_name output_data/preprocessing_out --output_path output_data/feature_out --estimator_type elasticnet```python3 predict.py --tenant standard --bucket nrcan-btap --param_search no --in_obj_name output_data/preprocessing_out --features output_data/feature_out --output_path output_data/predict_out 
    ```
    Option B: With hyperparameter search.
    - The surrogate model can be built usind default parameters by setting the value of --param_search as yes
    ```
    python3 feature_selection.py --tenant standard --bucket nrcan-btap --in_obj_name output_data/preprocessing_out --output_path output_data/feature_out --estimator_type elasticnet```python3 predict.py --tenant standard --bucket nrcan-btap --param_search yes --in_obj_name output_data/preprocessing_out --features output_data/feature_out --output_path output_data/predict_out 
    ```
    
- **step 4)** tensorboard (optional):    
    - run tensorboard.ipynb
    - using the port opened from tensorboard.ipynb, open the [Tensorboard Dashboard] (https://git-scm.com/downloadshttps://kubeflow.aaw.cloud.statcan.ca/notebook/nrcan-btap/reg-cpu-notebook/proxy/6007/)


Option 2: Running the surrogate model as a kubeflow pipeline. 
- **step 1)** Build Image [THE COMMAND BELOW NEEDS TO BE RECHECKED AND UPDATED]
    ```
    cd pipeline
    docker build -t btap_ml  --build-arg GIT_API_TOKEN=$env:GIT_API_TOKEN .
    ```
    
- **step 2)** Run the pipeline
    - Check the arguements in the run.py file are correct
        - in_build_params: the simulation I/O output file used in buiding and testing the model.
        - in_hour: the hourly energy file associated with the in_build_params file.
        - in_build_params_val: the simulation I/O output file used in validating the model.
        - in_hour_val: the hourly energy file associated with the in_build_params_val file.
        - in_weather: the epw file converted to csv. 
    ```
    python3 run.py
    ```
    
                             
 



Copyright 2021, Data Science Division - Statistics Canada | Developer: Bukola Ishola