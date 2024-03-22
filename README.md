# BTAP Surrogate Model
Advances in clean technologies and building practices can make new buildings “net-zero energy”, meaning they require so little energy they could potentially rely on their own renewable energy supplies for all of their energy needs.

Through research and development, technology costs continue to fall, and government and industry efforts and investments will accelerate that trend. These advances, supported by a model “net-zero energy ready” building code, will enable all builders to adopt these practices and lower lifecycle costs for homeowners.’

The goal is to develop surrogate models that generate data to inform in the design of near-net-zero energy buildings in Canada.

# Documentation

Refer to the [documentation](docs/) to see how to install and run the model.

# Docker Image

The Docker image is named juliantemp/btap_ml:latest and is available at [Docker Hub](https://hub.docker.com/r/juliantemp/btap_ml). The image documentation on Docker Hub highlights how the image can be instantiated and how to use the image.

Notes:
- At present, the sphinx version must be changed within requirements.txt and requirements-dev.txt to be version 7.1.2 due to compatibility issues when building the image.
- The image can be built in either docker or podman with the appropriate build commands:       
  `docker build -t juliantemp/btap_ml:latest`    
  OR `podman build -t juliantemp/btap_ml:latest`        
- Docker Desktop can be used on Windows to easily manage and deploy the updated image.
- For Podman, the image can be pushed to the docker repository through the following series of commands:          
  `podman login docker.io`, where the user logs into their docker account              
  `podman tag localhost/juliantemp/btap_ml docker.io/juliantemp/btap_ml`                   
  `podman push docker.io/juliantemp/btap_ml`         

## Structure
- Block Storage Guide.ipynb: sample notebook of how to access minio using s3f3. Note that direct minio use has been decommissioned starting May 6, 2022 and now requires a drive to be mounted before running.
- tensorboard.ipynb: use this notebook to start the tensorboard dashboard for metrics visualization and scrutinization of the surrogate model.
- src: Contains all source code used in building the surrogate model.
    - train_model_pipeline.py: runs all preprocessing steps for the provided input data, using the outputs to train a surrogate model for future use
    - run_model.py: given a dataset and trained model, outputs the energy use predictions for the dataset
    - prepare_weather.py: downloads a specified weather file and saves it as a .parquet file in the specified output directory
    - preprocessing.py: downloads all the dataset from minio, preprocess the data, split the data into train, test and validation set.
    - feature_selection.py: use the output from preprocoessing to extract the features that would be used in building the surrogate model
    - predict.py: builds the surrogate model using the preprocessed data and the selected features described above

# License

Unless otherwise noted, the source code of this project is covered under Crown Copyright, Government of Canada, and is distributed under the [GNU GPL v3 License](LICENSE).

The Canada wordmark and related graphics associated with this distribution are protected under trademark law and copyright law. No permission is granted to use them outside the parameters of the Government of Canada's corporate identity program. For more information, see [Federal identity requirements](https://www.canada.ca/en/treasury-board-secretariat/topics/government-communications/federal-identity-requirements.html).
