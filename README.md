# BTAP Surrogate Model
Advances in clean technologies and building practices can make new buildings “net-zero energy”, meaning they require so little energy they could potentially rely on their own renewable energy supplies for all of their energy needs. 

Through research and development, technology costs continue to fall, and government and industry efforts and investments will accelerate that trend. These advances, supported by a model “net-zero energy ready” building code, will enable all builders to adopt these practices and lower lifecycle costs for homeowners.’

The goal is to develop surrogate models that generate data to inform in the design of near-net-zero energy buildings in Canada.

# Documentation

Refer to the [documentation](docs/) to see how to install and run the model.

## Structure
- Block Storage Guide.ipynb: sample notebook of how to access minio using s3f3. 
- tensorboard.ipynb: use this notebook to start the tensorboard dashboard for metrics visualization and scrutinization of the surrogate model.
- src: Contain all source code used in building the surrogate model. 
    - preprocessing.py: downloads all the dataset from minio, preprocess the data, split the data into train, test and validation set. 
    - feature_selection.py: use the output from preprocoessing to extract the features that would be used in building the surrogate model
    - predict.py: builds the surrogate model using the preprocessed data and the selected features described above
    - plot.py: contains functions used to create plots. 
    
### License

Unless otherwise noted, the source code of this project is covered under Crown Copyright, Government of Canada, and is distributed under the [GNU GPL v3 License](LICENSE).

The Canada wordmark and related graphics associated with this distribution are protected under trademark law and copyright law. No permission is granted to use them outside the parameters of the Government of Canada's corporate identity program. For more information, see [Federal identity requirements](https://www.canada.ca/en/treasury-board-secretariat/topics/government-communications/federal-identity-requirements.html).
