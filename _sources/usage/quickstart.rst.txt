Quickstart
==========

If you just want to run all of the steps from training a model to using the model to obtain predictions, here are the steps to run. You may be able to skip some
of these depending on if the data has already been processed.

Each of the Python files can use an input configuration .yml file which contains all inputs to be used alongside descriptions of the inputs.
If used, the input file, which will be referenced as ``input_config.yml``, will be passed when calling the training process (this file will be discussed below).
Each Python file can also use values passed to through command line arguements. All values except the config file are optional.
Thus, the config file must still be passed, but the CLI arguements will be given priority. A mix of command line and
config files can also be used.

.. note::

   The scripts for the project are in the ``src/`` folder of the repository. All CLI parameters can be viewed
   by calling the python file with ``--help`` arguement (Example: ``python train_model_pipeline.py --help``).

.. note::

   Every file can be run independently, but the recommended steps are to run one of the two pipelines presented below.

.. warning::

   Depending on the size of the input files, the amount of RAM needed can be exceedingly high. Thus, the inputs should be split
   as needed if the RAM usage is too high. Further works can add the functionality to further train a model.

Training a model
----------------

The first step to perform is to generate a trained Machine Learning model to later be used for obtaining predictions on energy or costing values.
Obtaining a trained model can be performed by calling the ``train_model_pipeline.py`` file and passing the appropriate arguements.
Assuming that only the ``input_config.yml`` file is used and located inside of the current directory, the arguement to pass is::

    python train_model_pipeline.py input_config.yml

This will preprocess all input files and any corresponding weather files which need to be retrieved (outputting various preprocessed files which can later be removed),
perform feature selection (outputting a .json file), and train the model (outputting a .h5 trained model, two .pkl files, a .json file, and a .csv file). These
steps are performed appropriately for both energy and costing in sequence.

Getting model predictions
-------------------------

With a trained Machine Learning model in the form of a .h5 file, the model will be used to obtain the predicted daily Megajoules per meter squared
and the predicted aggregated Gigajoules per meter squared for energy predictions (for a specified timeframe) for a batch of specified building files
within a specified directory. Costing predictions will then be output. The ouputs can be provided as the total energy/costing values or the
brekdowns of those totals depending on which Docker image available is used.
Obtaining the predictions from a model can be performed by calling the ``run_model.py`` file and passing the appropriate arguements.
Assuming that only the ``input_config.yml`` file is used and located inside of the current directory, the arguement to pass is::

    python run_model.py input_config.yml

This will preprocess all input files and any corresponding weather files which need to be retrieved, and obtain predictions for all input data
(outputting two energy .csv files and one costing .csv file which contain the prediction outputs alongside identifiers
for each building in the form building_file_name/building_index).

Input configuration file (input_config.yml)
-------------------------------------------

When providing an input to one of the pipelines above, the command line arguements can be used or the input configuration can be used (or a combination of the two).
The template for the ``input_config.yml`` file can be found `here <https://github.com/canmet-energy/btap_ml/blob/main/src/input_config.yml>`_.
Within this configuration file are different fields which are required for training and/or running. Each section is clearly labelled with
comments describing what the input is used for and which are mandatory. If not provided within this file, the appropriate input must be provided through
the CLI.

It is important to note that any input locations specified within this file or which are provided through the CLI are modified within the code.
Specifically, any input filename or directory provided as an input will have the Docker file's input path appended to the beginning of the value if needed.
For example, if a file named electricity_building_params.xlsx is within the root folder which has been linked to the Docker's input path
(/home/btap_ml/input), then when input to the input_config.yml file, only the value electricity_building_params.xlsx should be used in the
configuration file. When running outside of Docker, the input and output paths can be adjusted within the ``config.py`` file.

Similarly, when passing the ``input_config.yml`` file to the pipelines through the CLI, if the configuration file is at the root of the linked
input file from the Docker container, only ``input_config.yml`` will need to be passed as an arguement to the CLI. More information on
the Docker image is available `on DockerHub <https://hub.docker.com/r/juliantemp/btap_ml>`_.
