Kubeflow pipeline
=================

A kubeflow pipeline exists to run the training and modelling steps automatically.

Step 1: build the image [THE COMMAND BELOW NEEDS TO BE RECHECKED AND UPDATED]::

    cd pipeline
    docker build -t btap_ml  --build-arg GIT_API_TOKEN=$env:GIT_API_TOKEN .

Step 2: run the pipeline

Check the arguements in the run.py file are correct

- in_build_params: the simulation I/O output file used in buiding and testing the model.
- in_hour: the hourly energy file associated with the in_build_params file.
- in_build_params_val: the simulation I/O output file used in validating the model.
- in_hour_val: the hourly energy file associated with the in_build_params_val file.
- in_weather: the epw file converted to csv.

.. code::

   python3 run.py
