The output from the data preparation and feature selection steps are used as input into this stage by running::

    $ python3 predict.py --param_search yes --in_obj_name output_data/preprocessing_out --features output_data/feature_out --output_path output_data/predict_out

The parameters to the above script are documented in the :py:mod:`predict`.
