Re-run Model
================================
If the data is already preprocessed and the features are selected with output_files from preprocessing and feature_selection already existing in minio; build the surrogate model and predict by::

    python3 predict.py --param_search no --in_obj_name output_data/preprocessing_out --features output_data/feature_out --output_path output_data/predict_out


.. note::
    The results from the prediction is available in minio in the output_path specified above.
