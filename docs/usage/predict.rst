The output from the preprocessing and feature selection outputs are used as input to ``predict.py``
to derive a trained model::

    python predict.py input_config.yml

This script will output the trained model as a .h5 file alongisde information on the training and testing
performed for analysis. The outputs from this step will be used when obtaining predictions with the model.

The parameters to the above script are documented in the :py:mod:`predict`.
