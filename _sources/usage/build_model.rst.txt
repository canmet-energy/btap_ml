Model building
==============

There are two options when building the model: (1) with hyperparameter search, and (2) without hyperparameter search.
Hyperparameter search is controlled by the ``--param_search`` switch on ``feature_selection.py``.

With hyperparameter search::

    python3 feature_selection.py --tenant standard --bucket nrcan-btap --in_obj_name output_data/preprocessing_out --output_path output_data/feature_out --estimator_type elasticnet```python3 predict.py --tenant standard --bucket nrcan-btap --param_search no --in_obj_name output_data/preprocessing_out --features output_data/feature_out --output_path output_data/predict_out

Without hyperparameter search::

    python3 feature_selection.py --tenant standard --bucket nrcan-btap --in_obj_name output_data/preprocessing_out --output_path output_data/feature_out --estimator_type elasticnet```python3 predict.py --tenant standard --bucket nrcan-btap --param_search yes --in_obj_name output_data/preprocessing_out --features output_data/feature_out --output_path output_data/predict_out

Tensorboard
-----------

You can use the tensorboard dashboard to inspect the performance of the model.

1. Open ``notebooks/tensorboard.ipynb``.
2. Run all the contents of the notebook.
3. Navigate to the appropriate port in a web browser.

.. note::

   The tensorboard opens on a random port inside your notebook container. The URL looks like
   https://kubeflow.aaw.cloud.statcan.ca/notebook/nrcan-btap/<notebook_name>/proxy/<port>/.
