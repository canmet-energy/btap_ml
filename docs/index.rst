.. Building Technology Assessment Platform: Machine Learning Implementation documentation master file, created by
   sphinx-quickstart on Mon Oct 25 13:53:23 2021.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

Building Technology Assessment Platform (BTAP): Machine Learning Implementation
===============================================================================

Based on a whole building energy simulation engine, BTAP calculates capital and operating costs, energy consumption
and demand, and related GHG emissions for over 60,000 difference reference housing and building models. This supports
the development of the next generation of building codes for new construction and the development of the first code on
alternations to existing buildings. It also supports the building industry in the cost effective design of solutions
to meet energy consumption, cost, and GHG targets.

Through the use of surrogate models, machine learning is being used to try to improve the overall processing time
associated with calculating such a large solution space. Even with High Performance Computing, calculating the entire
stock of models is estimated to take 57 centuries. Surrogate models significantly reduce the time and resources
required to produce usable outputs.

After installation, the processing can either take the form of running a model training pipeline or of running a trained model to obtain predictions for
both energy and costing:

.. graphviz::
   :name: Training a model

   digraph G {
      bgcolor=transparent;
      rankdir=LR;

      start -> preprocess -> features -> build -> end;

      start [shape=Mdiamond];
      preprocess [label="Preprocess input data and prepare weather data"];
      features [label="Feature selection"];
      build [label="Build the model"];
      end [shape=Msquare];
    }

.. graphviz::
   :name: Obtaining predictions

   digraph G {
      bgcolor=transparent;
      rankdir=LR;

      start -> preprocess -> run -> end;

      start [shape=Mdiamond];
      preprocess [label="Preprocess input data and prepare weather data"];
      run [label="Obtain predictions"];
      end [shape=Msquare];
    }

Contents
--------

.. toctree::
   :maxdepth: 2
   :caption: Setup

   installation
   aaw_setup

Once installation is complete, a good place to get an overview of the complete process can be found in :doc:`usage/retrain`.

.. toctree::
   :maxdepth: 2
   :caption: Usage

   usage/quickstart
   usage/retrain
   usage/run_model


.. toctree::
   :maxdepth: 1
   :caption: API

   api/config
   api/feature_selection
   api/predict
   api/prepare_weather
   api/preprocessing
   api/run_model
   api/train_model_pipeline


Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`
