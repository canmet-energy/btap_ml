import os
import time
import datetime
from math import sqrt
from pathlib import Path

import tensorflow as tf
import tensorflow.keras as keras
from keras import backend as K
from keras import regularizers  # for l2 regularization
from keras.callbacks import EarlyStopping
from keras.layers.core import Dense, Dropout, Flatten
from keras.models import Sequential
from keras_tuner import Hyperband
from matplotlib import pyplot as plt
from sklearn import metrics
from sklearn.ensemble import RandomForestRegressor
from tensorboard.plugins.hparams import api as hp
from tensorflow.keras import layers
from xgboost import XGBRegressor

from sklearn.model_selection import RandomizedSearchCV, GridSearchCV

import config
import plot as pl
import preprocessing
from models.predict_model import PredictModel

def det_coeff(y_true, y_pred):
    """
    Used to compute the R^2 (coefficient of determination)

    Args:
        y_test: y testset
        y_pred: y predicted value from the model
    Returns:
        R^2 score from comparing the y_test and y_pred values
    """
    u = K.sum(K.square(y_true - y_pred))
    v = K.sum(K.square(y_true - K.mean(y_true)))
    
    return K.ones_like(v) - (u / v)

def rmse_loss(y_true, y_pred):
    """
    RMSE loss function

    Args:
        y_true: y testset
        y_pred: y predicted value from the model
    Returns:
        RMSE loss from comparing the y_test and y_pred values
    """

    return K.sqrt(tf.reduce_mean(K.square(y_pred - y_true)))

def tune_mlp(X_train, y_train, X_test, y_test, col_length, output_nodes, selected_feature, output_path, random_seed, process_type):
    """
    DECOMMISIONED: May need updates for multi-outputs
    Using the set of hyperparameter combined,the model built is used to make predictions.

    Args:
        X_train: X train set
        y_train: y train set
        X_test: X test set
        y_test: y test set
        col_length: The number of input nodes
        output_nodes: The number of output nodes
        selected_feature: selected features that would be used to build the model
        output_path: Where the output files should be placed.
        random_seed: The random seed to be used
    Returns:
       Model built from the set of hyperparameters combined.
    """
    # Create the output directories if they do not exist
    parameter_search_path = str(Path(output_path).joinpath("parameter_search"))
    log_path = str(Path(parameter_search_path).joinpath("btap"))

    config.create_directory(parameter_search_path)
    config.create_directory(log_path)

    if process_type == 'energy':      
        BATCH_SIZE = 1024
    else:
        BATCH_SIZE = 32

    print(col_length)
    print(output_nodes)

    stop_early = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=10)

    # Define the d
    tuner = Hyperband(lambda hp: model_builder(hp, col_length, output_nodes),
                      objective='val_loss',
                      max_epochs=100,
                      overwrite=True,
                      factor=3,
                      directory=parameter_search_path,
                      project_name='btap',
                      seed=random_seed)

    tuner.search(X_train,
                 y_train,
                 epochs=100,
                 batch_size=BATCH_SIZE,
                 use_multiprocessing=True,
                 validation_split=0.10)

    tuner.search_space_summary()
    
    # Get the optimal hyperparameters
    best_hps = tuner.get_best_hyperparameters(num_trials=1)[0]

    print(f"""
            The optimal hyperparameters are:
            - Activation function: {best_hps.get('activation')}
            - Number of layers: {best_hps.get('num_layers')}
            - Units in each layer: {[best_hps.get(f'units_{i}') for i in range(best_hps.get('num_layers'))]}
            - Dropout rate: {best_hps.get('dropout_rate')}
            - Learning rate: {best_hps.get('learning_rate')}
            - Optimizer: {best_hps.get('optimizer')}
            """)

    logdir = os.path.join(log_path, datetime.datetime.now().strftime("%Y%m%d-%H%M%S"))
    hist_callback = tf.keras.callbacks.TensorBoard(logdir,
                                                   histogram_freq=1,
                                                   embeddings_freq=1,
                                                   update_freq='epoch',
                                                   write_graph=True,
                                                   write_steps_per_second=False
                                                   )

    # Build the model with the optimal hyperparameters
    model = tuner.hypermodel.build(best_hps)
    history = model.fit(X_train,
                        y_train,
                        epochs=100,
                        batch_size=BATCH_SIZE,
                        validation_split=0.10,
                        callbacks=[hist_callback],
                        )
    pl.shared_learning_curve_plot(history)

    '''
    val_acc_per_epoch = history.history['mae']
    best_epoch = val_acc_per_epoch.index(max(val_acc_per_epoch)) + 1
    print('Best epoch: %d' % (best_epoch,))

    # Re-instantiate the hypermodel and train it with the optimal number of epochs from above.
    hypermodel = tuner.hypermodel.build(best_hps)
    hypermodel.fit(X_train, y_train, epochs=best_epoch, validation_split=0.2)
    '''
    return model

def model_builder(hp, col_length, output_nodes):
    """
    Builds the MLP model that would be used to search for hyperparameter.
    The hyperparameters search includes activation, number of hidden layers, number of , dropout_rate, learning_rate, and optimizer

    Args:
        hp: hyperband object with different hyperparameters to be checked.
        col_length: The number of input nodes
        output_nodes: The number of output nodes
    Returns:
        Model will be built based on the different hyperparameter combinations.
    """
    model = keras.Sequential()
    model.add(keras.layers.Flatten())

    hp_activation= hp.Choice('activation', values=['relu'])
    for i in range(hp.Int("num_layers", 1, 3)):
        model.add(layers.Dense(
            units=hp.Int("units_" + str(i), min_value=8, max_value=4096, step=8),
            activation=hp_activation,
            input_shape=(col_length, ),
            kernel_initializer='normal',
            ))
        model.add(Dropout(hp.Choice('dropout_rate', values=[0.0, 0.05, 0.1, 0.2, 0.3])))
    model.add(Dense(output_nodes, activation='linear'))

    hp_learning_rate = hp.Choice('learning_rate', values=[1e-3, 0.0005, 1e-4, 0.00005, 1e-5])
    hp_optimizer = hp.Choice('optimizer', values=['adam', 'sgd', 'rmsprop'])

    if hp_optimizer == "adam":
        optimizer = tf.optimizers.Adam(learning_rate=hp_learning_rate)
    elif hp_optimizer == "sgd":
        optimizer = tf.optimizers.SGD(learning_rate=hp_learning_rate)
    else:
        optimizer = tf.optimizers.RMSprop(learning_rate=hp_learning_rate)

    # Compile the model with the optimizer and learning rate specified in hparams
    model.compile(optimizer=optimizer,
                  loss=rmse_loss,
                  metrics=['mae', 'mse','mape', det_coeff])

    return model

def tune_gradient_boosting(X_train, y_train, output_path):
    """
    Perform grid search to tune the gradient boosting models to find the optimal hyperparameter values.

    Args:
        X_train: X trainset
        y_train: y trainset
        X_test: X testset
        y_test: y testset
        y_test_complete: dataframe containing the target variable with corresponding datapointid for the test set
        scalery: y scaler used to transform the y values to the original scale
        X_validate: X validation set
        y_validate: y validation set
        y_validate_complete: dataframe containing the target variable with corresponding datapointid for the validation set
        output_path: Where the outputs will be placed
        path_elec: Filepath of the electricity building file which has been used
        path_gas: Filepath of the gas building file, if it has been used (pass nothing otherwise)
        val_building_path: Filepath of the validation building file, if it has been used (pass nothing otherwise).
        process_type: Either 'energy' or 'costing' to specify the operations to be performed.
    Returns:
        Model will be built based on the different hyperparameter combinations.
    """
    # Create the output directories if they do not exist
    parameter_search_path = str(Path(output_path).joinpath("parameter_search"))
    btap_log_path = str(Path(parameter_search_path).joinpath("btap"))

    config.create_directory(parameter_search_path)
    config.create_directory(btap_log_path)

    # Specify the hyperparameter search space and tune the model using grid search.
    param_grid = {
        'estimator__n_estimators' : [100, 250, 500],
        'estimator__max_depth': [5, 10, 15],
        'estimator__learning_rate': [0.01, 0.05, 0.1],
        'estimator__subsample': [0.5, 0.7, 1],
        #'estimator__min_child_weight': [1, 3, 5],
    }

    model = XGBRegressor(random_state=42, n_jobs=-1)

    grid_search = GridSearchCV(model,
                               param_grid,
                               scoring='neg_mean_squared_error',
                               cv=3,
                               verbose=2,
                               n_jobs=-1)

    grid_search.fit(X_train, y_train, verbose=True)

    # Output the best hyperparameter values and the CV score
    best_model = grid_search.best_estimator_
    
    print("Best parameters:", grid_search.best_params_)
    print("Best CV score (neg MSE):", grid_search.best_score_)

    return best_model

def tune_rf(X_train, y_train, output_path):
    """
    Perform randomized search to tune the random forest models to find the optimal hyperparameter values.

    Args:
        X_train: X trainset
        y_train: y trainset
        X_test: X testset
        y_test: y testset
        y_test_complete: dataframe containing the target variable with corresponding datapointid for the test set
        scalery: y scaler used to transform the y values to the original scale
        X_validate: X validation set
        y_validate: y validation set
        y_validate_complete: dataframe containing the target variable with corresponding datapointid for the validation set
        output_path: Where the outputs will be placed
        path_elec: Filepath of the electricity building file which has been used
        path_gas: Filepath of the gas building file, if it has been used (pass nothing otherwise)
        val_building_path: Filepath of the validation building file, if it has been used (pass nothing otherwise).
        process_type: Either 'energy' or 'costing' to specify the operations to be performed.
    Returns:
        annual_metric: predicted value for each datapooint_id is summed to calculate the annual energy consumed and the loss value from the testset prediction,
        output_df: merge of y_pred, y_test, datapoint_id, the final dataframe showing the model output using the testset
        val_metric:evaluation results containing the loss value from the validationset prediction,
        val_annual_metric:predicted value for each datapooint_id is summed to calculate the annual energy consumed and the loss value from the validationset prediction,,
        output_val_df: merge of y_pred, y_validate, datapoint_id, the final dataframe showing the model output using the validation set
    """
    # Create the output directories if they do not exist
    parameter_search_path = str(Path(output_path).joinpath("parameter_search"))
    btap_log_path = str(Path(parameter_search_path).joinpath("btap"))

    config.create_directory(parameter_search_path)
    config.create_directory(btap_log_path)

    rf_search_space = {
        'n_estimators': [100, 200, 300],
        'max_depth': [None, 5, 10, 15],
        'min_samples_split': [2, 4, 8],
        'min_samples_leaf': [1, 2, 4],
    }

    model = RandomForestRegressor(random_state=42)

    random_search = RandomizedSearchCV(model,
                                       param_distributions=rf_search_space,
                                       scoring='neg_mean_squared_error',
                                       n_iter=70,
                                       cv=3,
                                       n_jobs=-1,
                                       verbose=3,
                                       random_state=42)

    random_search.fit(X_train, y_train)

    # Output the best hyperparameter values
    best_model = random_search.best_estimator_

    print(random_search.best_params_)

    return best_model
