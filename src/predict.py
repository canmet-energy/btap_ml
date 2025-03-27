"""
Uses the output from preprocessing and feature selection to build, train, and evaluate the model.
"""
import csv
import datetime
import json
import logging
import os
import time
from math import sqrt
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import tensorflow as tf
import tensorflow.keras as keras

import typer

from keras import backend as K
from keras.callbacks import EarlyStopping
from keras.layers.core import Dense, Dropout, Flatten
from keras.models import Sequential
from keras_tuner import Hyperband

from matplotlib import pyplot as plt

from sklearn import metrics
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler
from tensorboard.plugins.hparams import api as hp
from tensorflow.keras import layers
from xgboost import XGBRegressor

import config
import plot as pl
import preprocessing
from models.predict_model import PredictModel

from tune_model import det_coeff, rmse_loss, tune_mlp, tune_gradient_boosting, tune_rf

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def score(y_test, y_pred):
    """
    Used to compute the mse, rmse and mae scores

    Args:
        y_test: y testset
        y_pred: y predicted value from the model
    Returns:
       mse, rmse, mae and mape scores from comparing the y_test and y_pred values
    """
    mse = metrics.mean_squared_error(y_test, y_pred)
    mae = metrics.mean_absolute_error(y_test, y_pred)
    rmse = sqrt(mse)
    mape = metrics.mean_absolute_percentage_error(y_test, y_pred)
    scores = {"mse": mse,
              "rmse": rmse,
              "mae": mae,
              "mape": mape}
    return scores

def convert_dataframe_to_annual(df):
    """
    Converts a dataframe of daily predictions into one with annual predictions (energy only).

    Args:
        df: The dataframe being transformed.
    Returns:
        updated_df: The updated dataframe.
    """
    # Sum the daily values
    updated_df = df.groupby(['datapoint_id']).sum()

    # Scale the values to GigaJoules per meter squared
    updated_df['energy_elec'] = updated_df['energy_elec'].apply(lambda r : float((r*1.0)/1000))
    updated_df['energy_gas'] = updated_df['energy_gas'].apply(lambda r : float((r*1.0)/1000))
    updated_df['y_pred_elec_transformed'] = updated_df['y_pred_elec_transformed'].apply(lambda r : float((r*1.0)/1000))
    updated_df['y_pred_gas_transformed'] = updated_df['y_pred_gas_transformed'].apply(lambda r : float((r*1.0)/1000))

    return updated_df

def compute_building_weather_errors(df, actual_label, prediction_label):
    """
    Calculate the absolute difference and squared difference between the predicted and actual energy/costing use, and
    group by the building type and epw file to compute the means for BOTH the actual and predicted energy/costing

    Args:
        df: The dataframe being manipulated.
        actual_label: The string used to describe the class being predicted (i.e. electricity, gas, ...).
        prediction_label: The string used to describe the prediction which a model outputs (i.e. predicted_electricity, ...).

    Returns:
        df: The updated dataframe.
        building_errors: The errors for each building type.
        climate_errors: The errors for each climate zone.
    """
    df['abs_difference_elec'] = abs(df[actual_label] - df[prediction_label])
    df['mse_difference_elec'] = (df[actual_label] - df[prediction_label]) ** 2

    # 2. Group by building type and epw file, then compute the means for BOTH the actual and predicted energy
    building_errors = df[[":building_type", actual_label, prediction_label, "abs_difference_elec", "mse_difference_elec"]].groupby([':building_type'], sort=False, as_index=False).mean()
    climate_errors = df[[":epw_file", actual_label, prediction_label, "abs_difference_elec", "mse_difference_elec"]].groupby([':epw_file'], sort=False, as_index=False).mean()

    return df, building_errors, climate_errors

def evaluate(model, X_test, y_test, scalery, X_validate, y_validate, y_test_complete, y_validate_complete, path_elec, path_gas, val_building_path, process_type):
    """
    The model selected with the best hyperparameter is used to make predictions.

    Args:
        model: model built from training
        X_test: X testset
        y_test: y testset
        scalery: y scaler used to transform the y values to the original scale
        X_validate: X validationset
        y_validate: y validationset
        y_test_complete: test dataset
        y_validate_complete: validation dataset
        path_elec: Filepath of the electricity building file which has been used
        path_gas: Filepath of the gas building file, if it has been used (pass nothing otherwise)
        val_building_path: Filepath of the validation building file, if it has been used (pass nothing otherwise).
        process_type: Either 'energy' or 'costing' to specify the operations to be performed.
    Returns:
        metric: evaluation results containing the loss value from the testset prediction,
        annual_metric: predicted value for each datapooint_id is summed to calculate the annual energy consumed and the loss value from the test set prediction,
        output_df: merge of y_pred, y_test, datapoint_id, the final dataframe showing the model output using the test set
        val_metric:evaluation results containing the loss value from the validationset prediction,
        val_annual_metric:predicted value for each datapooint_id is summed to calculate the annual energy consumed and the loss value from the validationset prediction,,
        output_val_df: merge of y_pred, y_validate, datapoint_id, the final dataframe showing the model output using the validation set
        output_df_average_predictions_buildings: The mean energy predictions and actual energy values per building type in the test set
        output_df_average_predictions_climates: The mean energy predictions and actual energy values per climate zone in the test set
        output_val_df_average_predictions_buildings: The mean energy predictions and actual energy values per building type in the validation set
        output_val_df_average_predictions_climates:  The mean energy predictions and actual energy values per climate zone in the validation set
    """
    test_predictions = model.predict(X_test)
    validate_predictions = model.predict(X_validate)

    ENERGY_PREDICTIONS = [
        'y_pred_elec',
        'y_pred_gas'
    ]
    ENERGY_ACTUAL = [
        'energy_elec',
        'energy_gas'
    ]
    COSTING_PREDICTIONS = [
        'y_pred_envelope',
        'y_pred_heat_cool',
        'y_pred_lighting',
        'y_pred_ventilation',
        'y_pred_renewables',
        'y_pred_shw'
    ]
    COSTING_ACTUAL = [
        'cost_equipment_envelope_total_cost_per_m_sq',
        'cost_equipment_heating_and_cooling_total_cost_per_m_sq',
        'cost_equipment_lighting_total_cost_per_m_sq',
        'cost_equipment_ventilation_total_cost_per_m_sq',
        'cost_equipment_renewables_total_cost_per_m_sq',
        'cost_equipment_shw_total_cost_per_m_sq'
    ]
    TEST_PREFIX = 'test_'
    VAL_PREFIX = 'val_'
    DAILY_PREFIX = 'daily_metric_'
    ANNUAL_PREFIX = 'annual_metric_'
    TRANSFORMATION_SUFFIX = '_transformed'
    BUILDING_ERRORS_LABEL = 'building_errors'
    CLIMATE_ERRORS_LABEL = 'climate_errors'

    # Choose the prediction set to work with
    prediction_set = ENERGY_PREDICTIONS
    actual_set = ENERGY_ACTUAL
    if process_type.lower() == config.Settings().APP_CONFIG.COSTING:
        prediction_set = COSTING_PREDICTIONS
        actual_set = COSTING_ACTUAL

    # For each initial prediction, map it to the appropriate dictionary entry
    for i in range(len(prediction_set)):
        y_test[prediction_set[i]] = [elem[i] for elem in test_predictions]
        y_validate[prediction_set[i]] = [elem[i] for elem in validate_predictions]

    # Transform each prediction to be the scaled appropriately
    test_predictions_transformed = scalery.inverse_transform(test_predictions)
    validate_predictions_transformed = scalery.inverse_transform(validate_predictions)

    # For each scaled prediction, map it to the appropriate dictionary entry
    for i in range(len(prediction_set)):
        y_test[prediction_set[i] + TRANSFORMATION_SUFFIX] = [elem[i] for elem in test_predictions_transformed]
        y_validate[prediction_set[i] + TRANSFORMATION_SUFFIX] = [elem[i] for elem in validate_predictions_transformed]

    # Track each score within a dictionary
    test_val_scores = {}

    costing_test_actual = 0
    costing_test_predicted = 0
    energy_test_actual = 0
    energy_test_predicted = 0

    energy_rmse = 0
    costing_rmse = 0
    
    # Loop through all outputs and track how well the model does at predicting them for daily predictions or costing annual prediction
    annual_daily_label = DAILY_PREFIX
    if process_type.lower() == config.Settings().APP_CONFIG.COSTING:
        annual_daily_label = ANNUAL_PREFIX
    for actual_label, predicted_label in zip(actual_set, prediction_set):
        # Retrieve the score and store it with the appropriate title for both the test and validation sets
        performance_score = score(y_test[actual_label], y_test[predicted_label + TRANSFORMATION_SUFFIX])
        print("[" + actual_label, TEST_PREFIX + "score loss, test mae, test mse]:", performance_score)
        test_val_scores[TEST_PREFIX + annual_daily_label + actual_label] = performance_score

        performance_score = score(y_validate[actual_label], y_validate[predicted_label + TRANSFORMATION_SUFFIX])
        print("[" + actual_label, VAL_PREFIX + "score loss, test mae, test mse]:", performance_score)
        test_val_scores[VAL_PREFIX + annual_daily_label + actual_label] = performance_score

        costing_test_actual += y_test[actual_label]
        costing_test_predicted += y_test[predicted_label + TRANSFORMATION_SUFFIX]

    costing_rmse = metrics.mean_squared_error(costing_test_actual, costing_test_predicted, squared=False)

    # Get the energy annual predictions
    # TODO: Up until final triple quote, fit within new metric trackiign and refactor
    if process_type.lower() == config.Settings().APP_CONFIG.ENERGY:
        y_test = convert_dataframe_to_annual(y_test)
        y_validate = convert_dataframe_to_annual(y_validate)
    else:
        # Drop the duplicate datapoint_id entry
        y_test_complete = y_test_complete.drop(['datapoint_id'], axis=1)
        y_validate_complete = y_validate_complete.drop(['datapoint_id'], axis=1)  

    # Retrieve the annual energy predictions
    if process_type.lower() == config.Settings().APP_CONFIG.ENERGY:
        for actual_label, predicted_label in zip(actual_set, prediction_set):
            # Retrieve the score and store it with the appropriate title for both the test and validation sets
            performance_score = score(y_test[actual_label], y_test[predicted_label + TRANSFORMATION_SUFFIX])
            print("[" + actual_label, TEST_PREFIX + "score loss, test mae, test mse]:", performance_score)
            test_val_scores[TEST_PREFIX + ANNUAL_PREFIX + actual_label] = performance_score

            performance_score = score(y_validate[actual_label], y_validate[predicted_label + TRANSFORMATION_SUFFIX])
            print("[" + actual_label, VAL_PREFIX + "score loss, test mae, test mse]:", performance_score)
            test_val_scores[VAL_PREFIX + ANNUAL_PREFIX + actual_label] = performance_score

            energy_test_actual += y_test[actual_label]
            energy_test_predicted += y_test[predicted_label + TRANSFORMATION_SUFFIX]

        energy_rmse = metrics.mean_squared_error(energy_test_actual, energy_test_predicted, squared=False)

    # Maintain original implemented notation by using two new variable names
    output_df = y_test
    output_val_df = y_validate

    # Load the original building data to allow for more detailed results to be returned
    train_building_data, _, _ = preprocessing.process_building_files(path_elec, path_gas, False)
    print(output_df.columns)
    output_df = pd.merge(output_df, train_building_data, left_on="datapoint_id", right_on= ":datapoint_id", how="left")
    # Generate the average predictions for both the building types and the climate zones
    # 1. Merge back the datapoint ids with the building data (test and validation)
    if val_building_path:
        val_building_data, _, _ = preprocessing.process_building_files(val_building_path, "", False)
        output_val_df = pd.merge(output_val_df, val_building_data, left_on="datapoint_id", right_on= ":datapoint_id", how="left")
    else:
        output_val_df = pd.merge(output_val_df, train_building_data, left_on="datapoint_id",  right_on= ":datapoint_id", how="left")

    # Remove the initial predictions from the outputs, keeping only the transformed values
    output_df = output_df.drop(prediction_set, axis=1, errors='ignore')
    output_val_df = output_val_df.drop(prediction_set, axis=1, errors='ignore')

    # Track all building and weather errors within a dictionary
    output_label = '(mean actual energy, mean predicted energy, MAE, MSE)'
    if process_type.lower() == config.Settings().APP_CONFIG.COSTING:
        output_label = '(mean actual costing, mean predicted costing, MAE, MSE)'
    building_weather_errors = {}
    for actual_label, predicted_label in zip(actual_set, prediction_set):
        # Obtain the error values and update the dataframes (for both the test and validation sets)
        output_df, building_errors, climate_errors = compute_building_weather_errors(output_df, actual_label, predicted_label + TRANSFORMATION_SUFFIX)
        building_weather_errors[TEST_PREFIX + BUILDING_ERRORS_LABEL + " (" + actual_label + ") " + output_label] = building_errors.values.tolist()
        building_weather_errors[TEST_PREFIX + CLIMATE_ERRORS_LABEL + " (" + actual_label + ") " + output_label] = climate_errors.values.tolist()

        output_val_df, building_errors, climate_errors = compute_building_weather_errors(output_val_df, actual_label, predicted_label + TRANSFORMATION_SUFFIX)
        building_weather_errors[VAL_PREFIX + BUILDING_ERRORS_LABEL + " (" + actual_label + ") " + output_label] = building_errors.values.tolist()
        building_weather_errors[VAL_PREFIX + CLIMATE_ERRORS_LABEL + " (" + actual_label + ") " + output_label] = climate_errors.values.tolist()

    print('****************TEST SET****************************')
    print(output_df.head(50))
    #print(annual_metric)

    print('****************VALIDATION SET****************************')
    print(output_val_df.head(50))
    #print(annual_metric_val)

    results = {
            **test_val_scores,
            'output_df': output_df.values.tolist(),
            'output_val_df': output_val_df.values.tolist(),
            **building_weather_errors
            }
    
    return results, energy_rmse, costing_rmse


def create_model_mlp(idx, dense_layers, activation, optimizer, dropout_rate, length, learning_rate, epochs,
                     batch_size, scaling_performance, X_train, y_train, X_test, y_test, y_test_complete,
                     scalery, X_validate, y_validate, y_validate_complete, output_path, path_elec, path_gas,
                     val_building_path, process_type, output_nodes):
    """
    Creates a MLP model with defaulted values without need to perform an hyperparameter search at all times.
    Its initutive to have run the hyperparameter search beforehand to know the hyperparameter value to set.

    Args:
        dense_layers: number of layers for the model architecture e.g for a model with 3 layers, values will be passed as [8,20,30]
        activation: activation function to be used e.g relu, tanh
        optimizer: optimizer to be used in compiling the model e.g relu, rmsprop, adam
        dropout_rate: used to make the model avoid overfitting, value should be less than 1 e.g 0.3
        length: length of the trainset
        learning_rate: learning rate determines how fast or how slow the model will converge to an optimal loss value. Value should be less or equal 0.1 e.g 0.001
        epochs: number of iterations the model should perform
        batch_size: batch size to be used
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
        output_nodes: The number of outputs which the model needs to predict.
    Returns:
        metric: evaluation results containing the loss value from the testset prediction,
        annual_metric: predicted value for each datapooint_id is summed to calculate the annual energy consumed and the loss value from the testset prediction,
        output_df: merge of y_pred, y_test, datapoint_id, the final dataframe showing the model output using the testset
        val_metric:evaluation results containing the loss value from the validationset prediction,
        val_annual_metric:predicted value for each datapooint_id is summed to calculate the annual energy consumed and the loss value from the validationset prediction,,
        output_val_df: merge of y_pred, y_validate, datapoint_id, the final dataframe showing the model output using the validation set
    """
    parameter_search_path = str(Path(output_path).joinpath("parameter_search"))
    btap_log_path = str(Path(parameter_search_path).joinpath("btap"))
    # Create the output directories if they do not exist
    config.create_directory(parameter_search_path)
    config.create_directory(btap_log_path)

    model = Sequential()
    model.add(Flatten())
    # model.add(Dropout(dropout_rate, input_shape=(length,)))
    for index, lsize in enumerate(dense_layers):
        model.add(Dense(lsize, input_shape=(length,),
                        activation=activation, kernel_initializer='normal',
                        # kernel_regularizer=regularizers.l1(1e-5),
                        ))
        if dropout_rate > 0 and dropout_rate <= 1:
            model.add(Dropout(dropout_rate))
    model.add(Dense(output_nodes, activation='linear'))

    if optimizer == "adam":
        optimizer = tf.optimizers.Adam(learning_rate=learning_rate)
    elif optimizer == "sgd":
        optimizer = tf.optimizers.SGD(learning_rate=learning_rate)
    else:
        optimizer = tf.optimizers.RMSprop(learning_rate=learning_rate)

    # Comiple the mode with the optimizer and learninf rate specified in hparams
    model.compile(optimizer=optimizer,
                  loss=rmse_loss,
                  metrics=['mae', 'mse', 'mape', det_coeff])
    # Define callback
    early_stopping = EarlyStopping(monitor='val_loss', patience=10)
    logdir = os.path.join(btap_log_path, datetime.datetime.now().strftime("%Y%m%d-%H%M%S"))
    hist_callback = tf.keras.callbacks.TensorBoard(logdir, histogram_freq=1)
    logger = keras.callbacks.CSVLogger(output_path + '/metric.csv', append=True)
    output_df = ''

    # See how the model performs based on the training dataset size.
    isMLP = True
    if scaling_performance == True:
        test_scaling(model, isMLP, epochs, batch_size, X_train, y_train, X_test, y_test, scalery,
                     X_validate, y_validate, y_test_complete, y_validate_complete,
                     path_elec, path_gas, val_building_path, process_type)

    # prepare the model with target scaling
    scores_metric = ''
    history = model.fit(X_train,
                        y_train,
                        callbacks=[
                                   logger,
                                   hist_callback,
                                   ],
                        epochs=epochs,
                        batch_size=batch_size,
                        verbose=1,
                        #shuffle=False,
                        validation_split=0.10)
    
    pl.mlp_learning_curve_plot(idx, process_type, history)
    print(model.summary())

    return model

def create_model_rf(idx, n_estimators, max_depth, min_samples_split, min_samples_leaf, scaling_performance,
                    X_train, y_train, X_test, y_test, y_test_complete, scalery, X_validate, y_validate,
                    y_validate_complete, output_path, path_elec, path_gas, val_building_path, process_type):
    """
    Creates a model with defaulted values without need to perform an hyperparameter search at all times.
    Its initutive to have run the hyperparameter search beforehand to know the hyperparameter value to set.

    Args:
        n_estimators: the number of trees in the random forest
        max_depth: the maximum depth of the tree.
        min_samples_split: The minimum number of samples required to split an internal node:
        min_samples_leaf: The minimum number of samples required to be at a leaf node.
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
        output_nodes: The number of outputs which the model needs to predict.
    Returns:
        metric: evaluation results containing the loss value from the testset prediction,
        annual_metric: predicted value for each datapooint_id is summed to calculate the annual energy consumed and the loss value from the testset prediction,
        output_df: merge of y_pred, y_test, datapoint_id, the final dataframe showing the model output using the testset
        val_metric:evaluation results containing the loss value from the validationset prediction,
        val_annual_metric:predicted value for each datapooint_id is summed to calculate the annual energy consumed and the loss value from the validationset prediction,,
        output_val_df: merge of y_pred, y_validate, datapoint_id, the final dataframe showing the model output using the validation set
    """
    parameter_search_path = str(Path(output_path).joinpath("parameter_search"))
    btap_log_path = str(Path(parameter_search_path).joinpath("btap"))

    # Create the output directories if they do not exist
    config.create_directory(parameter_search_path)
    config.create_directory(btap_log_path)

    model = RandomForestRegressor(n_estimators=n_estimators,
                                  max_depth=max_depth,
                                  min_samples_split=min_samples_split,
                                  min_samples_leaf=min_samples_leaf,
                                  random_state=42,
                                  n_jobs = -1,
                                  verbose = 1)
    
    # See how the model performs based on the training dataset size.
    if scaling_performance == True:
        test_scaling(model, X_train, y_train, X_test, y_test, scalery,
                     X_validate, y_validate, y_test_complete, y_validate_complete,
                     path_elec, path_gas, val_building_path, process_type)
    
    # Train the model
    model.fit(X_train, y_train)

    return model
        
def create_model_gradient_boosting(idx, n_estimators, max_depth, learning_rate, subsample, scaling_performance,
                                   X_train, y_train, X_test, y_test, y_test_complete, scalery, X_validate, y_validate, y_validate_complete, modified_y_validate,
                                   output_path, path_elec, path_gas, val_building_path, process_type):
    """
    Creates a XGBoost model and graphs the learning curve.

    Args:
        n_estimators: the number of trees in the random forest
        max_depth: the maximum depth of the tree.
        min_samples_split: The minimum number of samples required to split an internal node:
        min_samples_leaf: The minimum number of samples required to be at a leaf node.
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
        output_nodes: The number of outputs which the model needs to predict.
    Returns:
        XGBoost model.

    Note:
        There's an option to test how the model performs based on the training dataset size.
        To see how this performs you need to set "scaling_performance" to True.
    """
    # Create the output directories if they do not exist
    parameter_search_path = str(Path(output_path).joinpath("parameter_search"))
    btap_log_path = str(Path(parameter_search_path).joinpath("btap"))
    
    config.create_directory(parameter_search_path)
    config.create_directory(btap_log_path)
    
    model = XGBRegressor(n_estimators=n_estimators,
                         max_depth=max_depth,
                         learning_rate=learning_rate,
                         subsample=subsample,
                         random_state=42,
                         n_jobs=-1)
    
    model.fit(X_train,
              y_train,
              eval_set=[(X_train, y_train),
                        (X_validate, modified_y_validate)],
              verbose=True)

    # Visualize performance of the machine learning models
    model_results = model.evals_result()

    train_rmse = model_results["validation_0"]["rmse"]
    val_rmse = model_results["validation_1"]["rmse"]

    pl.xgboost_learning_curve_plot(idx, train_rmse, val_rmse)

    # See how the model performs based on the training dataset size.
    if scaling_performance == True:
        test_scaling(model, X_train, y_train, X_test, y_test, scalery,
                     X_validate, y_validate, y_test_complete, y_validate_complete,
                     path_elec, path_gas, val_building_path, process_type)

    return model
    
def test_scaling(model, isMLP, epochs, batch_size, X_train, y_train, X_test, y_test, scalery, X_validate, y_validate, y_test_complete, y_validate_complete, path_elec, path_gas, val_building_path, process_type):
    time_array = []
    energy_result_array = []
    costing_result_array = []

    X_train_full = X_train.copy()
    y_train_full = y_train.copy()

    print(X_train_full.shape)
    print(y_train_full.shape)

    training_size = np.array([500, 1000, 2000, 3000, 4000])

    # Energy dataset is based on daily data and not yearly data. Costing dataset is based on yearly data.
    if process_type == 'energy':
        training_size = training_size * 365

    print(training_size)

    for index, amount in enumerate(training_size): 
        idx = np.random.choice(X_train.shape[0], size=amount, replace=False)

        X_train_subset = X_train_full[idx, :]
        y_train_subset = y_train_full[idx, :]
    
        current_time = time.time()

        if isMLP == True:
            print(batch_size)
            model.fit(X_train_subset,
                      y_train_subset,
                      epochs=epochs,
                      batch_size=batch_size,
                      verbose=1,
                      #shuffle=False,
                      validation_split=0.10)
        else:
            model.fit(X_train_subset, y_train_subset)

        time_taken = ((time.time() - current_time) / 60)
        time_array.append(time_taken)

        result, energy_rmse, costing_rmse = evaluate(model, X_test, y_test, scalery, X_validate, y_validate, y_test_complete, y_validate_complete, path_elec, path_gas, val_building_path, process_type)
        
        energy_result_array.append(energy_rmse)
        costing_result_array.append(costing_rmse)
    
    print(energy_result_array)
    print(costing_result_array)
    print(time_array)

    if process_type.lower() == config.Settings().APP_CONFIG.ENERGY:
        global fig
        global ax
        
        fig, ax = plt.subplots(2, 2, figsize=(10,5))
        
        ax[0, 0].set_title('Energy')

        ax[0, 0].plot(training_size/365, energy_result_array, color='darkcyan')
        ax[1, 0].plot(training_size/365, time_array, color='yellowgreen')

        ax[0, 0].set_ylabel("RMSE", color='black')
        ax[1, 0].set_ylabel("Time (minutes)", color='black')

    else:
        ax[0, 1].set_title('Costing')

        ax[0, 1].plot(training_size, costing_result_array, color='darkcyan')
        ax[1, 1].plot(training_size, time_array, color='yellowgreen')

        ax[1, 0].set_xlabel("Training size", color='black')
        ax[1, 1].set_xlabel("Training size", color='black')

    plt.savefig('./output/training_size.png')

def fit_evaluate(idx, preprocessed_data_file, selected_features_file, selected_model_type, param_search, output_path, random_seed, path_elec, path_gas, val_building_path, process_type, use_updated_model, use_dropout):
    """
    Loads the output from preprocessing and feature selection, builds the model, then evaluates the model.

    Args:
        preprocessed_data_file: Location and name of a .json preprocessing file to be used.
        selected_features_file: Location and name of a .json feature selection file to be used.
        selected_model_type: the type of model to be used. Can either be 'mlp', 'rf', or 'gb'
        param_search: 'yes' if hyperparameter tuning should be performed (increases runtime), 'no' if the default hyperparameters should be used.
        output_path: Where output data should be placed. Note that this value should be empty unless this file is called from a pipeline.
        random_seed: Random seed to be used when training. Should not be -1 when used through the CLI.
        path_elec: Filepath of the electricity building file which has been used.
        path_gas: Filepath of the gas building file, if it has been used (pass nothing otherwise).
        val_building_path: Filepath of the validation building file, if it has been used (pass nothing otherwise).
        process_type: Either 'energy' or 'costing' to specify the operations to be performed.
        use_updated_model: True if the larger model architecture should be used for training. Should be False if a costing model is being trained.
        use_dropout: True if the regularization technique should be used (on by default). False if tests are desired without dropout. Note that not using dropout may cause bias to learned when training.
    Returns:
        the results from the model prediction is uploaded to minio
    """

    # Resets all state generated by Keras.
    K.clear_session()
    start_time = time.time()
    # Set the random seeds
    np.random.seed(random_seed)
    tf.random.set_seed(random_seed)
    os.environ['PYTHONHASHSEED'] = str(random_seed)
    # Load the training, testing, and validation sets
    with open(preprocessed_data_file, 'r', encoding='UTF-8') as preprocessing_file:
        preprocessing_json = json.load(preprocessing_file)
    # Load the set of features to be used for training
    with open(selected_features_file, 'r', encoding='UTF-8') as feature_selection_file:
        features_json = json.load(feature_selection_file)
    # Configure the dataframes using the features specified from the preprocessing
    features = preprocessing_json["features"]
    selected_features = features_json["features"]
    X_train = pd.DataFrame(preprocessing_json["X_train"], columns=features)
    X_test = pd.DataFrame(preprocessing_json["X_test"], columns=features)
    X_validate = pd.DataFrame(preprocessing_json["X_validate"], columns=features)
    y_train = pd.read_json(preprocessing_json["y_train"], orient='values').values#.ravel()
    # Remove the total from the loaded json
    y_train = y_train[:, 1:]

    # Extract the selected features from feature engineering
    X_train = X_train[selected_features]
    X_test = X_test[selected_features]
    X_validate = X_validate[selected_features]
    col_length = X_train.shape[1]

    json_columns = ['energy', 'datapoint_id', 'Total Energy', 'energy_elec', 'energy_gas']
    if process_type.lower() == config.Settings().APP_CONFIG.COSTING:
        json_columns = ['cost_equipment_total_cost_per_m_sq',
                        'cost_equipment_envelope_total_cost_per_m_sq',
                        'cost_equipment_heating_and_cooling_total_cost_per_m_sq',
                        'cost_equipment_lighting_total_cost_per_m_sq',
                        'cost_equipment_ventilation_total_cost_per_m_sq',
                        'cost_equipment_renewables_total_cost_per_m_sq',
                        'cost_equipment_shw_total_cost_per_m_sq',
                        'datapoint_id']
    # Extract the test and valuidation data for the target variable
    y_test_complete = pd.DataFrame(preprocessing_json["y_test_complete"], columns=json_columns)
    y_validate_complete = pd.DataFrame(preprocessing_json["y_validate_complete"], columns=json_columns)
    # Remove "Total Energy" from json_columns if the energy predictions are being performed
    output_nodes = len(y_train[0])
    print("Output nodes: " + str(output_nodes))
    print("Input shape: " + str(col_length))
    if process_type.lower() == config.Settings().APP_CONFIG.ENERGY:
        #json_columns = json_columns[:-1]
        json_columns.remove('Total Energy')
    y_test = pd.DataFrame(preprocessing_json["y_test"], columns=json_columns)
    y_validate= pd.DataFrame(preprocessing_json["y_validate"], columns=json_columns)

    # Scale the data to be used for training and testing
    scalerx = StandardScaler()
    scalery = StandardScaler()

    X_train = scalerx.fit_transform(X_train)
    X_test = scalerx.transform(X_test)
    X_validate = scalerx.transform(X_validate)
    y_train = scalery.fit_transform(y_train)
    
    # Define the path where the output files should be placed
    model_path = Path(output_path).joinpath(config.Settings().APP_CONFIG.TRAINING_BUCKET_NAME)
    config.create_directory(str(model_path))

    # If set to "yes", search for best hyperparameters before training
    if param_search.lower() == "yes":
        if selected_model_type.lower() == 'mlp':
            model = tune_mlp(X_train,
                             y_train,
                             X_test,
                             y_test,
                             col_length,
                             output_nodes,
                             features,
                             output_path,
                             random_seed,
                             process_type)

            model_output_path = str(model_path.joinpath(config.Settings().APP_CONFIG.TRAINED_MODEL_FILENAME_MLP))
            model.save(model_output_path) 

        elif selected_model_type.lower() == 'rf':
            model = tune_rf(X_train, y_train, output_path)

            model_output_path = str(model_path.joinpath(config.Settings().APP_CONFIG.TRAINED_MODEL_FILENAME_RF))
            joblib.dump(model, model_output_path)

        else:
            model = tune_gradient_boosting(X_train, y_train, output_path)
            
            model_output_path = str(model_path.joinpath(config.Settings().APP_CONFIG.TRAINED_MODEL_FILENAME_GB))
            joblib.dump(model, model_output_path)
        
        results_pred, _, _ = evaluate(model, X_test, y_test, scalery, X_validate, y_validate, y_test_complete, y_validate_complete,
                                      path_elec, path_gas, val_building_path, process_type)
    else:
        SCALING_PERFORMANCE = False # Try out different training dataset size and see how they perform.

        if selected_model_type.lower() == 'mlp':
            if process_type == 'energy':
                DROPOUT_RATE = 0.0
                LEARNING_RATE = 0.001
                EPOCHS = 100
                BATCH_SIZE = 1024
                NUMBER_OF_NODES = [856, 3744]
                ACTIVATION = 'relu'
                OPTIMIZER = 'adam'
            else:
                DROPOUT_RATE = 0.0
                LEARNING_RATE = 0.001
                EPOCHS = 100
                BATCH_SIZE = 32
                NUMBER_OF_NODES = [2744, 248]
                ACTIVATION = 'relu'
                OPTIMIZER = 'adam'

            if not use_updated_model:
                LEARNING_RATE = 0.001
                NUMBER_OF_NODES = 56
            if not use_dropout:
                DROPOUT_RATE = -1

            model = create_model_mlp(idx=idx,
                                     dense_layers=NUMBER_OF_NODES,
                                     activation=ACTIVATION,
                                     optimizer=OPTIMIZER,
                                     dropout_rate=DROPOUT_RATE,
                                     length=col_length,
                                     learning_rate=LEARNING_RATE,
                                     epochs=EPOCHS,
                                     batch_size=BATCH_SIZE,
                                     scaling_performance=SCALING_PERFORMANCE,
                                     X_train=X_train,
                                     y_train=y_train,
                                     X_test=X_test,
                                     y_test=y_test,
                                     y_test_complete=y_test_complete,scalery=scalery,
                                     X_validate = X_validate,
                                     y_validate=y_validate,
                                     y_validate_complete= y_validate_complete,
                                     output_path=output_path,
                                     path_elec=path_elec,
                                     path_gas=path_gas,
                                     val_building_path=val_building_path,
                                     process_type=process_type,
                                     output_nodes=output_nodes)
            
            model_output_path = str(model_path.joinpath(config.Settings().APP_CONFIG.TRAINED_MODEL_FILENAME_MLP))
            model.save(model_output_path)

        elif selected_model_type.lower() == 'rf':
            if process_type == 'energy':
                N_ESTIMATORS = 100
                MAX_DEPTH = None
                MIN_SAMPLES_SPLIT = 2
                MIN_SAMPLES_LEAF = 1
            else:
                N_ESTIMATORS = 100
                MAX_DEPTH = 15
                MIN_SAMPLES_SPLIT = 2
                MIN_SAMPLES_LEAF = 1

            model = create_model_rf(idx=idx,
                                    n_estimators = N_ESTIMATORS,
                                    max_depth=MAX_DEPTH,
                                    min_samples_split=MIN_SAMPLES_SPLIT,
                                    min_samples_leaf=MIN_SAMPLES_LEAF,
                                    scaling_performance=SCALING_PERFORMANCE,
                                    X_train=X_train,
                                    y_train=y_train,
                                    X_test=X_test,
                                    y_test=y_test,
                                    y_test_complete=y_test_complete,
                                    scalery=scalery,
                                    X_validate = X_validate,
                                    y_validate=y_validate,
                                    y_validate_complete= y_validate_complete,
                                    output_path=output_path,
                                    path_elec=path_elec,
                                    path_gas=path_gas,
                                    val_building_path=val_building_path,
                                    process_type=process_type)
            
            model_output_path = str(model_path.joinpath(config.Settings().APP_CONFIG.TRAINED_MODEL_FILENAME_RF))
            joblib.dump(model, model_output_path)
        
        else:
            if process_type == 'energy':
                N_ESTIMATORS = 500
                MAX_DEPTH = 10
                LEARNING_RATE = 0.05
                SUBSAMPLE = 0.5
            else:
                N_ESTIMATORS = 500
                MAX_DEPTH = 5
                LEARNING_RATE = 0.05
                SUBSAMPLE = 0.5

            if process_type.lower() == config.Settings().APP_CONFIG.COSTING:
                modified_y_validate = y_validate.iloc[:, 1:-1]
                modified_y_validate = scalery.transform(modified_y_validate)
            else:
                modified_y_validate = y_validate.iloc[:, 2:]
                modified_y_validate = scalery.transform(modified_y_validate)

            model = create_model_gradient_boosting(idx=idx,
                                                   n_estimators = N_ESTIMATORS,
                                                   max_depth=MAX_DEPTH,
                                                   learning_rate=LEARNING_RATE,
                                                   subsample=SUBSAMPLE,
                                                   scaling_performance=SCALING_PERFORMANCE,
                                                   X_train=X_train,
                                                   y_train=y_train,
                                                   X_test=X_test,
                                                   y_test=y_test,
                                                   y_test_complete=y_test_complete,
                                                   scalery=scalery,
                                                   X_validate = X_validate,
                                                   y_validate=y_validate,
                                                   y_validate_complete= y_validate_complete,
                                                   modified_y_validate=modified_y_validate,
                                                   output_path=output_path,
                                                   path_elec=path_elec,path_gas=path_gas,
                                                   val_building_path=val_building_path,
                                                   process_type=process_type)
            
            model_output_path = str(model_path.joinpath(config.Settings().APP_CONFIG.TRAINED_MODEL_FILENAME_GB))
            joblib.dump(model, model_output_path)
            
        results_pred, _, _ = evaluate(model, X_test, y_test, scalery, X_validate, y_validate, y_test_complete, y_validate_complete,
                                path_elec, path_gas, val_building_path, process_type)

    # Calculate the time spent training in minutes
    time_taken = ((time.time() - start_time) / 60)
    print("********* Total time spent is " + str(time_taken) + " minutes ***********" )

    # Write processing time in processing_time.txt file
    processing_time_file_path = str(model_path) + "/" + 'processing_time.txt'
    processing_time_file = open(processing_time_file_path, 'w')
    processing_time_file.write(str(time_taken))

    # Define the output files
    output_filename_json = str(model_path) + "/" + config.Settings().APP_CONFIG.TRAINING_RESULTS_FILENAME + ".json"
    output_filename_csv = str(model_path) + "/" + config.Settings().APP_CONFIG.TRAINING_RESULTS_FILENAME + ".csv"

    # Specify the indices where the target outputs are
    # The order is [acutal_0, predicted_0, actual_1, predicted_1, ..., actual_n, predicted_n]
    output_indices = [1, 3, 2, 4]
    output_label = config.Settings().APP_CONFIG.ENERGY
    if process_type.lower() == config.Settings().APP_CONFIG.COSTING:
        output_indices = [1, 8, 2, 9, 3, 10, 4, 11, 5, 12, 6, 13]
        output_label = config.Settings().APP_CONFIG.COSTING
    # Output the results within a csv for each prediction
    # TODO: Update to work with costing and energy for any amount of output_indices
    with open(output_filename_csv, 'a', encoding='utf-8') as csv_output:
        writer = csv.writer(csv_output)
        if process_type.lower() == config.Settings().APP_CONFIG.COSTING:
            writer.writerow(['ID',
                             'Actual cost_equipment_envelope_total_cost_per_m_sq ', 'Predicted cost_equipment_envelope_total_cost_per_m_sq ',
                             'Actual cost_equipment_heating_and_cooling_total_cost_per_m_sq ', 'Predicted cost_equipment_heating_and_cooling_total_cost_per_m_sq ',
                             'Actual cost_equipment_lighting_total_cost_per_m_sq ', 'Predicted cost_equipment_lighting_total_cost_per_m_sq ',
                             'Actual cost_equipment_ventilation_total_cost_per_m_sq ', 'Predicted cost_equipment_ventilation_total_cost_per_m_sq ',
                             'Actual cost_equipment_renewables_total_cost_per_m_sq ', 'Predicted cost_equipment_renewables_total_cost_per_m_sq ',
                             'Actual cost_equipment_shw_total_cost_per_m_sq ', 'Predicted cost_equipment_shw_total_cost_per_m_sq '
                             ])
            for i, pair in enumerate(results_pred['output_df']):
                writer.writerow(['Test_' + str(i),
                                 pair[output_indices[0]], pair[output_indices[1]],
                                 pair[output_indices[2]], pair[output_indices[3]],
                                 pair[output_indices[4]], pair[output_indices[5]],
                                 pair[output_indices[6]], pair[output_indices[7]],
                                 pair[output_indices[8]], pair[output_indices[9]],
                                 pair[output_indices[10]], pair[output_indices[11]]
                                 ])
            for i, pair in enumerate(results_pred['output_val_df']):
                writer.writerow(['Validation_' + str(i),
                                 pair[output_indices[0]], pair[output_indices[1]],
                                 pair[output_indices[2]], pair[output_indices[3]],
                                 pair[output_indices[4]], pair[output_indices[5]],
                                 pair[output_indices[6]], pair[output_indices[7]],
                                 pair[output_indices[8]], pair[output_indices[9]],
                                 pair[output_indices[10]], pair[output_indices[11]]
                                 ])
        else:
            writer.writerow(['ID', 'Actual Electricity ' + output_label, 'Predicted Electricity ' + output_label, 'Actual Gas ' + output_label, 'Predicted Gas ' + output_label])
            for i, pair in enumerate(results_pred['output_df']):
                writer.writerow(['Test_' + str(i), pair[output_indices[0]], pair[output_indices[1]], pair[output_indices[2]], pair[output_indices[3]]])
            for i, pair in enumerate(results_pred['output_val_df']):
                writer.writerow(['Validation_' + str(i), pair[output_indices[0]], pair[output_indices[1]], pair[output_indices[2]], pair[output_indices[3]]])
    # Also output all training information within one json file
    with open(output_filename_json, 'w', encoding='utf8') as json_output:
        json.dump(results_pred, json_output)

    # Output the scalers used to scale the X and y data
    joblib.dump(scalerx, str(model_path.joinpath(config.Settings().APP_CONFIG.SCALERX_FILENAME)))
    joblib.dump(scalery, str(model_path.joinpath(config.Settings().APP_CONFIG.SCALERY_FILENAME)))
    # Returns the model output filepath and the results output filepath
    return model_output_path, output_filename_csv

def main(config_file: str = typer.Argument(..., help="Location of the .yml config file (default name is input_config.yml)."),
         process_type: str = typer.Argument(..., help="Either 'energy' or 'costing' to specify the operations to be performed."),
         preprocessed_data_file: str = typer.Argument(..., help="Location and name of a .json preprocessing file to be used."),
         selected_features_file: str = typer.Argument(..., help="Location and name of a .json feature selection file to be used."),
         selected_model_type: str = typer.Option("mlp", help="Type of model selected. can either be 'mlp' or 'rf'"),
         perform_param_search: str = typer.Option("no", help="'yes' if hyperparameter tuning should be performed (increases runtime), 'no' if the default hyperparameters should be used."),
         output_path: str = typer.Option("", help="The output path to be used. Note that this value should be empty unless this file is called from a pipeline."),
         random_seed: int = typer.Option(-1, help="Random seed to be used when training. Should not be -1 when used through the CLI."),
         path_elec: str = typer.Argument(..., help="Filepath of the electricity building file which has been used."),
         path_gas: str = typer.Option("", help="Filepath of the gas building file, if it has been used (pass nothing otherwise)."),
         val_building_path: str = typer.Option("", help="Filepath of the validation building file, if it has been used (pass nothing otherwise)."),
         use_updated_model: bool = typer.Option(True, help="True if the larger model architecture should be used for training. Should be False if a costing model is being trained."),
         use_dropout: bool = typer.Option(True, help="True if the regularization technique should be used (on by default). False if tests are desired without dropout. Note that not using dropout may cause bias to learned when training."),
         idx: int = typer.Option(0, help="Keeps track of the training process type")
         ):
    """
    Using all preprocessed data, build and train a Machine Learning model to predict the total energy or costing values.
    All steps of this process are saved, and the model is evaluated to determine its effectiveness overall and on
    specific building types and climate zones.

    Args:
        config_file: Location of the .yml config file (default name is input_config.yml).
        process_type: Either 'energy' or 'costing' to specify the operations to be performed.
        preprocessed_data_file: Location and name of a .json preprocessing file to be used.
        selected_features_file: Location and name of a .json feature selection file to be used.
        selected_model_type: Type of model selected. can either be 'mlp' for Multilayer Perceptron or 'rf' for Random Forest
        perform_param_search: 'yes' if hyperparameter tuning should be performed (increases runtime), 'no' if the default hyperparameters should be used.
        output_path: Where output data should be placed. Note that this value should be empty unless this file is called from a pipeline.
        random_seed: Random seed to be used when training. Should not be -1 when used through the CLI.
        path_elec: Filepath of the electricity building file which has been used.
        path_gas: Filepath of the gas building file, if it has been used (pass nothing otherwise).
        val_building_path: Filepath of the validation building file, if it has been used (pass nothing otherwise).
        use_updated_model: True if the larger model architecture should be used for training. Should be False if a costing model is being trained.
        use_dropout: True if the regularization technique should be used (on by default). False if tests are desired without dropout. Note that not using dropout may cause bias to learned when training.
        idx: Keeps track of the training process type.
    """
    DOCKER_INPUT_PATH = config.Settings().APP_CONFIG.DOCKER_INPUT_PATH

    # Load all content stored from the config file, if provided
    if len(config_file) > 0:
        # Load the specified config file
        cfg = config.get_config(DOCKER_INPUT_PATH + config_file)
        if random_seed < 0: random_seed = cfg.get(config.Settings().APP_CONFIG.RANDOM_SEED)
        if perform_param_search == "": perform_param_search = cfg.get(config.Settings().APP_CONFIG.PARAM_SEARCH)

    # Validate all input files
    input_model = PredictModel(input_prefix=DOCKER_INPUT_PATH,
                               preprocessed_data_file=preprocessed_data_file,
                               selected_features_file=selected_features_file,
                               selected_model_type=selected_model_type,
                               perform_param_search=perform_param_search,
                               random_seed=random_seed,
                               building_param_files=[path_elec, path_gas],
                               val_building_params_file=val_building_path
                              )
    # If the output path is blank, map to the docker output path
    if len(output_path) < 1:
        output_path = config.Settings().APP_CONFIG.DOCKER_OUTPUT_PATH
        
    return fit_evaluate(idx, input_model.preprocessed_data_file, input_model.selected_features_file, input_model.selected_model_type, input_model.perform_param_search, output_path, input_model.random_seed, input_model.building_param_files[0], input_model.building_param_files[1], input_model.val_building_params_file, process_type, use_updated_model,
                        use_dropout)

if __name__ == '__main__':
    # Load settings from the environment
    settings = config.Settings()
    # Run the CLI interface
    typer.run(main)
