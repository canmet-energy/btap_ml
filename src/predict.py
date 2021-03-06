"""
Uses the output from preprocessing and feature selection from mino, builds the model and then evaluate the model.

CLI arguments match those defined by ``main()``.
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
from keras import regularizers  # for l2 regularization
from keras.callbacks import EarlyStopping
from keras.layers.core import Dense, Dropout, Flatten
from keras.models import Sequential
from keras_tuner import Hyperband
from matplotlib import pyplot as plt
from sklearn import metrics
from sklearn.preprocessing import RobustScaler
from tensorboard.plugins.hparams import api as hp
from tensorflow.keras import layers

import config
import plot as pl
from models.predict_model import PredictModel

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


def rmse_loss(y_true, y_pred):
    """
    A customized rmse score that takes a sum of y_pred and y_test before computing the rmse score
    Args:
        y_test: y testset
        y_pred: y predicted value from the model
    Returns:
       rmse loss from comparing the y_test and y_pred values
    """
    sum_pred = K.sum(y_pred, axis=-1)
    sum_true = K.sum(y_true, axis=-1)
    loss = K.sqrt(K.mean(K.square(sum_pred - sum_true)))

    return loss


def model_builder(hp):
    """
    Builds the model that would be used to search for hyperparameter.
    The hyperparameters search inclues activation, regularizers, dropout_rate, learning_rate, and optimizer
    Args:
        hp: hyperband object with different hyperparameters to be checked.
    Returns:
       model will be built based on the different hyperparameter combinations.
    """
    model = keras.Sequential()
    model.add(keras.layers.Flatten())

    hp_activation= hp.Choice('activation', values=['relu','tanh','sigmoid'])
    for i in range(hp.Int("num_layers", 1,1)):
        model.add(layers.Dense(
            units=hp.Int("units_" + str(i), min_value=8, max_value=96, step=8),
            activation=hp_activation,
            input_shape=(36, ),
            kernel_initializer='normal',
            kernel_regularizer=regularizers.l1_l2(l1=1e-5, l2=1e-5),
            bias_regularizer=regularizers.l2(1e-4),
            activity_regularizer=regularizers.l2(1e-5)
            ))
        model.add(Dropout(hp.Choice('dropout_rate', values=[0.1, 0.2, 0.3])))
    model.add(Dense(1, activation='linear'))

    hp_learning_rate = hp.Choice('learning_rate', values=[1e-3, 1e-4, 1e-5])
    hp_optimizer = hp.Choice('optimizer', values=['rmsprop', 'adam', 'sgd'])

    if hp_optimizer == "adam":
        optimizer = tf.optimizers.Adam(learning_rate=hp_learning_rate)
    elif hp_optimizer == "sgd":
        optimizer = tf.optimizers.SGD(learning_rate=hp_learning_rate)
    else:
        optimizer = tf.optimizers.RMSprop(learning_rate=hp_learning_rate)

    # Comiple the mode with the optimizer and learninf rate specified in hparams
    model.compile(optimizer=optimizer,
                  loss=rmse_loss,
                  metrics=['mae', 'mse','mape'])

    return model


def predicts_hp(X_train, y_train, X_test, y_test, selected_feature, output_path, random_seed):
    """
    Using the set of hyperparameter combined,the model built is used to make predictions
    Args:
        X_train: X train set
        y_train: y train set
        X_test: X test set
        y_test: y test set
        selected_feature: selected features that would be used to build the model
        output_path: Where the output files should be placed.
        random_seed: The random seed to be used
    Returns:
       Model built from the set of hyperparameters combined.
    """
    parameter_search_path = str(Path(output_path).joinpath("parameter_search"))
    log_path = str(Path(parameter_search_path).joinpath("btap"))
    # Create the output directories if they do not exist
    config.create_directory(parameter_search_path)
    config.create_directory(log_path)

    tuner = Hyperband(model_builder,
                      objective='val_loss',
                      max_epochs=50,
                      overwrite=True,
                      factor=3,
                      directory=parameter_search_path,
                      project_name='btap',
                      seed=random_seed)

    stop_early = tf.keras.callbacks.EarlyStopping(monitor='loss', patience=5)
    tuner.search(X_train,
                 y_train,
                 epochs=100,
                 batch_size=365,
                 callbacks=[stop_early],
                 use_multiprocessing=True,
                 validation_split=0.2)

    tuner.search_space_summary()
    # Get the optimal hyperparameters
    best_hps = tuner.get_best_hyperparameters(num_trials=1)[0]

    print(f"""
    The hyperparameter search is complete. The optimal number of units in the first densely-connected
    layer is {best_hps.get('units_0')} and the optimal learning rate for the optimizer
    is {best_hps.get('learning_rate')}.
    """)
    result = best_hps

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
                        epochs=50,
                        batch_size=365,
                        validation_split=0.2,
                        callbacks=[stop_early, hist_callback],
                        )
    pl.save_plot(history)

    val_acc_per_epoch = history.history['mae']
    best_epoch = val_acc_per_epoch.index(max(val_acc_per_epoch)) + 1
    print('Best epoch: %d' % (best_epoch,))

    # Re-instantiate the hypermodel and train it with the optimal number of epochs from above.
    hypermodel = tuner.hypermodel.build(best_hps)
    hypermodel.fit(X_train, y_train, epochs=best_epoch, validation_split=0.2)

    return hypermodel


def evaluate(model, X_test, y_test, scalery, X_validate, y_validate, y_test_complete, y_validate_complete):
    """
    The model selected with the best hyperparameter is used to make predictions.
    Args:
        model: model built from training
        X_test: X testset
        y_test: y testset
        scalery: y scaler used to transform the y values to the original scale
        X_validate: X validationset
        y_validate: y validationset
        validate: validation dataset
    Returns:
        metric: evaluation results containing the loss value from the testset prediction,
        annual_metric: predicted value for each datapooint_id is summed to calculate the annual energy consumed and the loss value from the testset prediction,
        output_df: merge of y_pred, y_test, datapoint_id, the final dataframe showing the model output using the testset
        val_metric:evaluation results containing the loss value from the validationset prediction,
        val_annual_metric:predicted value for each datapooint_id is summed to calculate the annual energy consumed and the loss value from the validationset prediction,,
        output_val_df: merge of y_pred, y_validate, datapoint_id, the final dataframe showing the model output using the validation set
    """
    # evaluate the hypermodel on the test data.
    y_test['y_pred'] = model.predict(X_test)
    y_validate['y_pred'] = model.predict(X_validate)
    # Retrain the model
    y_test['y_pred_transformed'] = scalery.inverse_transform(y_test['y_pred'].values.reshape(-1,1))
    y_validate['y_pred_transformed'] = scalery.inverse_transform(y_validate['y_pred'].values.reshape(-1,1))
    test_score = score( y_test['energy'], y_test['y_pred_transformed'])
    val_score = score( y_validate['energy'], y_validate['y_pred_transformed'])

    print("[Score test loss, test mae, test mse]:", test_score)
    print("[Score val loss, val mae, val mse]:", val_score)

    y_test_complete = y_test_complete.groupby(['datapoint_id']).sum()
    y_test_complete['Total Energy'] = y_test_complete['Total Energy'].apply(lambda r: float(r / 365))
    output_df = ''
    y_test = y_test.groupby(['datapoint_id']).sum()
    y_test['energy'] = y_test['energy'].apply(lambda r : float((r*1.0)/1000))
    y_test['y_pred_transformed'] = y_test['y_pred_transformed'].apply(lambda r : float((r*1.0)/1000))
    output_df = pd.merge(y_test, y_test_complete, left_index=True, right_index=True, how='left')
    annual_metric=score(output_df['Total Energy'], output_df['y_pred_transformed'])

    y_validate_complete = y_validate_complete.groupby(['datapoint_id']).sum()
    y_validate_complete['Total Energy'] = y_validate_complete['Total Energy'].apply(lambda r: float(r / 365))
    output_val_df = ''
    y_validate = y_validate.groupby(['datapoint_id']).sum()
    y_validate['energy'] = y_validate['energy'].apply(lambda r : float((r*1.0)/1000))
    y_validate['y_pred_transformed'] = y_validate['y_pred_transformed'].apply(lambda r : float((r*1.0)/1000))


    output_val_df = pd.merge(y_validate, y_validate_complete, left_index=True, right_index=True, how='left')
    annual_metric_val = score(output_val_df['Total Energy'], output_val_df['y_pred_transformed'])

    output_df = output_df.drop(['y_pred', 'energy_y', 'energy_x'],axis=1)
    output_val_df = output_val_df.drop(['y_pred','energy_y','energy_x'],axis=1)

#     pl.daily_plot(y_test,'test_set')
#     pl.daily_plot(y_validate,'validation_set')

#     pl.annual_plot(output_df,'test_set')
#     pl.annual_plot(output_val_df,'validation_set')


    print('****************TEST SET****************************')
    print(output_df.head(50))
    print(annual_metric)

    print('****************VALIDATION SET****************************')
    print(output_val_df.head(50))
    print(annual_metric_val)

    results = {
            'test_daily_metric': test_score,
            'test_annual_metric':annual_metric,
            'output_df':output_df.values.tolist(),
            'val_daily_metric' : val_score,
            'val_annual_metric':annual_metric_val,
            'output_val_df':output_val_df.values.tolist(),
            }

    return results


def create_model(dense_layers, activation, optimizer, dropout_rate, length, learning_rate, epochs, batch_size, X_train, y_train, X_test, y_test, y_test_complete, scalery,
                 X_validate, y_validate, y_validate_complete, output_path):
    """
    Creates a model with defaulted values without need to perform an hyperparameter search at all times.
    Its initutive to have run the hyperparameter search beforehand to know the hyperparameter value to set.
    Args:
        dense_layers: number of layers for the model architecture e.g for a model with 3 layers, values will be passed as [8,20,30]
        activation: activation function to be used e.g relu, tanh
        optimizer: optimizer to be used in compiling the model e.g relu, rmsprop, adam
        dropout_rate: used to make the model avoid overfitting, value should be less than 1 e.g 0.3
        length: length of the trainset
        learning_rate: learning rate determines how fast or how slow the model will converge to an optimal loss value. Value should be less or equal 0.1 e.g 0.001
        epochs: number of iterations the model should perform
        X_train: X trainset
        y_train: y trainset
        X_test: X testset
        y_test: y testset
        y_test_complete: dataframe containing the target variable with corresponding datapointid for the test set
        scalery: y scaler used to transform the y values to the original scale
        X_validate: X validation set
        y_validate: y validation set
        y_validate_complete: dataframe containing the target variable with corresponding datapointid for the validation set
        output_path: Where the outputs will be placed/
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
                        kernel_regularizer=regularizers.l1_l2(l1=1e-2, l2=1e-2),
                        bias_regularizer=regularizers.l2(1e-2),
                        activity_regularizer=regularizers.l2(1e-2)
                        ))
        model.add(Dropout(dropout_rate))
    model.add(Dense(1, activation='linear'))

    if optimizer == "adam":
        optimizer = tf.optimizers.Adam(learning_rate=learning_rate)
    elif optimizer == "sgd":
        optimizer = tf.optimizers.SGD(learning_rate=learning_rate)
    else:
        optimizer = tf.optimizers.RMSprop(learning_rate=learning_rate)

    # Comiple the mode with the optimizer and learninf rate specified in hparams
    model.compile(optimizer=optimizer,
                  loss=rmse_loss,
                  metrics=['mae', 'mse', 'mape'])
    # Define callback
    early_stopping = EarlyStopping(monitor='loss', patience=5)
    logdir = os.path.join(btap_log_path, datetime.datetime.now().strftime("%Y%m%d-%H%M%S"))
    hist_callback = tf.keras.callbacks.TensorBoard(logdir, histogram_freq=1)
    logger = keras.callbacks.CSVLogger(output_path + '/metric.csv', append=True)
    output_df = ''

    # prepare the model with target scaling
    scores_metric = ''
    history = model.fit(X_train,
                        y_train,
                        callbacks=[logger,
                                   early_stopping,
                                   hist_callback,
                                   ],
                        epochs=epochs,
                        #batch_size=batch_size,
                        verbose=1,
                        #shuffle=False,
                        validation_split=0.2)
#     pl.save_plot(history)

    print(model.summary())
    plt.ylabel('loss')

    result = evaluate(model, X_test, y_test, scalery, X_validate, y_validate, y_test_complete, y_validate_complete)

    return result, model


def fit_evaluate(preprocessed_data_file, selected_features_file, param_search, output_path, random_seed):
    """
    Downloads the output from preprocessing and feature selection from mino, builds the model and then evaluate the model.
    Args:
         args: arguements provided from the main
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
    y_train = pd.read_json(preprocessing_json["y_train"], orient='values').values.ravel()

    # Extract the selected features from feature engineering
    X_train = X_train[selected_features]
    X_test = X_test[selected_features]
    X_validate = X_validate[selected_features]
    col_length = X_train.shape[1]

    # Extract the test data for the target variable
    y_test_complete = pd.DataFrame(preprocessing_json["y_test_complete"], columns=['energy', 'datapoint_id', 'Total Energy'])
    y_test = pd.DataFrame(preprocessing_json["y_test"], columns=['energy', 'datapoint_id'])
    y_validate_complete = pd.DataFrame(preprocessing_json["y_validate_complete"], columns=['energy', 'datapoint_id', 'Total Energy'])
    y_validate= pd.DataFrame(preprocessing_json["y_validate"], columns=['energy', 'datapoint_id'])

    # Scale the data to be used for training and testing
    scalerx = RobustScaler()
    scalery = RobustScaler()
    X_train = scalerx.fit_transform(X_train)
    X_test = scalerx.transform(X_test)
    X_validate = scalerx.transform(X_validate)
    y_train = scalery.fit_transform(y_train.reshape(-1, 1))

    # If set to "yes", search for best hyperparameters before training
    if param_search.lower() == "yes":
        hypermodel = predicts_hp(X_train, y_train, X_test, y_test, features, output_path, random_seed)
        results_pred = evaluate(hypermodel, X_test, y_test, scalery, X_validate, y_validate, y_test_complete, y_validate_complete)
    # Otherwise use a default model design and train with that model
    else:
        results_pred, hypermodel = create_model(
                                    dense_layers=[56],
                                    activation='relu',
                                    optimizer='adam',
                                    dropout_rate=0.1,
                                    length=col_length,
                                    learning_rate=0.001,
                                    epochs=20,
                                    batch_size=90,
                                    X_train=X_train,
                                    y_train=y_train,
                                    X_test=X_test,
                                    y_test=y_test,
                                    y_test_complete=y_test_complete,
                                    scalery=scalery,
                                    X_validate = X_validate,
                                    y_validate=y_validate,
                                    y_validate_complete= y_validate_complete,
                                    output_path=output_path
                                   )
    # Calculate the time spent training in minutes
    time_taken = ((time.time() - start_time) / 60)
    print("********* Total time spent is " + str(time_taken) + " minutes ***********" )

    # Define the path where the output files should be placed
    model_path = Path(output_path).joinpath(config.Settings().APP_CONFIG.TRAINING_BUCKET_NAME)
    config.create_directory(str(model_path))
    # Define the output files
    output_filename_json = str(model_path) + "/" + config.Settings().APP_CONFIG.TRAINING_RESULTS_FILENAME + ".json"
    output_filename_csv = str(model_path) + "/" + config.Settings().APP_CONFIG.TRAINING_RESULTS_FILENAME + ".csv"
    # Output the results within a csv for each prediction
    with open(output_filename_csv, 'a', encoding='utf-8') as csv_output:
        writer = csv.writer(csv_output)
        writer.writerow(['ID', 'Predicted energy', 'Actual energy'])
        for i, pair in enumerate(results_pred['output_df']):
            writer.writerow(['Test_' + str(i), pair[0], pair[1]])
        for i, pair in enumerate(results_pred['output_val_df']):
            writer.writerow(['Validation_' + str(i), pair[0], pair[1]])
    # Also output all training information within one json file
    with open(output_filename_json, 'w', encoding='utf8') as json_output:
        json.dump(results_pred, json_output)

    # Output the trained model architecture
    model_output_path = str(model_path.joinpath(config.Settings().APP_CONFIG.TRAINED_MODEL_FILENAME))
    hypermodel.save(model_output_path)
    # Output the scalers used to scale the X and y data
    joblib.dump(scalerx, str(model_path.joinpath(config.Settings().APP_CONFIG.SCALERX_FILENAME)))
    joblib.dump(scalery, str(model_path.joinpath(config.Settings().APP_CONFIG.SCALERY_FILENAME)))
    # Returns the model output filepath and the results output filepath
    return model_output_path, output_filename_csv

def main(config_file: str = typer.Argument(..., help="Location of the .yml config file (default name is input_config.yml)."),
         preprocessed_data_file: str = typer.Argument(..., help="Location and name of a .json preprocessing file to be used."),
         selected_features_file: str = typer.Argument(..., help="Location and name of a .json feature selection file to be used."),
         perform_param_search: str = typer.Option("no", help="'yes' if hyperparameter tuning should be performed (increases runtime), 'no' if the default hyperparameters should be used."),
         output_path: str = typer.Option("", help="The output path to be used. Note that this value should be empty unless this file is called from a pipeline."),
         random_seed: int = typer.Option(-1, help="Random seed to be used when training. Should not be -1 when used through the CLI."),
         ):
    """
    Select the feature which contribute most to the prediction for the total energy consumed.
    Default estimator_type used for feature selection is 'LassoCV'

    Args:
        config_file: Location of the .yml config file (default name is input_config.yml).
        preprocessed_data_file: Location and name of a .json preprocessing file to be used.
        selected_features_file: Location and name of a .json feature selection file to be used.
        perform_param_search: 'yes' if hyperparameter tuning should be performed (increases runtime), 'no' if the default hyperparameters should be used.
        output_path: Where output data should be placed. Note that this value should be empty unless this file is called from a pipeline.
        random_seed: Random seed to be used when training. Should not be -1 when used through the CLI.
    """
    DOCKER_INPUT_PATH = config.Settings().APP_CONFIG.DOCKER_INPUT_PATH
    # Load all content stored from the config file, if provided
    if len(config_file) > 0:
        # Load the specified config file
        cfg = config.get_config(DOCKER_INPUT_PATH + config_file)
        if random_seed < 0: random_seed = cfg.get(config.Settings().APP_CONFIG.RANDOM_SEED)
        if perform_param_search == "": perform_param_search = cfg.get(config.Settings().APP_CONFIG.PARAM_SEARCH)
    # Validate all input files
    # Validate all inputs
    input_model = PredictModel(input_prefix=DOCKER_INPUT_PATH,
                               preprocessed_data_file=preprocessed_data_file,
                               selected_features_file=selected_features_file,
                               perform_param_search=perform_param_search,
                               random_seed=random_seed)
    # If the output path is blank, map to the docker output path
    if len(output_path) < 1:
        output_path = config.Settings().APP_CONFIG.DOCKER_OUTPUT_PATH
    return fit_evaluate(input_model.preprocessed_data_file, input_model.selected_features_file, input_model.perform_param_search, output_path, input_model.random_seed)

if __name__ == '__main__':
    # Load settings from the environment
    settings = config.Settings()
    # Run the CLI interface
    typer.run(main)
