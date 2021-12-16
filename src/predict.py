'''
Uses the output from preprocessing and feature selection from mino, builds the model and then evaluate the model.

Args:
    in_obj_name: minio locationa and name of data file to be read, ideally the output file generated from preprocessing i.e. preprocessing_out
    features: minio locationa and name of data file to be read, ideally the output file generated from feature selection i.e. feature_out
    param_search: This parameter is used to determine if hyperparameter search can be performed or not, accepted value is yes or no
    output_path: The minio location and filename where the output file should be written.
'''
import argparse
import datetime
import glob
import json
import logging
import os
import shutil
import time
from math import sqrt

import numpy as np
import pandas as pd
import s3fs
import tensorflow as tf
import tensorflow.keras as keras
from keras import backend as K
from keras import regularizers  # for l2 regularization
from keras.callbacks import EarlyStopping
from keras.layers import BatchNormalization
from keras.layers.core import Dense, Dropout, Flatten
from keras.models import Sequential
from keras.utils import np_utils
from keras.wrappers.scikit_learn import KerasRegressor
from keras_tuner import Hyperband
from matplotlib import pyplot as plt
from sklearn import metrics
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.model_selection import (GridSearchCV, KFold, cross_val_predict,
                                     cross_val_score)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import (MinMaxScaler, Normalizer, RobustScaler,
                                   StandardScaler)
from tensorboard.plugins.hparams import api as hp
from tensorflow.keras import layers
from tensorflow.keras.optimizers import Adam

import config as acm
import plot as pl

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

#######################################################
# Predict energy consumed
############################################################


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
#     hp_regularizers= hp.Choice('regularizers', values=[1e-4, 1e-5])
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


def predicts_hp(X_train, y_train, X_test, y_test, selected_feature):
    """
    Using the set of hyperparameter combined,the model built is used to make predictions

    Args:
        X_train: X train set
        y_train: y train set
        X_test: X test set
        y_test: y test set
        selected_feature: selected features that would be used to build the model

    Returns:
       Model built from the set of hyperparameters combined.
    """
    tuner = Hyperband(model_builder,
                         objective='val_loss',
                         max_epochs=50,
                         overwrite=True,
                         factor=3,
                         directory='../output/parameter_search',
                         project_name='btap')

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

    logdir = os.path.join("../output/parameter_search/btap", datetime.datetime.now().strftime("%Y%m%d-%H%M%S"))
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
    y_test['y_pred_transformed'] =scalery.inverse_transform(y_test['y_pred'].values.reshape(-1,1))
    y_validate['y_pred_transformed'] =scalery.inverse_transform(y_validate['y_pred'].values.reshape(-1,1))

    test_score = score( y_test['energy'], y_test['y_pred_transformed'])
    val_score = score( y_validate['energy'], y_validate['y_pred_transformed'])


    print("[Score test loss, test mae, test mse]:", test_score)
    print("[Score val loss, val mae, val mse]:", val_score)

    y_test_complete = y_test_complete.groupby(['datapoint_id']).sum()
    y_test_complete['Total Energy'] = y_test_complete['Total Energy'].apply(lambda r: float(r / 365))
    output_df = ''
    y_test = y_test.groupby(['datapoint_id']).sum()
    y_test['energy'] =y_test['energy'].apply(lambda r : float((r*1.0)/1000))
    y_test['y_pred_transformed'] =y_test['y_pred_transformed'].apply(lambda r : float((r*1.0)/1000))
    output_df = pd.merge(y_test,y_test_complete,left_index=True, right_index=True,how='left')
    annual_metric=score(output_df['Total Energy'],output_df['y_pred_transformed'])

    y_validate_complete = y_validate_complete.groupby(['datapoint_id']).sum()
    y_validate_complete['Total Energy'] = y_validate_complete['Total Energy'].apply(lambda r: float(r / 365))
    output_val_df = ''
    y_validate = y_validate.groupby(['datapoint_id']).sum()
    y_validate['energy'] =y_validate['energy'].apply(lambda r : float((r*1.0)/1000))
    y_validate['y_pred_transformed'] =y_validate['y_pred_transformed'].apply(lambda r : float((r*1.0)/1000))


    output_val_df = pd.merge(y_validate,y_validate_complete,left_index=True, right_index=True,how='left')
    annual_metric_val=score(output_val_df['Total Energy'],output_val_df['y_pred_transformed'])

    output_df = output_df.drop(['y_pred','energy_y','energy_x'],axis=1)
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

    result= {
            'test_daily_metric': test_score,
            'test_annual_metric':annual_metric,
            'output_df':output_df.values.tolist(),
            'val_daily_metric' : val_score,
            'val_annual_metric':annual_metric_val,
            'output_val_df':output_val_df.values.tolist(),
            }

    return result


def create_model(dense_layers, activation, optimizer, dropout_rate, length, learning_rate, epochs, batch_size, X_train, y_train, X_test, y_test, y_test_complete, scalery,
                 X_validate, y_validate, y_validate_complete):
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

    Returns:
        metric: evaluation results containing the loss value from the testset prediction,
        annual_metric: predicted value for each datapooint_id is summed to calculate the annual energy consumed and the loss value from the testset prediction,
        output_df: merge of y_pred, y_test, datapoint_id, the final dataframe showing the model output using the testset
        val_metric:evaluation results containing the loss value from the validationset prediction,
        val_annual_metric:predicted value for each datapooint_id is summed to calculate the annual energy consumed and the loss value from the validationset prediction,,
        output_val_df: merge of y_pred, y_validate, datapoint_id, the final dataframe showing the model output using the validation set
    """
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
    logdir = os.path.join("../output/parameter_search/btap", datetime.datetime.now().strftime("%Y%m%d-%H%M%S"))
    hist_callback = tf.keras.callbacks.TensorBoard(logdir, histogram_freq=1)
    logger = keras.callbacks.CSVLogger('../output/metric.csv', append=True)
    output_df = ''

    # prepare the model with target scaling
    scores_metric = ''
    np.random.seed(7)
    history = model.fit(X_train,
                        y_train,
                        callbacks=[logger,
                                   early_stopping,
                                   hist_callback,
                                   ],
                        epochs=epochs,
                        #batch_size =batch_size,
                        verbose=1,
                        # shuffle=False,
                        validation_split=0.2)
#     pl.save_plot(history)


    print(model.summary())
    plt.ylabel('loss')

    result = evaluate(model, X_test, y_test, scalery, X_validate, y_validate, y_test_complete, y_validate_complete)

    return result


def fit_evaluate(args):
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

    data = acm.access_minio(operation='read',
                            path=args.in_obj_name,
                            data='')

    data2 = acm.access_minio(operation='read',
                             path=args.features,
                             data='')
    logger.info("read_output s3 connection %s", data)

    # removing log directory
    shutil.rmtree('../output/parameter_search/btap', ignore_errors=True)

    try:
        os.remove('../output/metric.csv')
    except OSError:
        pass

    data = json.load(data)
    data2 = json.load(data2)

    features = data["features"]
    selected_features = data2["features"]
    X_train = pd.DataFrame(data["X_train"], columns=features)
    X_test = pd.DataFrame(data["X_test"], columns=features)
    X_validate = pd.DataFrame(data["X_validate"], columns=features)
    y_train = pd.read_json(data["y_train"], orient='values').values.ravel()

    # extracting the selected features from feature engineering
    X_train = X_train[selected_features]
    X_test = X_test[selected_features]
    X_validate = X_validate[selected_features]
    col_length = X_train.shape[1]

    #extracting the test data for the target variable
    y_test_complete = pd.DataFrame(data["y_test_complete"],columns=['energy','datapoint_id','Total Energy'])
    y_test = pd.DataFrame(data["y_test"],columns=['energy','datapoint_id'])
    y_validate_complete = pd.DataFrame(data["y_validate_complete"],columns=['energy','datapoint_id','Total Energy'])
    y_validate= pd.DataFrame(data["y_validate"],columns=['energy','datapoint_id'])

    scalerx= RobustScaler()
    scalery= RobustScaler()
    X_train = scalerx.fit_transform(X_train)
    X_test = scalerx.transform(X_test)
    X_validate = scalerx.transform(X_validate)
    y_train = scalery.fit_transform(y_train.reshape(-1, 1))

    np.random.seed(7)
    # search for best hyperparameters
    if args.param_search == "yes":
        hypermodel = predicts_hp(X_train, y_train, X_test, y_test, features)
        results_pred = evaluate(hypermodel, X_test, y_test, scalery, X_validate, y_validate, y_test_complete, y_validate_complete)
    else:
        results_pred = create_model(
                             dense_layers=[56],
                             #dense_layers=[88],
                             activation='relu',
                             optimizer='adam',
                             dropout_rate=0.1,
                             length=col_length,
                             learning_rate=0.001,
                             epochs=20,
                             batch_size=90,
                             X_train=X_train,y_train=y_train,
                             X_test=X_test,y_test=y_test
                             ,y_test_complete=y_test_complete,scalery=scalery,
                             X_validate = X_validate, y_validate=y_validate,
                             y_validate_complete= y_validate_complete,

                            )
    time_taken = ((time.time() - start_time)/60)
    print("********* Total time spent is ***********" + str(time_taken)+" minutes" )

    data_json = json.dumps(results_pred).encode('utf-8')

    # copy data to minio
    write_to_minio = acm.access_minio(operation='copy',
                     path=args.output_path,
                     data=data_json)

    logger.info("write to mino  %s", write_to_minio)

    return


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    # Paths must be passed in, not hardcoded
    parser.add_argument('--in_obj_name', type=str, help='minio locationa and name of data file to be read, ideally the output file generated from preprocessing i.e. preprocessing_out')
    parser.add_argument('--features', type=str, help='minio locationa and name of data file to be read, ideally the output file generated from feature selection i.e. feature_out')
    parser.add_argument('--param_search', type=str, help='This parameter is used to determine if hyperparameter search can be performed or not, accepted value is yes or no')
    parser.add_argument('--output_path', type=str, help='The minio location and filename where the output file should be written.')
    args = parser.parse_args()

    fit_evaluate(args)

    # python3 predict.py --param_search no --in_obj_name output_data/preprocessing_out --features output_data/feature_out --output_path output_data/predict_out.json

    # launch tensorboard
    # python -m tensorboard.main --logdir="./parameter_search/btap/"
    # https://kubeflow.aaw.cloud.statcan.ca/notebook/nrcan-btap/reg-cpu-notebook/proxy/6007/
