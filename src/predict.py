import tensorflow as tf
import tensorflow.keras as keras
from tensorflow.keras import layers
import tensorflow_docs as tfdocs
import tensorflow_docs.plots
import tensorflow_docs.modeling
import numpy as np
#np.random.seed(1337)
from keras import backend as K
from keras.models import Sequential
from keras.utils import np_utils
#from keras.layers import Dense, Dropout, GaussianNoise, Conv1D,Flatten
from keras.layers import  BatchNormalization
from keras.layers.core import Dense, Flatten, Dropout
from keras import regularizers      #for l2 regularization
from keras.wrappers.scikit_learn import KerasRegressor
from sklearn.metrics import mean_squared_error
from keras import backend as K
import os
import glob
from keras.callbacks import EarlyStopping
from tensorflow.keras.optimizers import Adam
from sklearn.compose import TransformedTargetRegressor
from sklearn.pipeline import Pipeline
from sklearn import metrics
from sklearn.preprocessing import StandardScaler,MinMaxScaler,Normalizer 
from sklearn.model_selection import cross_val_score,cross_val_predict
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import mean_absolute_error, mean_squared_error
from matplotlib import pyplot as plt
import json
import argparse
import pandas as pd
from sklearn.model_selection import KFold
import shutil
from math import sqrt
import keras_tuner as kt
import time
import datetime
from tensorboard.plugins.hparams import api as hp
import s3fs

#######################################################    
# Predict energy consumed
############################################################
def access_minio(tenant,bucket,path,operation,data):    
    with open(f'/vault/secrets/minio-{tenant}-tenant-1.json') as f:
        creds = json.load(f)
        
    minio_url = creds['MINIO_URL']
    access_key=creds['MINIO_ACCESS_KEY'],
    secret_key=creds['MINIO_SECRET_KEY']

    # Establish S3 connection
    s3 = s3fs.S3FileSystem(
        anon=False,
        key=access_key[0],
        secret=secret_key,
        #use_ssl=False, # Used if Minio is getting SSL verification errors.
        client_kwargs={
            'endpoint_url': minio_url,
            #'verify':False
        }
    )

    if operation == 'read':
            data = s3.open('{}/{}'.format(bucket, path), mode='rb')
    else:
        with s3.open('{}/{}'.format(bucket, path), 'wb') as f:
            f.write(data)   
    return data

def plot_metric(df):
    plt.figure()
    plt.plot(df['loss'],label='mae')
    plt.savefig('./output/mae.png')
    plt.ylabel('Metric')
    plt.xlabel('Epoch number')
    plt.figure()
    plt.plot(df['mae'],label='mae')
    plt.savefig('./output/mse.png')
    plt.ylabel('Metric')
    plt.xlabel('Epoch number')

    
    return


def save_plot(H):
    # plot the training loss and accuracy
    plt.style.use("ggplot")
    plt.figure()
    plt.plot(H.history["loss"], label="train_loss")
    plt.plot(H.history["val_loss"], label="val_loss")
    plt.xlabel("Epoch #")
    plt.ylabel("Loss")
    plt.legend()
    plt.savefig('../output/turner_train_test_loss.png')
    
    plt.style.use("ggplot")
    plt.figure(2)
    plt.plot(H.history["mae"], label="train_mae")
    plt.plot(H.history["val_mae"], label="val_mae")
    plt.title("Training Loss and MAE")
    plt.xlabel("Epoch #")
    plt.ylabel("MAE")
    plt.legend()
    plt.savefig('../output/turner_train_test_mae.png')
    
    return


def score(y_test,val_predict):    
#     sum_square_error = 0.0
#     for i in range(len(y_test)):
#         sum_square_error += (y_test[i] - val_predict[i])**2.0
    
    mse = metrics.mean_squared_error(y_test,val_predict)
    
    #mse = np.square(np.subtract(y_test,val_predict)).mean()
    mae = metrics.mean_absolute_error(y_test,val_predict)
    #mape = metrics.mean_absolute_percentage_error(y_test,val_predict)
    rmse = sqrt(mse)
    #r2_scores=metrics.r2_score(y_test,val_predict)    
    scores = {
              #"mape":mape,
              #"r2_scores": r2_scores,
              "mse":mse,
              "rmse": rmse,
              "mae":mae}
    return scores
def mae_loss(y_true, y_pred):        
    sum_pred =  K.sum(y_pred, axis=-1)
    sum_true = K.sum(y_true, axis=-1)
    loss = K.mean(K.abs(sum_pred - sum_true) )
    
    return loss

def mse_loss(y_true, y_pred):        
    sum_pred =  K.sum(y_pred, axis=-1)
    sum_true = K.sum(y_true, axis=-1)
    loss = K.mean(K.square(sum_pred - sum_true) )
    
    return loss

def rmse_loss(y_true, y_pred):        
    sum_pred =  K.sum(y_pred, axis=-1)
    sum_true = K.sum(y_true, axis=-1)
    loss = K.sqrt(K.mean(K.square(sum_pred - sum_true) ))
    
    return loss

    
def model_builder(hp):
    model = keras.Sequential()
#     model.add(keras.layers.Flatten())
    hp_activation= hp.Choice('activation', values=['relu','tanh','sigmoid'])
    hp_regularizers= hp.Choice('regularizers', values=[1e-4, 1e-5])
    for i in range(hp.Int("num_layers", 1,5)):
        model.add(layers.Dense(
                units=hp.Int("units_" + str(i), min_value=8, max_value=96, step=8),
                activation= hp_activation,
                input_shape=(36, ),
                kernel_initializer='normal',
                kernel_regularizer=regularizers.l1_l2(l1=1e-5,l2=1e-5),
                bias_regularizer=regularizers.l2(1e-4),
                activity_regularizer=regularizers.l2(1e-5)
            ))
        model.add(Dropout( hp.Choice('dropout_rate', values=[0.1,0.2,0.3,0.4])))
    model.add(Dense(1,activation='linear'))
    hp_learning_rate = hp.Choice('learning_rate', values=[1e-3,1e-4, 1e-5])
    hp_optimizer = hp.Choice('optimizer', values=['rmsprop','adam','sgd'])
           
    if hp_optimizer == "adam":
        optimizer = tf.optimizers.Adam(learning_rate=hp_learning_rate)
    elif hp_optimizer == "sgd":
        optimizer = tf.optimizers.SGD(learning_rate=hp_learning_rate)
    else: 
        optimizer = tf.optimizers.RMSprop(learning_rate=hp_learning_rate)
    
    # Comiple the mode with the optimizer and learninf rate specified in hparams
    model.compile(optimizer=optimizer,
              #loss='mean_squared_error',
              loss =rmse_loss,
              metrics=['mae','mse'])
    
    return model



def predicts_hp(X_train,y_train,X_test,y_test,selected_feature,validate):
    
    tuner = kt.Hyperband(model_builder,
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
    #Get the optimal hyperparameters
    best_hps=tuner.get_best_hyperparameters(num_trials=1)[0]

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
                        callbacks=[stop_early,hist_callback],
                        )
    save_plot(history)
    
    val_acc_per_epoch = history.history['mae']
    best_epoch = val_acc_per_epoch.index(max(val_acc_per_epoch)) + 1
    print('Best epoch: %d' % (best_epoch,))
    
    #Re-instantiate the hypermodel and train it with the optimal number of epochs from above.
    hypermodel = tuner.hypermodel.build(best_hps)
    hypermodel.fit(X_train, y_train, epochs=best_epoch, validation_split=0.2)
    
    
    return hypermodel

def hyper_evaluate(hypermodel,X_test,y_test,scalery,validate):
    #evaluate the hypermodel on the test data.
    eval_result = hypermodel.evaluate(X_test, y_test['energy'])
    y_test['y_pred'] = hypermodel.predict(X_test)
    # Retrain the model
    y_test['y_pred_transformed'] =scalery.inverse_transform(y_test['y_pred'].values.reshape(-1,1))

    #y_pred =scalery.inverse_transform(y_pred)
    print("[test loss, test mae, test mse]:", eval_result)
    validate = validate.groupby(['datapoint_id']).sum()
    validate['Total Energy'] = validate['Total Energy'].apply(lambda r : float(r/365))
    output_df =''
    y_test = y_test.groupby(['datapoint_id']).sum()
    y_test['energy'] =y_test['energy'].apply(lambda r : float((r*1.0)/1000))
    y_test['y_pred'] =y_test['y_pred'].apply(lambda r : float((r*1.0)/1000))
    
    output_df = pd.merge(y_test,validate,left_index=True, right_index=True,how='left')
    annual_metric=score(output_df['Total Energy'],output_df['y_pred_transformed'])
    
    print('********************annual metric below')
    print(annual_metric)
    
    result= {
            'best_epoch':best_epoch,
            'metric' : eval_result,
            'annual_metric':annual_metric,
            'output_df':output_df.values.tolist(),
            }

    return result


def predict(model,X_test,y_test,scalery,test_complete):
    y_test['y_pred']  = model.predict(X_test)
    y_test['y_pred_transformed'] =scalery.inverse_transform(y_test['y_pred'].values.reshape(-1,1))
   
    #eval_result = model.evaluate(X_test,y_test['energy'],batch_size=30)
    eval_result = model.evaluate(X_test,y_test['energy'])
    scores_metric  = score(y_test['energy'],y_test['y_pred_transformed'])

    #aggregating annual energy
    test_complete = test_complete.groupby(['datapoint_id']).sum()
    test_complete['Total Energy'] = test_complete['Total Energy'].apply(lambda r : float(r/365))
    
    #converting to GJ
    y_test = y_test.groupby(['datapoint_id']).sum()
    y_test['energy'] =y_test['energy'].apply(lambda r : float((r*1.0)/1000))
    y_test['y_pred_transformed'] =y_test['y_pred_transformed'].apply(lambda r : float((r*1.0)/1000))
    
    #merging to create the output dataframe
    output_df = pd.merge(y_test,test_complete,left_index=True, right_index=True,how='left')
    output_df = output_df.drop(['y_pred','energy_y','energy_x'],axis=1)
    annual_metric=score(output_df['Total Energy'],output_df['y_pred_transformed'])
    
    print(output_df)
    print('****************TEST SET****************************')
    print(eval_result)
    print(scores_metric)
    print('********************annual metric below')
    print(annual_metric)
   
    result= scores_metric,annual_metric,output_df
    
    plt.style.use("ggplot")
    plt.figure()
    plt.plot(output_df['Total Energy'],label='actual')
    plt.plot(output_df['y_pred_transformed'],label='pred')
    plt.title("Annual Energy")
    plt.savefig('../output/annual_energy.png')
    
     
    plt.style.use("ggplot")
    plt.figure()
    #plt.hist(y_train,label='train')
    plt.hist(output_df['Total Energy'],label='test')
    plt.savefig('../output/daily_energy_train.png')
    return result   



def create_model(dense_layers,activation,optimizer,dropout_rate,length,learning_rate,epochs,X_train,y_train,X_test,y_test,test_complete,scalery,
                X_validate,y_validate,validate_complete):
    
    model = Sequential()
    model.add(Flatten(input_shape=(length,)))
    #model.add(Dropout(dropout_rate, input_shape=(length,)))
    for index, lsize in enumerate(dense_layers):
        model.add(Dense(lsize,activation=activation, kernel_initializer='normal', 
                        # kernel_regularizer=regularizers.l1(1e-5),
                        kernel_regularizer=regularizers.l1_l2(l1=1e-2, l2=1e-2),
                        bias_regularizer=regularizers.l2(1e-2),
                        activity_regularizer=regularizers.l2(1e-2)
        
                ))
        model.add(Dropout(dropout_rate))
        #model.add(BatchNormalization())
    model.add(Dense(1,activation='linear'))
   
    if optimizer == "adam":
        optimizer = tf.optimizers.Adam(learning_rate=learning_rate)
    elif optimizer == "sgd":
        optimizer = tf.optimizers.SGD(learning_rate=learning_rate)
    else: 
        optimizer = tf.optimizers.RMSprop(learning_rate=learning_rate)

    # Comiple the mode with the optimizer and learninf rate specified in hparams
    model.compile(optimizer=optimizer,
              #loss='mean_squared_error',
              #loss=CustomMSE(),
              loss = rmse_loss,
              metrics=['mae','mse'])
              #metrics =[mse_loss,mae_loss]
    # Define callback
    early_stopping = EarlyStopping(monitor='loss', patience=5)
    logdir = os.path.join("../output/parameter_search/btap", datetime.datetime.now().strftime("%Y%m%d-%H%M%S"))
    hist_callback = tf.keras.callbacks.TensorBoard(logdir, histogram_freq=1)
    logger = keras.callbacks.CSVLogger('../output/metric.csv', append=True)
    output_df =''
    
    # prepare the model with target scaling
    scores_metric=''
    np.random.seed(7)
    history = model.fit(X_train,
                        y_train,
                        callbacks=[logger,
                                    early_stopping,
                                    hist_callback,
                                    #tfdocs.modeling.EpochDots()
                                    ],
                        epochs=epochs,
                        #batch_size =batch_size,                    
                        verbose=1,
                        # shuffle=False,
                        validation_split=0.1)
    save_plot(history)
    
    
    print(model.summary())
    #model.summary()
    #plotter = tfdocs.plots.HistoryPlotter(smoothing_std=2)
    plt.ylabel('loss')
    


    scores_metric,annual_metric,output_df = predict(model,X_test,y_test,scalery,test_complete)
    scores_metric_val,annual_metric_val,output_val_df = predict(model,X_validate,y_validate,scalery,validate_complete)

    result= {
            'metric' : scores_metric,
            'annual_metric':annual_metric,
            'output_df':output_df.values.tolist(),
            'val_metric' : scores_metric_val,
            'val_annual_metric':annual_metric_val,
            'output_val_df':output_val_df.values.tolist(),
            }
   
    return result
    

def fit_evaluate(args): 
    #Resets all state generated by Keras.
    K.clear_session()
    start_time = time.time()
    data = access_minio(tenant=args.tenant,
                           bucket=args.bucket,
                           operation= 'read',
                           path=args.in_obj_name,
                           data='')
    
    data2 = access_minio(tenant=args.tenant,
                           bucket=args.bucket,
                           operation= 'read',
                           path=args.features,
                           data='')
#     data = open('../preprocessing/output/preprocessing_out',)
#     data2 = open('../feature_Selection/output/feature_out',)
    
    
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
    X_train = pd.DataFrame(data["X_train"],columns=features)
    X_test = pd.DataFrame(data["X_test"],columns=features)
    X_validate= pd.DataFrame(data["X_validate"],columns=features)
    y_train = pd.read_json(data["y_train"], orient='values').values.ravel()
    
    #extracting the selected features from feature engineering
    X_train = X_train[selected_features]
    X_test = X_test[selected_features]
    X_validate = X_validate[selected_features]

    col_length = X_train.shape[1]

    #extracting the test data for the target variable
    y_valid = pd.DataFrame(data["y_valid"],columns=['energy','datapoint_id','Total Energy'])
    y_test = pd.DataFrame(data["y_test"],columns=['energy','datapoint_id'])
    validate_complete = pd.DataFrame(data["y_complete"],columns=['energy','datapoint_id','Total Energy'])
    y_validate = pd.DataFrame(data["y_validate"],columns=['energy','datapoint_id'])
    
    scalerx= Normalizer()
    scalery= StandardScaler()
    X_train = scalerx.fit_transform(X_train)
    X_test = scalerx.transform(X_test)
    X_validate = scalerx.transform(X_validate)
    y_train = scalery.fit_transform(y_train.reshape(-1, 1))
    
    np.random.seed(7)
    #search for best hyperparameters 
    if args.param_search == "yes":
        hypermodel = predicts_hp(X_train,y_train,X_test,y_test,features,y_valid)
        results_pred = hyper_evaluate(hypermodel,X_test,y_test,scalery,y_valid)
        print(results_pred)
        results_pred = hyper_evaluate(hypermodel,X_validate,y_validate,scalery,validate_complete)
        print(results_pred)
    else:
        results_pred = create_model(
                             dense_layers=[50,10],
                             activation='relu',
                             optimizer='rmsprop',
                             dropout_rate=0.1,
                             length=col_length,
                             learning_rate=0.001,
                             epochs=2,
                             #batch_size=90,
                             X_train=X_train,y_train=y_train,
                             X_test=X_test,y_test=y_test
                             ,test_complete=y_valid,scalery=scalery,
                             X_validate = X_validate, y_validate=y_validate,
                             validate_complete= validate_complete,
                             
                            )
    time_taken = ((time.time() - start_time)/60)
    print("********* Total time spent is ***********" + str(time_taken)+" minutes" )
    
    data_json = json.dumps(results_pred).encode('utf-8')
    
    #copy data to minio
    access_minio(tenant=args.tenant,
                 bucket=args.bucket,
                 operation='copy',
                 path=args.output_path,
                 data=data_json)

#     out_file =open('./output/predict_out','w')
#     json.dump(results_pred, out_file, indent = 6) 
#     out_file.close()
    
    return 


if __name__ == '__main__':
    
    # This component does not receive any input it only outpus one artifact which is `data`.
    # Defining and parsing the command-line arguments
    parser = argparse.ArgumentParser()
    
     # Paths must be passed in, not hardcoded
    parser.add_argument('--tenant', type=str, help='The minio tenant where the data is located in')
    parser.add_argument('--bucket', type=str, help='The minio bucket where the data is located in')
    parser.add_argument('--in_obj_name', type=str, help='Name of data file to be read')
    parser.add_argument('--features', type=str, help='selected features')
    parser.add_argument('--param_search', type=str, help='param search')
    parser.add_argument('--output_path', type=str, help='Path of the local file where the output file should be written.')
    args = parser.parse_args()
    
    #Path(args.output_path).parent.mkdir(parents=True, exist_ok=True)
    fit_evaluate(args)
    
     #python3 predict.py --tenant standard --bucket nrcan-btap --param_search no --in_obj_name output_data/preprocessing_out --features output_data/feature_out --output_path output_data/predict_out 
    
    #launch tensorboard
    #python -m tensorboard.main --logdir="./parameter_search/btap/"
    #https://kubeflow.aaw.cloud.statcan.ca/notebook/nrcan-btap/reg-cpu-notebook/proxy/6007/
    