from sklearn.feature_selection import RFECV 
from sklearn.linear_model import LinearRegression, LassoCV, Lasso,ElasticNetCV
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import KFold
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn import linear_model
import numpy as np
import xgboost as xgb 
import json
import argparse
import pandas as pd
import s3fs
#from kfp.components import load_component_from_file

############################################################    
# feature selection
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

def select_features(args,estimator_type ='lasso', min_features=20): 
    """
    Select the feature which contribute most to the prediction for the total energy consumed.  
    """
    data = access_minio(tenant=args.tenant,
                           bucket=args.bucket,
                           operation= 'read',
                           path=args.in_obj_name,
                           data='')
#    
#     data = open('../preprocessing/output/preprocessing_out',)
    data = json.load(data)
    features =data["features"] 
    X_train = pd.DataFrame(data["X_train"],columns=features)
    X_test = pd.DataFrame(data["X_test"],columns=features)
    
  
    #standardize
    scalerx= MinMaxScaler()
    scalery= MinMaxScaler()
    X_train = scalerx.fit_transform(data["X_train"])
    y_train = pd.read_json(data["y_train"], orient='values').values.ravel()
    
    #y_train =scalery.fit_transform(y_train.reshape(-1, 1))
   
    if estimator_type == "linear":
        estimator = LinearRegression()
        rfecv = RFECV(estimator=estimator, step=1, cv=KFold(10),scoring='neg_mean_squared_error',min_features_to_select=min_features)
        #fit = rfecv.fit(data["X_train"],data["y_train"])
        fit = rfecv.fit(X_train,y_train)
        rank_features_nun = pd.DataFrame(rfecv.ranking_, columns=["rank"], index = data["features"])
        selected_features = rank_features_nun.loc[rank_features_nun["rank"]==1].index.tolist() 
        #print(selected_features)
        
        data = {'selected_features' : selected_features}
    
        # Creates a json object based on `data`
        data_json = json.dumps(data).encode('utf-8')
        #data_json = json.dumps(data)
        csv_buffer = io.BytesIO(data_json)
        client.put_object(bucket_name=args.bucket,
                  object_name=args.output_path,  
                  data=csv_buffer, 
                  length=len(data_json), 
                  content_type='application/csv')
        #return selected_features
    
    elif estimator_type == "rf":
#         estimator = RandomForestRegressor(**params, n_jobs = -1)
        estimator = RandomForestRegressor(n_jobs = -1)
        rfecv = RFECV(estimator=estimator, step=1, cv=KFold(10),scoring='neg_mean_squared_error', min_features_to_select=min_features)
        fit = rfecv.fit(X_train,y_train)
        rank_features_nun = pd.DataFrame(rfecv.ranking_, columns=["rank"], index = data["features"])
        selected_features = rank_features_nun.loc[rank_features_nun["rank"]==1].index.tolist() 
        #return selected_features
    elif estimator_type == "elasticnet":
        estimator = ElasticNetCV(n_jobs=-1,cv=10)
        rfecv = RFECV(estimator=estimator, step=1, cv=KFold(10),scoring='neg_mean_squared_error', min_features_to_select=min_features)
        fit = rfecv.fit(X_train,y_train)
        rank_features_nun = pd.DataFrame(rfecv.ranking_, columns=["rank"], index = data["features"])
        selected_features = rank_features_nun.loc[rank_features_nun["rank"]==1].index.tolist()
        score = rfecv.score(X_train,y_train)     
    elif estimator_type == "xgb":
        
        estimator = xgb.XGBRegressor(n_jobs = -1)
        #estimator = xgb.XGBRegressor(**params, n_jobs = multiprocessing.cpu_count())        
        rfecv = RFECV(estimator=estimator, step=1, cv=KFold(10),scoring='neg_mean_squared_error', min_features_to_select=min_features)
        fit = rfecv.fit(X_train,y_train)
        rank_features_nun = pd.DataFrame(rfecv.ranking_, columns=["rank"], index = data["features"])
        selected_features = rank_features_nun.loc[rank_features_nun["rank"]==1].index.tolist()
        print(selected_features)
        #return selected_features
    else:
        #estimator =LassoCV(cv=10, tol=0.001,max_iter=100000,n_jobs = -1,alphas=[0.1,0.001], normalize=True)
#         estimator =LassoCV(cv=10,n_jobs=-1)
#         rfecv = RFECV(estimator=estimator, step=1, cv=KFold(10),scoring='neg_mean_squared_error', min_features_to_select=min_features)
#         fit = rfecv.fit(X_train,y_train)
#         rank_features_nun = pd.DataFrame(rfecv.ranking_, columns=["rank"], index = data["features"])
#         selected_features = rank_features_nun.loc[rank_features_nun["rank"]==1].index.tolist()
#         print(len(selected_features))
#         print(selected_features)
       
        reg = linear_model.LassoCV(cv=10,n_jobs=-1,n_alphas=100)
        fit = reg.fit(X_train,y_train)
        score = reg.score(X_train,y_train)
        print('************************************lasso no rfecv*')
        print(score)
        rank_features_nun = pd.DataFrame(reg.coef_, columns=["rank"], index = data["features"])
        selected_features = rank_features_nun.loc[abs(rank_features_nun["rank"])>0].index.tolist()
        print(len(selected_features))
        print(selected_features)

    #share train and test datasets.
    data = {'features' : selected_features}
    data_json = json.dumps(data).encode('utf-8')
    
    #copy data to minio
    access_minio(tenant=args.tenant,
                 bucket=args.bucket,
                 operation='copy',
                 path=args.output_path,
                 data=data_json)

#     out_file =open('./output/feature_out','w')
#     json.dump(data, out_file, indent = 6) 
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
    parser.add_argument('--output_path', type=str, help='Path of the local file where the output file should be written.')
    args = parser.parse_args()
    
    #Path(args.output_path).parent.mkdir(parents=True, exist_ok=True)
    select_features(args)
    #python3 feature_selection.py --tenant standard --bucket nrcan-btap --in_obj_name output_data/preprocessing_out --output_path output_data/feature_out

    
