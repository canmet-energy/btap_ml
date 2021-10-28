'''
Select features that are used to build the surrogate mode. 

'''
from sklearn.feature_selection import RFECV 
from sklearn.linear_model import LinearRegression, LassoCV, Lasso,ElasticNetCV
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import KFold
from sklearn.preprocessing import StandardScaler, MinMaxScaler,Normalizer
from sklearn import linear_model
import numpy as np
import xgboost as xgb 
import json
import argparse
import pandas as pd
import s3fs
import config as acm

############################################################    
# feature selection
############################################################

def select_features(args,estimator_type ='lasso', min_features=20): 
    """
    Select the feature which contribute most to the prediction for the total energy consumed.
    
    Default estimator_type used for feature selection is 'LassoCV'
    
    Args:
        args: arguements provided from the main

    Returns:
       selected features are returned and uploaded to minio. 
       
    """
    data = acm.access_minio(tenant=args.tenant,
                           bucket=args.bucket,
                           operation= 'read',
                           path=args.in_obj_name,
                           data='')

    data = json.load(data)
    features =data["features"] 
    X_train = pd.DataFrame(data["X_train"],columns=features)
    X_test = pd.DataFrame(data["X_test"],columns=features)
    
  
    #normalize
    scalerx= Normalizer()
    scalery= Normalizer()
    X_train = scalerx.fit_transform(data["X_train"])
    y_train = pd.read_json(data["y_train"], orient='values').values.ravel()
    
    if estimator_type == "linear":
        estimator = LinearRegression()
        rfecv = RFECV(estimator=estimator, step=1, cv=KFold(10),scoring='neg_mean_squared_error',min_features_to_select=min_features)
        fit = rfecv.fit(X_train,y_train)
        rank_features_nun = pd.DataFrame(rfecv.ranking_, columns=["rank"], index = data["features"])
        selected_features = rank_features_nun.loc[rank_features_nun["rank"]==1].index.tolist() 
    
    elif estimator_type == "elasticnet":
        reg = ElasticNetCV(n_jobs=-1,cv=10)
        fit = reg.fit(X_train,y_train)
        rank_features_nun = pd.DataFrame(rfecv.ranking_, columns=["rank"], index = data["features"])
        selected_features = rank_features_nun.loc[rank_features_nun["rank"]==1].index.tolist()
        score = rfecv.score(X_train,y_train)
        rank_features_nun = pd.DataFrame(reg.coef_, columns=["rank"], index = data["features"])
        selected_features = rank_features_nun.loc[abs(rank_features_nun["rank"])>0].index.tolist()
        
    elif estimator_type == "xgb":
        estimator = xgb.XGBRegressor(n_jobs = -1)
        rfecv = RFECV(estimator=estimator, step=1, cv=KFold(10),scoring='neg_mean_squared_error', min_features_to_select=min_features)
        fit = rfecv.fit(X_train,y_train)
        rank_features_nun = pd.DataFrame(rfecv.ranking_, columns=["rank"], index = data["features"])
        selected_features = rank_features_nun.loc[rank_features_nun["rank"]==1].index.tolist()
    else:
        reg = linear_model.LassoCV(cv=10,n_jobs=-1,n_alphas=100,tol=600)
        fit = reg.fit(X_train,y_train)
        score = reg.score(X_train,y_train)
        rank_features_nun = pd.DataFrame(reg.coef_, columns=["rank"], index = data["features"])
        selected_features = rank_features_nun.loc[abs(rank_features_nun["rank"])>0.3].index.tolist()
        print(score)
        print(len(selected_features))
        print(selected_features)

    data = {'features' : selected_features}
    data_json = json.dumps(data).encode('utf-8')
    
    #copy data to minio
    acm.access_minio(tenant=args.tenant,
                 bucket=args.bucket,
                 operation='copy',
                 path=args.output_path,
                 data=data_json)
    
    return              
        

if __name__ == '__main__':
    
    parser = argparse.ArgumentParser()
    
    # Paths must be passed in, not hardcoded
    parser.add_argument('--tenant', type=str, help='The minio tenant where the data is located in')
    parser.add_argument('--bucket', type=str, help='The minio bucket where the data is located in')
    parser.add_argument('--in_obj_name', type=str, help='Name of data file to be read')
    parser.add_argument('--estimator_type', type=str, help='Name of data file to be read')
    parser.add_argument('--output_path', type=str, help='Path of the local file where the output file should be written.')
    args = parser.parse_args()
    
    select_features(args)
    #python3 feature_selection.py --tenant standard --bucket nrcan-btap --in_obj_name output_data/preprocessing_out --output_path output_data/feature_out --estimator_type elasticnet

    
