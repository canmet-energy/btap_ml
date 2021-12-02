'''
Select features that are used to build the surrogate mode.

Args:
    in_obj_name: minio locationa and name of data file to be read, ideally the output file generated from preprocessing i.e. preprocessing.out
    estimator_type: Type of estimator to be used, default is lasso
    output_path: he minio location and filename where the output file should be written.
'''
import argparse
import json
import logging

import numpy as np
import pandas as pd
import s3fs

############################################################
# feature selection
############################################################

logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

def select_features(args):
    """
    Select the feature which contribute most to the prediction for the total energy consumed.

    Default estimator_type used for feature selection is 'LassoCV'

    Args:
        args: arguements provided from the main

    Returns:
       selected features are returned and uploaded to minio.
    """
    data = acm.access_minio(operation='read',
                            path=args.in_obj_name,
                            data='')
    logger.info("read from mino  ", data)

    data = json.load(data)
    features =data["features"]
    X_train = pd.DataFrame(data["X_train"],columns=features)
    print(X_train)
    #standardize
    scalerx= RobustScaler()
    scalery= RobustScaler()
    X_train = scalerx.fit_transform(data["X_train"])
    y_train = pd.read_json(data["y_train"], orient='values').values.ravel()

    if args.estimator_type == "linear":
        estimator = LinearRegression()
        rfecv = RFECV(estimator=estimator, step=1, cv=KFold(10), scoring='neg_mean_squared_error')
        fit = rfecv.fit(X_train, y_train)
        rank_features_nun = pd.DataFrame(rfecv.ranking_, columns=["rank"], index=data["features"])
        selected_features = rank_features_nun.loc[rank_features_nun["rank"] == 1].index.tolist()
    elif args.estimator_type == "elasticnet":
        reg = ElasticNetCV(n_jobs=-1, cv=10)
        fit = reg.fit(X_train, y_train)
        rank_features_nun = pd.DataFrame(rfecv.ranking_, columns=["rank"], index=data["features"])
        selected_features = rank_features_nun.loc[rank_features_nun["rank"] == 1].index.tolist()
        score = rfecv.score(X_train, y_train)
        rank_features_nun = pd.DataFrame(reg.coef_, columns=["rank"], index=data["features"])
        selected_features = rank_features_nun.loc[abs(rank_features_nun["rank"]) > 0].index.tolist()
    elif args.estimator_type == "xgb":
        estimator = xgb.XGBRegressor(n_jobs=-1)
        rfecv = RFECV(estimator=estimator, step=1, cv=KFold(10), scoring='neg_mean_squared_error')
        fit = rfecv.fit(X_train, y_train)
        rank_features_nun = pd.DataFrame(rfecv.ranking_, columns=["rank"], index=data["features"])
        selected_features = rank_features_nun.loc[rank_features_nun["rank"] == 1].index.tolist()
    else:
        reg = linear_model.LassoCV(cv=10,n_jobs=-1,n_alphas=100,tol=600)
        fit = reg.fit(X_train,y_train)
        score = reg.score(X_train,y_train)
        rank_features_nun = pd.DataFrame(reg.coef_, columns=["rank"], index = data["features"])
        selected_features = rank_features_nun.loc[abs(rank_features_nun["rank"])>0].index.tolist()
        print(score)
        print(len(selected_features))
        print(selected_features)

    data = {'features': selected_features}
    data_json = json.dumps(data).encode('utf-8')

    # copy data to minio
    write_to_minio = acm.access_minio(operation='copy',
                     path=args.output_path,
                     data=data_json)
    logger.info("write to mino  ", write_to_minio)

    return


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    # Paths must be passed in, not hardcoded
    parser.add_argument('--in_obj_name', type=str, help='minio locationa and name of data file to be read, ideally the output file generated from preprocessing i.e. preprocessing.out')
    parser.add_argument('--estimator_type', type=str, help='Type of estimator to be used, default is lasso')
    parser.add_argument('--output_path', type=str, help='The minio location and filename where the output file should be written.')
    args = parser.parse_args()

    select_features(args)
    # python3 feature_selection.py --in_obj_name output_data/preprocessing_out --output_path output_data/feature_out --estimator_type lasso
