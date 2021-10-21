import numpy as np
import pandas as pd
import argparse
from pathlib import Path
from minio import Minio
from sklearn.model_selection import train_test_split, KFold,cross_val_score,cross_val_predict
from sklearn.preprocessing import StandardScaler, LabelEncoder, OneHotEncoder,MinMaxScaler
from sklearn.compose import ColumnTransformer
#from kfp import dsl
import io
from io import StringIO
import os
import glob
#from kfp.components import load_component_from_file
import plot as pl
import sys
from datetime import datetime
import re
from sklearn.model_selection import GroupShuffleSplit
import json
import s3fs

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
            if 'csv' in path :
                data = pd.read_csv(s3.open('{}/{}'.format(bucket, path), mode='rb'))
            else:
                data =  pd.read_excel(s3.open('{}/{}'.format(bucket, path), mode='rb'))
    else:
        with s3.open('{}/{}'.format(bucket, path), 'wb') as f:
            f.write(data)
       
    return data


def clean_data(df):
    # dropping columns with one unique value
    for col in df.columns:
        if ((len(df[col].unique()) ==1) and (col not in ['energy_eui_additional_fuel_gj_per_m_sq','energy_eui_electricity_gj_per_m_sq','energy_eui_natural_gas_gj_per_m_sq'])):
            df.drop(col,inplace=True,axis=1)
   
    # Drop any column with more than 50% missing values
    half_count = len(df) / 2
    df = df.dropna(thresh=half_count,axis=1)
    
    #Again, there may be some columns with more than one unique value, but one value that has insignificant frequency in the data set. 
    for col in df.columns:
        num = len(df[col].unique())
    
        if ((len(df[col].unique()) <3) and (col not in ['energy_eui_additional_fuel_gj_per_m_sq','energy_eui_electricity_gj_per_m_sq','energy_eui_natural_gas_gj_per_m_sq'])):
            df.drop(col,inplace=True,axis=1)
    return df

def read_output(tenant,bucket,path):
#     output_data = './input/output.xlsx'
    btap_df = access_minio(tenant=tenant,
                               bucket=bucket,
                               operation= 'read',
                               path=path,
                               data='')
    
#     btap_df = pd.read_excel(output_data, engine='openpyxl')
    floor_sq = btap_df['bldg_conditioned_floor_area_m_sq'].unique()
    
    #dropping output features present in the output file and dropping columns with one unique value
    output_drop_list =['Unnamed: 0',':erv_package',  ':template'] 
    for col in btap_df.columns:
        if ((':' not in col) and (col not in ['energy_eui_additional_fuel_gj_per_m_sq','energy_eui_electricity_gj_per_m_sq','energy_eui_natural_gas_gj_per_m_sq','net_site_eui_gj_per_m_sq'])):
            output_drop_list.append(col)
    btap_df = btap_df.drop(output_drop_list,axis=1)
    btap_df = clean_data(btap_df)
    #btap_df['Total Energy'] = btap_df[['net_site_eui_gj_per_m_sq','energy_eui_natural_gas_gj_per_m_sq']].sum(axis=1)
    btap_df['Total Energy'] = btap_df[['net_site_eui_gj_per_m_sq']].sum(axis=1)
    drop_list=['energy_eui_additional_fuel_gj_per_m_sq','energy_eui_electricity_gj_per_m_sq','energy_eui_natural_gas_gj_per_m_sq','net_site_eui_gj_per_m_sq']
    btap_df = btap_df.drop(drop_list,axis=1)
    
    return btap_df,floor_sq

def read_weather(tenant,bucket,path):
    weather_df = access_minio(tenant=tenant,
                               bucket=bucket,
                               operation= 'read',
                               path=path,
                               data='')
#     weather_data ='./input/montreal_epw.csv'
#     weather_df = pd.read_csv(weather_data, skiprows=0, low_memory=False)
#     dropping columns not used by Enegyplus calculation
    weather_drop_list= ['Minute','Uncertainty Flags','Extraterrestrial Horizontal Radiation', 'Extraterrestrial Direct Normal Radiation','Global Horizontal Radiation','Global Horizontal Illuminance','Direct Normal Illuminance', 'Diffuse Horizontal Illuminance', 'Zenith Luminance', 'Total Sky Cover', 'Opaque Sky Cover', 'Visibility', 'Ceiling Height', 'Precipitable Water', 'Aerosol Optical Depth', 'Days Since Last Snowfall', 'Albedo'
           , 'Liquid Precipitation Quantity','Present Weather Codes' ]
    weather_df = weather_df.drop(weather_drop_list,axis=1)
    weather_df = clean_data(weather_df)
    weather_df["date_int"]= weather_df.apply(lambda r : datetime(int(r['Year']), int( r['Month']),int( r['Day']), int(r['Hour']-1)).strftime("%m%d"), axis =1)
    weather_df["date_int"]=weather_df["date_int"].apply(lambda r : int(r))
    weather_df=weather_df.groupby(['date_int']).agg(lambda x: x.sum())
    
    return weather_df

def read_hour_energy(tenant,bucket,path,floor_sq):
    energy_hour_df = access_minio(tenant=tenant,
                               bucket=bucket,
                               operation= 'read',
                               path=path,
                               data='')
#     energy_file = './input/total_hourly_res.csv'
#     energy_hour_df = pd.read_csv(energy_file, skiprows=0, low_memory=False)

    eletricity_hour_df = energy_hour_df[energy_hour_df['Name'] != "Electricity:Facility"].groupby(['datapoint_id']).sum()
    #gas_hour_df = energy_hour_df[energy_hour_df['Name'] == "Gas:Facility"].groupby(['datapoint_id']).sum()
    #energy_df = pd.concat([eletricity_hour_df,gas_hour_df])
    #energy_df = energy_df.groupby(['datapoint_id']).sum()
    #converting to GJ/sq_m
    #energy_df= eletricity_hour_df.agg(lambda x: x /(floor_sq*1000000) )
    energy_df= eletricity_hour_df.agg(lambda x: x /(floor_sq*1000000) )
    energy_df = energy_df.drop(['KeyValue'],axis=1)
    #energy_df = energy_df[0:10]
   
    energy_df = clean_data(energy_df)
    energy_hour_df = energy_df.reset_index()
    energy_hour_melt =energy_hour_df.melt(id_vars=['datapoint_id'],var_name='Timestamp', value_name='energy')
    energy_hour_melt["date_int"]=energy_hour_melt['Timestamp'].apply(lambda r : datetime.strptime(r, '%Y-%m-%d %H:%M'))
    
    energy_hour_melt["date_int"]=energy_hour_melt["date_int"].apply(lambda r : r.strftime("%m%d"))
    energy_hour_melt["date_int"]=energy_hour_melt["date_int"].apply(lambda r : int(r))
    energy_hour_melt=energy_hour_melt.groupby(['datapoint_id','date_int'])['energy'].agg(lambda x: x.sum()).reset_index()  
   
    return energy_hour_melt

def train_test_split(energy_daily_df,validate):
    drop_list= ['index', 'Dew Point Temperature', 'Horizontal Infrared Radiation Intensity',  ':datapoint_id','level_0', 'index','date_int',':datapoint_id','Year', 'Month', 'Day', 'Hour']
    #split to train and test datasets
    
    y = energy_daily_df[['energy','datapoint_id','Total Energy']]
    X = energy_daily_df.drop(['energy'],axis = 1)
    
#     test =  energy_daily_df[energy_daily_df['datapoint_id'] == '9ab2cbff-cfad-4155-9e60-977e371e388a']
#     print(test[['energy','Total Energy']])
#     print('^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^6')
#     print(test.groupby(['datapoint_id']).sum())
    
    gs = GroupShuffleSplit(n_splits=2, test_size=.2)
    train_ix, test_ix = next(gs.split(X, y, groups=X.datapoint_id))
    X_train = X.loc[train_ix]
    y_train = y.loc[train_ix]
    X_test = X.loc[test_ix]
    y_valid = y.loc[test_ix]
    
    energy_daily_df= energy_daily_df.drop(drop_list,axis = 1)
    X_train = X_train.drop(drop_list,axis=1)
    X_test = X_test.drop(drop_list,axis=1)
    y_train = y_train['energy']
    y_test = y_valid[['energy','datapoint_id']]

    X_train= X_train.drop(['datapoint_id','Total Energy'],axis = 1)
    X_test= X_test.drop(['datapoint_id','Total Energy'],axis = 1)
    
    
    
    if validate == 'yes' :
        X = X.drop(drop_list,axis=1)
        X = X.drop(['datapoint_id','Total Energy'],axis = 1)
        y_valid = y
        y = y[['energy','datapoint_id']]
        
        
        return X, y, y_valid
    else:
        return X_train, X_test, y_train, y_test, y_valid

def categorical_encode(x_train,x_test,x_validate):
    
    #extracting the categorical columns
    cat_cols = x_train.select_dtypes(include=['object']).columns
    other_cols = x_train.drop(columns=cat_cols).columns
    
    # Create the encoder.
    encoder = OneHotEncoder(handle_unknown="ignore")
    ct = ColumnTransformer([('ohe', OneHotEncoder(sparse=False), cat_cols)], remainder=MinMaxScaler())
    #encoded_matrix = ct.fit(x_train)
    # Apply the encoder.
    x_train_oh = ct.fit_transform(x_train)
    x_test_oh = ct.fit_transform(x_test)
    x_val_oh = ct.fit_transform(x_validate)
    encoded_cols = ct.named_transformers_.ohe.get_feature_names(cat_cols)
    all_features = np.concatenate([encoded_cols, other_cols])
    
    return x_train_oh, x_test_oh,x_val_oh, all_features



def process_data(args):
    btap_df,floor_sq = read_output(args.tenant,args.bucket,args.in_build_params)
    weather_df = read_weather(args.tenant,args.bucket,args.in_weather)
    energy_hour_df = read_hour_energy(args.tenant,args.bucket,args.in_hour,floor_sq)
    btap_df_val,floor_sq = read_output(args.tenant,args.bucket,args.in_build_params_val)
    energy_hour_df_val = read_hour_energy(args.tenant,args.bucket,args.in_hour_val,floor_sq)
    
    #btap_df, weather_df, energy_hour_df = read__files(args)
    energy_hour_merge = pd.merge(energy_hour_df, btap_df, left_on=['datapoint_id'],right_on=[':datapoint_id'],how='left').reset_index()
    energy_daily_df = pd.merge(energy_hour_merge, weather_df, on='date_int',how='left').reset_index()  
    
    energy_hour_merge_val = pd.merge(energy_hour_df_val, btap_df_val, left_on=['datapoint_id'],right_on=[':datapoint_id'],how='left').reset_index()
    energy_daily_df_val = pd.merge(energy_hour_merge_val, weather_df, on='date_int',how='left').reset_index()  
    
    X_train, X_test, y_train, y_test, y_valid = train_test_split(energy_daily_df,'no')
    X_validate, y_validate, y_complete = train_test_split(energy_daily_df_val,'yes')
    
    print('#####################################3')
    print(y_valid.groupby(['datapoint_id']).sum())
    print(y_complete.groupby(['datapoint_id']).sum())
    
    files = glob.glob('./img/*')
    for f in files:
        os.remove(f)
    
    #saving the plots
    pl.corr_plot(energy_daily_df)
    #pl.target_plot(verify_train_df['Total Energy'],verify_test_df['Total Energy'])
    
    #convertiing categorical values to numbers
    X_train_oh, X_test_oh, X_val_oh, all_features= categorical_encode(X_train,X_test,X_validate)
    
    print(y_validate.shape)
    print(y_complete.shape)
    #Creates `data` structure to save and share train and test datasets.
    data = {'features':all_features.tolist(),
            'y_train' : y_train.to_json(orient="values"),
            'X_train' : X_train_oh.tolist(),
            'X_test' : X_test_oh.tolist(),
            #'y_test' : y_test.to_json(orient="values"),
            'y_test' : y_test.values.tolist(),
            'y_valid':y_valid.values.tolist(),
            'X_validate' : X_val_oh.tolist(),
            'y_validate' : y_validate.values.tolist(),
            'y_complete':y_complete.values.tolist(),          
           }
    
    # Creates a json object based on `data`
    data_json = json.dumps(data).encode('utf-8')

#   copy data to minio
    access_minio(tenant=args.tenant,
                 bucket=args.bucket,
                 operation='copy',
                 path=args.output_path,
                 data=data_json)
#     out_file =open('../output/preprocessing_out','w')
#     json.dump(data, out_file, indent = 6) 
#     out_file.close()
    
    return 


if __name__ == '__main__':
    
    # This component does not receive any input it only outpus one artifact which is `data`. Defining and parsing the command-line arguments
    parser = argparse.ArgumentParser()
    
    # Paths must be passed in, not hardcoded
    parser.add_argument('--tenant', type=str, help='The minio tenant where the data is located in')
    parser.add_argument('--bucket', type=str, help='The minio bucket where the data is located in')
    parser.add_argument('--in_hour', type=str, help='The minio bucket where the data is located in')
    parser.add_argument('--in_build_params', type=str, help='Name of data file to be read')
    parser.add_argument('--in_weather', type=str, help='Name of weather file to be read')
    parser.add_argument('--output_path', type=str, help='Path of the local file where the output file should be written.')
    parser.add_argument('--in_hour_val', type=str, help='The minio bucket where the data is located in')
    parser.add_argument('--in_build_params_val', type=str, help='Name of data file to be read')
    args = parser.parse_args()
    
    # Creating the directory where the output file will be created (the directory may or may not exist).
    process_data(args)
    
    #to run the program use the command below
    #python3 preprocessing.py --tenant standard --bucket nrcan-btap --in_build_params input_data/output.xlsx --in_hour input_data/total_hourly_res.csv --in_weather input_data/montreal_epw.csv --output_path output_data/preprocessing_out --in_build_params_val input_data/output_validate.xlsx --in_hour_val input_data/total_hourly_res_validate.csv

#python3 preprocessing.py --tenant standard --bucket nrcan-btap --in_build_params input_data/output_2021-10-04.xlsx --in_hour input_data/total_hourly_res_2021-10-04.csv --in_weather input_data/montreal_epw.csv --output_path output_data/preprocessing_out --in_build_params_val input_data/output.xlsx --in_hour_val input_data/total_hourly_res.csv
