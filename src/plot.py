import numpy as np
import pandas as pd
import seaborn as sns
import plotly
from matplotlib import pyplot as plt

def show_var(btap_data_df):
    num_vars = list(btap_data_df.select_dtypes(include=[np.number]).columns.values)
    df_ax = btap_data_df[num_vars].plot(title='numerical values',figsize=(15,8))
    plt.savefig('../output/numerical_val_plot.png')
    

def norm_res(btap_data_df):
    results_normed = (btap_data_df - np.mean(btap_data_df)) / np.std(btap_data_df)
    return results_normed

def norm_res_plot(btap_data_df):
    total_heating_use = btap_data_df["daily_energy"]
    plt.scatter(norm_res(btap_data_df[":ext_wall_cond"]), total_heating_use, label="wall cond")
    plt.scatter(norm_res(btap_data_df[":ext_roof_cond"]), total_heating_use, label="roof cond")
    plt.scatter(norm_res(btap_data_df[":fdwr_set"]), total_heating_use,label="w2w ratio")
    plt.legend()
    plt.savefig('../output/Total_Energy_Scatter.png')
    
    
def corr_plot(btap_data_df):
    #Using Pearson Correlation
    plt.figure(figsize=(20,20))
    cor = btap_data_df.corr()
    sns.heatmap(cor, annot=True, cmap=plt.cm.Reds)
    plt.savefig('../output/corr_plot.png')
    
def target_plot(y_train,y_test):
    plt.figure()
    plt.hist(y_train,label='train')
    plt.hist(y_test,label='test')
    plt.savefig('../output/daily_energy_train.png')