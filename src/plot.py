'''
Contains helper functions used to create chart. The plots created are stored in ../output directory. 

'''
import numpy as np
import pandas as pd
import seaborn as sns
import plotly
from matplotlib import pyplot as plt

def show_var(btap_data_df):
    num_vars = list(btap_data_df.select_dtypes(include=[np.number]).columns.values)
    df_ax = btap_data_df[num_vars].plot(title='numerical values',figsize=(15,8))
    plt.savefig('../output/numerical_val_plot.png')
    
    
def corr_plot(btap_data_df):
    #Using Pearson Correlation
    plt.figure(figsize=(20,20))
    cor = btap_data_df.corr()
    sns.heatmap(cor, annot=True, cmap=plt.cm.Reds)
    plt.savefig('../output/corr_plot.png')
    
def target_plot(y_test):
    plt.figure(figsize=(20,5))
    plt.plot(y_test['energy'])
    plt.ylabel("Energy")
    plt.savefig('../output/daily_energy_test.png')
    
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
    plt.savefig('../output/train_test_loss.png')
    
    plt.style.use("ggplot")
    plt.figure(2)
    plt.plot(H.history["mae"], label="train_mae")
    plt.plot(H.history["val_mae"], label="val_mae")
    plt.title("Training Loss and MAE")
    plt.xlabel("Epoch #")
    plt.ylabel("MAE")
    plt.legend()
    plt.savefig('../output/train_test_mae.png')
    
    return

def annual_plot(output_df,desc):
    plt.style.use("ggplot")
    plt.figure()
    plt.plot(output_df['Total Energy'],label='actual')
    plt.plot(output_df['y_pred_transformed'],label='pred')
    plt.title("Annual Energy " + desc)
    plt.savefig('../output/annual_energy_'+desc+'.png')
    
    return

def daily_plot(output_df,desc):
    plt.style.use("ggplot")
    plt.figure()
    plt.plot(output_df['energy'],label='actual')
    plt.plot(output_df['y_pred_transformed'],label='pred')
    plt.title("Daily Energy " + desc)
    plt.savefig('../output/daily_energy_'+desc+'.png')
    
    return 