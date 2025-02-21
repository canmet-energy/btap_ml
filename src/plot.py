'''
Contains helper functions used to create chart. The plots created are stored in ../output directory.
'''

import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib import pyplot as plt


def show_var(btap_data_df: pd.DataFrame) -> None:
    num_vars = list(btap_data_df.select_dtypes(include=[np.number]).columns.values)
    df_ax = btap_data_df[num_vars].plot(title='numerical values', figsize=(15, 8))
    plt.savefig('../output/numerical_val_plot.png')


'''
def corr_plot(btap_data_df):
    # Using Pearson Correlation
    plt.figure(figsize=(20, 20))
    cor = btap_data_df.corr()
    sns.heatmap(cor, annot=True, cmap=plt.cm.Reds)
    plt.savefig('../output/corr_plot.png')


def target_plot(y_test):
    plt.figure(figsize=(20, 5))
    plt.plot(y_test['energy'])
    plt.ylabel("Energy")
    plt.savefig('../output/daily_energy_test.png')
'''

def norm_res(btap_data_df: pd.DataFrame):
    results_normed = (btap_data_df - np.mean(btap_data_df)) / np.std(btap_data_df)
    return results_normed


def norm_res_plot(btap_data_df: pd.DataFrame) -> None:
    total_heating_use = btap_data_df["daily_energy"]
    plt.scatter(norm_res(btap_data_df[":ext_wall_cond"]), total_heating_use, label="wall cond")
    plt.scatter(norm_res(btap_data_df[":ext_roof_cond"]), total_heating_use, label="roof cond")
    plt.scatter(norm_res(btap_data_df[":fdwr_set"]), total_heating_use, label="w2w ratio")
    plt.legend()
    plt.savefig('../output/Total_Energy_Scatter.png')


def corr_plot(btap_data_df: pd.DataFrame) -> None:
    # Using Pearson Correlation
    plt.figure(figsize=(20, 20))
    cor = btap_data_df.corr()
    sns.heatmap(cor, annot=True, cmap=plt.cm.Reds)
    plt.savefig('../output/corr_plot.png')


def target_plot(y_train, y_test) -> None:
    plt.figure()
    plt.hist(y_train, label='train')
    plt.hist(y_test, label='test')
    plt.savefig('../output/daily_energy_train.png')


def plot_metric(df: pd.DataFrame) -> None:
    plt.figure()
    plt.plot(df['loss'], label='mae')
    plt.savefig('./output/mae.png')
    plt.ylabel('Metric')
    plt.xlabel('Epoch number')
    plt.figure()
    plt.plot(df['mae'], label='mae')
    plt.savefig('./output/mse.png')
    plt.ylabel('Metric')
    plt.xlabel('Epoch number')

    return


def save_plot(H) -> None:
    # plot the training loss and accuracy
    plt.clf()
    plt.style.use("ggplot")
    plt.figure()
    plt.plot(H.history["loss"], label="train")
    plt.plot(H.history["val_loss"], label="validation")
    plt.xlabel("Epoch #", color='black')
    plt.ylabel("RMSE", color='black')
    plt.legend(fontsize="x-large")
    plt.savefig('./output/train_test_loss.png')

    plt.clf()
    plt.style.use("ggplot")
    plt.figure(2)
    plt.plot(H.history["mae"], label="train")
    plt.plot(H.history["val_mae"], label="validation")
    plt.xlabel("Epoch #", color='black')
    plt.ylabel("MAE", color='black')
    plt.legend(fontsize="x-large")
    plt.savefig('./output/train_test_mae.png')

    plt.clf()
    plt.style.use("ggplot")
    plt.figure(3)
    plt.plot(H.history["mse"], label="train")
    plt.plot(H.history["val_mse"], label="validation")
    plt.xlabel("Epoch #", color='black')
    plt.ylabel("MSE", color='black')
    plt.legend(fontsize="x-large")
    plt.savefig('./output/train_test_mse.png')

    plt.clf()
    plt.style.use("ggplot")
    plt.figure(4)
    plt.plot(H.history["mape"], label="train")
    plt.plot(H.history["val_mape"], label="validation")
    plt.xlabel("Epoch #", color='black')
    plt.ylabel("MAPE", color='black')
    plt.legend(fontsize="x-large")
    plt.savefig('./output/train_test_mape.png')

    return


def annual_plot(output_df: pd.DataFrame, desc: str) -> None:
    plt.style.use("ggplot")
    plt.figure()
    plt.plot(output_df['Total Energy'], label='actual')
    plt.plot(output_df['y_pred_transformed'], label='pred')
    plt.title("Annual Energy " + desc)
    plt.savefig('../output/annual_energy_' + desc + '.png')

    return


def daily_plot(output_df, desc):
    plt.style.use("ggplot")
    plt.figure()
    plt.plot(output_df['energy'], label='actual')
    plt.plot(output_df['y_pred_transformed'], label='pred')
    plt.title("Daily Energy " + desc)
    plt.savefig('../output/daily_energy_' + desc + '.png')

    return
