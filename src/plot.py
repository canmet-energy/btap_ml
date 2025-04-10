'''
Contains helper functions used to create chart. The plots created are stored in ./output directory.
'''

import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib import pyplot as plt


def show_var(btap_data_df: pd.DataFrame) -> None:
    num_vars = list(btap_data_df.select_dtypes(include=[np.number]).columns.values)
    df_ax = btap_data_df[num_vars].plot(title='numerical values', figsize=(15, 8))
    plt.savefig('./output/numerical_val_plot.png')


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
    plt.savefig('./output/Total_Energy_Scatter.png')


def corr_plot(btap_data_df: pd.DataFrame) -> None:
    # Using Pearson Correlation
    plt.figure(figsize=(20, 20))
    cor = btap_data_df.corr()
    sns.heatmap(cor, annot=True, cmap=plt.cm.Reds)
    plt.savefig('./output/corr_plot.png')


def target_plot(y_train, y_test) -> None:
    plt.figure()
    plt.hist(y_train, label='train')
    plt.hist(y_test, label='test')
    plt.savefig('./output/daily_energy_train.png')


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

plt.style.use('seaborn-darkgrid')
fig, ax = plt.subplots(2, 1, figsize=(8, 10))

def mlp_learning_curve_plot(idx: int, process_type, H) -> None:
    """
    Visualize MLP learning curve.

    Args:
        idx: Identify the iteration number.
        H: Training and validation results stored in a dictionary.

    Returns:
        None.
    """
    if idx == 0:
        ax[0].set_title('MLP', fontsize=16)
        ax[0].plot(H.history["loss"], label="Training dataset", color="coral")
        ax[0].plot(H.history["val_loss"], label="Validation dataset", color="darkolivegreen")

        ax[0].tick_params(axis='both', which='major', labelsize=13)
    else:
        ax[1].plot(H.history["loss"], label="Training dataset", color="coral")
        ax[1].plot(H.history["val_loss"], label="Validation dataset", color="darkolivegreen")
        ax[1].set_xlabel("Epoch #", color='black', fontsize=14)

        ax[1].tick_params(axis='both', which='major', labelsize=13)

        plt.savefig('./output/Stacked Learning curve shared plot')

    handles, labels = ax[0].get_legend_handles_labels()

    legend = fig.legend(handles, labels, loc='upper center', fontsize='x-large')
    legend.get_frame().set_facecolor('none')
    legend.get_frame().set_edgecolor('none')

    return

def xgboost_learning_curve_plot(idx: int, train_rmse, val_rmse):
    """
    Visualize XGBoost learning curve.

    Args:
        idx: Identify the iteration number.
        train_rmse: The training dataset RMSE values for each sequential tree added.
        val_rmse: The validation dataset RMSE values for each sequential tree added.

    Returns:
        None.
    """
    if idx == 0:
        plt.style.use('seaborn-darkgrid')
        global fig
        global ax

        fig, ax = plt.subplots(2, 1, figsize=(8, 10))

        ax[0].set_title("XGBoost", fontsize=16)
        ax[0].plot(train_rmse, label='Training dataset', color="coral")
        ax[0].plot(val_rmse, label='Validation dataset', color="darkolivegreen")
        ax[0].set_ylabel("Energy Use Intensity RMSE", color='black', fontsize=14)

        ax[0].tick_params(axis='both', which='major', labelsize=13)

    else:
        ax[1].plot(train_rmse, label='Training dataset', color="coral")
        ax[1].plot(val_rmse, label='Validation dataset', color="darkolivegreen")
        ax[1].set_xlabel("Number of trees", color='black', fontsize=14)
        ax[1].set_ylabel("Costing RMSE", color='black', fontsize=14)

        ax[1].tick_params(axis='both', which='major', labelsize=13)

        plt.savefig('./output/XGBoost_Learning_Curve.png')
    
    handles, labels = ax[0].get_legend_handles_labels()

    legend = fig.legend(handles, labels, loc='upper center', fontsize='x-large')
    legend.get_frame().set_facecolor('none')
    legend.get_frame().set_edgecolor('none')

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
    plt.savefig('./output/annual_energy_' + desc + '.png')

    return


def daily_plot(output_df, desc):
    plt.style.use("ggplot")
    plt.figure()
    plt.plot(output_df['energy'], label='actual')
    plt.plot(output_df['y_pred_transformed'], label='pred')
    plt.title("Daily Energy " + desc)
    plt.savefig('./output/daily_energy_' + desc + '.png')

    return
