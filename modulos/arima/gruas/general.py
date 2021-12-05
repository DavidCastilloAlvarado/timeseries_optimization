from sklearn.impute import SimpleImputer
import pandas as pd
import numpy as np
from sklearn.metrics import r2_score
from matplotlib import pyplot as plt
from statsmodels.tsa.arima.model import ARIMA


def hello():
    print("Hello from optimizer.py")


def format_timeseries(data, index_col='Periodo', columns_id='idArticulo', value_col='ventas', fillval=0):
    """
    Pivot the table, fill missing values with fillval and return a dataframe
    with the a time serie in every column.
    """
    imp_mean = SimpleImputer(missing_values=np.nan,
                             strategy='constant', fill_value=fillval)
    df_time_pre = data.pivot_table(
        index=index_col, columns=columns_id, values=value_col, aggfunc='sum', )
    df_time = imp_mean.fit_transform(df_time_pre)
    df_time = pd.DataFrame(
        df_time, columns=df_time_pre.columns, index=df_time_pre.index)
    return df_time


def show_results_r2(data_real, forecasts, idArticulo, score=1, score_name='r2'):
    """
    Show a graph with the results of the forecasts.
    also with the r2 score.
    return r2 score
    """
    # r2 = r2_score(data_real, forecasts)
    plt.figure(1, figsize=(12, 5))
    plt.rcParams.update({'font.size': 14})
    _ = plt.plot(data_real, color='blue',
                 label="real",
                 linewidth=1)
    _ = plt.plot(forecasts, color='red',
                 label="Forecast",
                 linewidth=3)
    plt.legend(loc='upper left')
    plt.title(str(idArticulo).upper() + f" {score_name}={round(score,3)}")
    plt.ylabel("Ventas UND")
    plt.xlabel("Meses")
    plt.grid(True)
    plt.show(block=False)
    # return r2


def show_optimizer_results(data, idArticulo='Place your idArticulo', color='m'):
    """
    Show a graph with the results of the forecasts.
    also with the r2 score.
    return r2 score
    """
    plt.figure(1, figsize=(12, 8))
    plt.rcParams.update({'font.size': 12})
    _ = plt.plot(data, 'o', color=color,
                 label="real",
                 linewidth=.5)
    plt.title(str(idArticulo).upper() + " - optimization")
    plt.ylabel("MSE Loss")
    plt.xlabel("# Iteration")
    plt.grid(True)
    plt.show(block=False)


def arima_forecasting(data, ar=2, ii=1, ma=2):
    """
    Forecasting using ARIMA model, 
    return model.fit()
    """
    df = data.copy()
    df.index = pd.DatetimeIndex(df.index).to_period('M')
    model = ARIMA(df, order=(ar, ii, ma))
    results = model.fit()

    return results


def total_forecasting(data, ar=2, ii=1, ma=2):
    """
    Forecasting using ARIMA model.
    Return results.fittedvalues → array of fitted values
    """
    df = data.copy()
    df.index = pd.DatetimeIndex(df.index).to_period('M')
    model = ARIMA(df, order=(ar, ii, ma))
    results = model.fit()
    return results.fittedvalues
