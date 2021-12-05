from sklearn.impute import SimpleImputer
import pandas as pd
import numpy as np

from matplotlib import pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import RobustScaler
from sklearn.metrics import mean_absolute_percentage_error
from sklearn.metrics import r2_score
from sklearn.metrics import mean_squared_error
from statsmodels.tsa.arima.model import ARIMA


def make_timeserie_arima(ts):
    scaler = StandardScaler()
    ts = np.trim_zeros(ts)
    ts = pd.Series(scaler.fit_transform(
        pd.DataFrame(ts)).flatten(), index=ts.index)
    y = pd.DataFrame({
        'y': ts,
    })
    return y, scaler


def make_timeserie(ts, lags, lead_time=1):
    scaler = StandardScaler()
    ts = np.trim_zeros(ts)

    ts = pd.Series(scaler.fit_transform(
        pd.DataFrame(ts)).flatten(), index=ts.index)

    X = make_lags(ts, lags, lead_time)
    y = pd.DataFrame({
        'y': ts,
    })

    y, X = y.align(X, join='inner', axis=0)
    return X, y, scaler


def make_lags(ts, lags, lead_time=1):
    return pd.concat(
        {
            f'y_lag_{i}': ts.shift(i)
            for i in range(lead_time, lags + lead_time)
        },
        axis=1).dropna()


def split_data_train(X, y, test_size=.2):
    X_train = X.iloc[:-int(test_size*len(y))]
    y_train = y.iloc[:-int(test_size*len(y))]
    return X_train, y_train


def cross_validation_ts_mape_r2(model, X, y, test_size=.2):
    forecasts = []
    for i_last in range(-int(test_size*len(y)), 0):
        X_train = X.iloc[:i_last]
        y_train = y.iloc[:i_last]
        X_test = pd.DataFrame(X.iloc[i_last]).T
        y_test = pd.DataFrame(y.iloc[i_last]).T
        model.fit(X_train, y_train)
        pred = model.predict(X_test).flatten()
        forecasts.append(pred)

    mse = mean_squared_error(y.iloc[-int(test_size*len(y)):], forecasts)
    r2_ = r2_score(y.iloc[-int(test_size*len(y)):], forecasts)
    return mse, r2_


def cross_validation_ts_mape_r2_ARIMA(model, order, ts, test_size=.2):
    forecasts = []
    # ts = pd.DataFrame(ts)
    for i_last in range(-int(test_size*len(ts)), 0):
        ts_train = ts.iloc[:i_last]
        ts_test = pd.DataFrame(ts.iloc[i_last]).T
        model = ARIMA(ts_train, order=order)
        predict = model.fit()
        pred = predict.forecast()
        forecasts.append(pred)
    mse = mean_squared_error(
        ts.iloc[-int(test_size*len(ts)):], forecasts)  # [val +1 for val in forecasts])
    r2_ = r2_score(ts.iloc[-int(test_size*len(ts)):], forecasts)
    return mse, r2_
