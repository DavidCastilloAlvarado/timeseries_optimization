

import os
import pandas as pd
import json
import time
import threading
import optuna
from sklearn.metrics import r2_score
from sklearn.metrics import mean_squared_error
from modulos.arima.gruas.general import show_results_r2, arima_forecasting, total_forecasting, show_optimizer_results
from sklearn.linear_model import LinearRegression
from modulos.LR.gruas.generals import make_lags, make_timeserie, cross_validation_ts_mape_r2, split_data_train
from sklearn.model_selection import cross_val_score
from modulos.DirRec.gruas.optimizer import MLOptimizer
from sklearn.model_selection import train_test_split
from sklearn.metrics import make_scorer
from sklearn.metrics import mean_absolute_percentage_error
TEST_SIZE = 0.2
RANDOM_STATE_TEST = 0


def LR_forecasting(self, ts, n_lags):
    X, y, scaler = make_timeserie(ts.copy(), n_lags)

    model = LinearRegression()
    # Test 20% del total
    mape, score_r2 = cross_validation_ts_mape_r2(
        model, X, y, test_size=TEST_SIZE)
    return model, X, y, score_r2, mape, scaler


def LR_score_cv(self, ts, n_lags, cv=6):
    X, y, scaler = make_timeserie(ts.copy(), n_lags)

    # Retiramos el test 20%
    X_train, y_train = split_data_train(X, y, test_size=TEST_SIZE)

    model = LinearRegression()

    # Validamos con el val 20% del total*80%
    mape, score_r2 = cross_validation_ts_mape_r2(
        model, X_train, y_train, test_size=TEST_SIZE)

    return score_r2, mape


# Operaciones en multi hilo
optuna.logging.disable_default_handler()


class LROptimizer(MLOptimizer):
    model = 'LR'
    Model_forecasting_process = LR_forecasting
    Model_score_cv = LR_score_cv

    def optuna_optimizer(self, idArticulo):
        def objective(trial):
            r_min = 2
            r_max = 12
            n_lags = trial.suggest_int('n_lags', r_min, r_max)
            score, mape = self.Model_score_cv(self.df_time[idArticulo], n_lags)
            return mape

        study = optuna.create_study(
            direction='minimize', sampler=optuna.samplers.TPESampler(seed=self.SEED))
        study.optimize(objective, n_trials=self.iterations)
        self.studies.append({'study': study, 'idArticulo': idArticulo})
        self.save_results(idArticulo, study)
