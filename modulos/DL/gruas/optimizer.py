from sklearn.model_selection import train_test_split
from .custom_models import ForecastinModel, cross_val_score_dl
import os
import pandas as pd
import numpy as np
import json
import time
import threading
import optuna
from sklearn.metrics import r2_score
from sklearn.metrics import mean_squared_error
from modulos.arima.gruas.general import show_results_r2, arima_forecasting, total_forecasting, show_optimizer_results
from sklearn.tree import DecisionTreeRegressor
from modulos.LR.gruas.generals import make_lags, make_timeserie, cross_validation_ts_mape_r2, split_data_train
from sklearn.model_selection import cross_val_score
optuna.logging.disable_default_handler()
TEST_SIZE = 0.1
######### XGBoost #############


def DL_forecasting(ts, n_lags, random_state, learning_rate, optimizer, n_epochs):
    X, y, scaler = make_timeserie(ts.copy(), n_lags)

    hyperparameters = dict(summary=False,
                           random_state=random_state,
                           lr=learning_rate,
                           optimizer=optimizer,
                           n_epochs=n_epochs)

    model = ForecastinModel((X.iloc[0].shape[0], 1), **hyperparameters)

    mse, r2_score = cross_val_score_dl(
        model, X, y, test_size=TEST_SIZE)

    return model, X, y, r2_score, mse, scaler


def DL_score_cv(ts, n_lags, random_state, learning_rate, optimizer, n_epochs):
    X, y, scaler = make_timeserie(ts.copy(), n_lags)
    X_train, y_train = split_data_train(X, y, test_size=TEST_SIZE)

    hyperparameters = dict(summary=False,
                           random_state=random_state,
                           lr=learning_rate,
                           optimizer=optimizer,
                           n_epochs=n_epochs)

    model = ForecastinModel((X.iloc[0].shape[0], 1), **hyperparameters)
    mse, r2_score = cross_val_score_dl(
        model, X_train, y_train, test_size=TEST_SIZE)
    return r2_score, mse


class DLOptimizer:
    def __init__(self, df_time, iterations, data_path, model='dl', subpath=None):
        self.df_time = df_time
        self.results = pd.DataFrame()
        self.iterations = iterations
        self.SEED = 5050
        self.idArticulos = df_time.columns.tolist()
        self.model_name = model
        self.DATA_PATH = data_path
        self.studies = []
        self.subpath = subpath

        # self.lock = threading.Lock()
    def result_path(self):
        if self.subpath is None:
            return os.path.join(self.DATA_PATH, 'result', self.model_name)
        else:
            return os.path.join(self.DATA_PATH, 'result', self.subpath, self.model_name,)

    def run(self, chunk_size=4):

        for i in range(0, len(self.idArticulos), chunk_size):
            event_idarticulos = self.idArticulos[i:i + chunk_size]
            self.worker(event_idarticulos)

        self.results.to_csv(os.path.join(
            self.result_path(), f'{self.model_name}.csv'), index=False)

    def worker(self, event_idarticulos):
        the_threads = []
        for idArticulo in event_idarticulos:
            x = threading.Thread(
                target=self.optuna_optimizer, args=(idArticulo,))
            the_threads.append(x)
            x.start()
        for thread_ in the_threads:
            thread_.join()

    def optuna_optimizer(self, idArticulo):
        def objective(trial):
            r_min = 2
            r_max = 12
            n_lags = trial.suggest_int('n_lags', r_min, r_max)
            random_state = trial.suggest_int('random_state', 100, 3000)
            opt = trial.suggest_categorical(
                "optimizer", ["Adam", "SGD", "RMSprop"])
            learning_rate = trial.suggest_float(
                'learning_rate', 1e-5, 11e-5, step=1e-5)
            n_epochs = trial.suggest_int('n_epochs', 100, 200, step=50)
            # pred = total_forecasting_DT(
            #     self.df_time[idArticulo], n_lags, max_depth, random_state)
            # score = r2_score(
            #     self.df_time[[idArticulo]], pred.apply(lambda x: round(x, 0)))
            r2, mse = DL_score_cv(self.df_time[idArticulo],
                                  n_lags=n_lags,
                                  random_state=random_state,
                                  n_epochs=n_epochs,
                                  optimizer=opt,
                                  learning_rate=learning_rate)
            # mse = mean_squared_error(
            #     self.df_time[[idArticulo]], pred.apply(lambda x: round(x, 0)))
            return mse

        study = optuna.create_study(
            direction='minimize', sampler=optuna.samplers.TPESampler(seed=self.SEED))
        study.optimize(objective, n_trials=self.iterations)
        self.studies.append({'study': study, 'idArticulo': idArticulo})
        self.save_results(idArticulo, study)

    def save_results(self, idArticulo, study):
        model, X, y, score, mse, scaler = DL_forecasting(
            self.df_time[idArticulo], **study.best_params)
        row = {'idArticulo': idArticulo, 'hyper': study.best_params,
               'r2_test': score, 'mse_test': mse, 'model': self.model_name}
        self.results = self.results.append(row, ignore_index=True)

    def print_results(self):
        opts = pd.read_csv(os.path.join(
            self.result_path(), f'{self.model_name}.csv'))
        for opt in opts.to_dict(orient='records'):
            hyper = json.loads(opt['hyper'].replace("'", '"'))
            model, X, y, score, mse, scaler = DL_forecasting(
                self.df_time[opt['idArticulo']],
                **hyper)
            y_fit = pd.DataFrame(model.model.predict(
                X), index=X.index, columns=y.columns)
            rmse_scl = scaler.inverse_transform([[np.sqrt(mse)]])[0][0]
            _ = show_results_r2(scaler.inverse_transform(y),
                                scaler.inverse_transform(y_fit),
                                opt['idArticulo'], rmse_scl, score_name='RMSE')

    def print_optimizer_results(self):
        for study in self.studies:
            data = [trial.value for trial in study['study'].trials]
            show_optimizer_results(data, study['idArticulo'])
