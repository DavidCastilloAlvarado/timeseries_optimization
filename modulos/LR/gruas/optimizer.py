

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
from modulos.LR.gruas.generals import make_lags
from sklearn.model_selection import cross_val_score


def total_forecasting_LR(ts, n_lags):
    X = make_lags(ts.copy(), n_lags)
    y = pd.DataFrame({
        'y': ts,
    })
    y, X = y.align(X, join='inner', axis=0)
    model = LinearRegression()
    model.fit(X, y)
    y_fit = pd.DataFrame(model.predict(X), index=X.index, columns=y.columns)
    return y_fit


def LR_forecasting(ts, n_lags):
    X = make_lags(ts.copy(), n_lags)
    y = pd.DataFrame({
        'y': ts,
    })
    y, X = y.align(X, join='inner', axis=0)
    model = LinearRegression()
    model.fit(X, y)
    return model, X, y


def LR_score_cv(ts, n_lags, cv=10):
    X = make_lags(ts.copy(), n_lags)
    y = pd.DataFrame({
        'y': ts,
    })
    y, X = y.align(X, join='inner', axis=0)
    model = LinearRegression()

    scores = cross_val_score(model, X, y, cv=cv, scoring='r2')
    return scores.mean()


# Operaciones en multi hilo
optuna.logging.disable_default_handler()


class LROptimizer:
    def __init__(self, df_time, iterations, data_path, model='LR', ):
        self.df_time = df_time
        self.results = pd.DataFrame()
        self.iterations = iterations
        self.SEED = 5050
        self.idArticulos = df_time.columns.tolist()
        self.model_name = model
        self.DATA_PATH = data_path
        self.studies = []

        # self.lock = threading.Lock()

    def run(self, chunk_size=4):

        for i in range(0, len(self.idArticulos), chunk_size):
            event_idarticulos = self.idArticulos[i:i + chunk_size]
            self.worker(event_idarticulos)

        self.results.to_csv(os.path.join(
            self.DATA_PATH, 'result', self.model_name, f'{self.model_name}.csv'), index=False)

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
            r_min = 1
            r_max = 6
            n_lags = trial.suggest_int('n_lags', r_min, r_max)
            # pred = total_forecasting_LR(self.df_time[idArticulo], n_lags)
            # score = r2_score(
            #     self.df_time[[idArticulo]], pred.apply(lambda x: round(x, 0)))
            score = LR_score_cv(self.df_time[idArticulo], n_lags)
            # mse = mean_squared_error(
            #     self.df_time[[idArticulo]], pred.apply(lambda x: round(x, 0)))
            return score

        study = optuna.create_study(
            direction='maximize', sampler=optuna.samplers.TPESampler(seed=self.SEED))
        study.optimize(objective, n_trials=self.iterations)
        self.save_results(idArticulo, study)
        self.studies.append({'study': study, 'idArticulo': idArticulo})

    def save_results(self, idArticulo, study):
        row = {'idArticulo': idArticulo, 'hyper': study.best_params,
               'r2': study.best_value, 'model': self.model_name}
        self.results = self.results.append(row, ignore_index=True)

    def print_results(self):
        opts = pd.read_csv(os.path.join(self.DATA_PATH, 'result',
                           self.model_name, f'{self.model_name}.csv'))
        for opt in opts.to_dict(orient='records'):
            hyper = json.loads(opt['hyper'].replace("'", '"'))
            model, X, y = LR_forecasting(
                self.df_time[opt['idArticulo']],  **hyper)
            y_fit = pd.DataFrame(model.predict(
                X), index=X.index, columns=y.columns)
            r2 = show_results_r2(self.df_time, y_fit. apply(
                lambda x: round(x, 0)), opt['idArticulo'])

    def print_optimizer_results(self):
        for study in self.studies:
            data = [trial.value for trial in study['study'].trials]
            show_optimizer_results(data, study['idArticulo'])
