# Operaciones en multi hilo
from modulos.arima.gruas.general import show_results_r2, arima_forecasting, total_forecasting, show_optimizer_results
from modulos.LR.gruas.generals import make_lags, make_timeserie_arima, make_timeserie, cross_validation_ts_mape_r2, split_data_train, cross_validation_ts_mape_r2_ARIMA
from sklearn.model_selection import cross_val_score
from modulos.DirRec.gruas.optimizer import MLOptimizer
from statsmodels.tsa.arima.model import ARIMA

from sklearn.metrics import r2_score
import optuna
import threading
import time
import json
import pandas as pd
import numpy as np
import os
optuna.logging.disable_default_handler()

TEST_SIZE = 0.1
RANDOM_STATE_TEST = 0


def ARIMA_forecasting(self, ts,  ar=2, ii=1, ma=2):
    ts, scaler = make_timeserie_arima(ts.copy())

    # Test 20% del total
    mse, score_r2 = cross_validation_ts_mape_r2_ARIMA(
        ARIMA, order=(ar, ii, ma), ts=ts, test_size=TEST_SIZE)
    results = arima_forecasting(ts, ar=ar, ii=ii, ma=ma)
    return results, ts, score_r2, mse, scaler


def ARIMA_score_cv(self, ts, ar=2, ii=1, ma=2):
    ts, scaler = make_timeserie_arima(ts.copy())

    # Retiramos el test 20%
    ts_train, _ = split_data_train(ts, ts, test_size=TEST_SIZE)

    ts_train.index = pd.DatetimeIndex(ts_train.index).to_period('M')

    # Validamos con el val 20% del total*80%
    mse, score_r2 = cross_validation_ts_mape_r2_ARIMA(
        ARIMA, order=(ar, ii, ma), ts=ts_train, test_size=TEST_SIZE)
    return score_r2, mse


class ARIMAOptimizer(MLOptimizer):
    model = 'ARIMA'
    Model_forecasting_process = ARIMA_forecasting
    Model_score_cv = ARIMA_score_cv

    def optuna_optimizer(self, idArticulo):
        def objective(trial):
            r_min = 0
            r_max = 6
            ar = trial.suggest_int('ar', 2, r_max)
            ii = trial.suggest_int('ii', r_min, 3)
            ma = trial.suggest_int('ma', r_min, r_max)
            # pred = total_forecasting(self.df_time[[idArticulo]], ar, ii, ma)
            # score = r2_score(
            #     self.df_time[idArticulo], pred.apply(lambda x: round(x, 0)))
            # return score
            try:
                score, mse = self.Model_score_cv(
                    self.df_time[idArticulo], ar, ii, ma)
            except Exception as e:
                print(e)
                score = -100.0
                mse = 100.0
            return mse

        study = optuna.create_study(
            direction='minimize', sampler=optuna.samplers.TPESampler(seed=self.SEED))
        study.optimize(objective, n_trials=self.iterations)
        self.studies.append({'study': study, 'idArticulo': idArticulo})
        self.save_results(idArticulo, study)

    def save_results(self, idArticulo, study):
        results, ts, score, mse, scaler = self.Model_forecasting_process(
            self.df_time[idArticulo], **study.best_params)
        row = {'idArticulo': idArticulo, 'hyper': study.best_params,
               'r2_test': score, 'mse_test': mse, 'model': self.model_name}
        self.results = self.results.append(row, ignore_index=True)

    def print_results(self):
        opts = pd.read_csv(os.path.join(
            self.result_path(), f'{self.model_name}.csv'))
        for opt in opts.to_dict(orient='records'):
            hyper = json.loads(opt['hyper'].replace("'", '"'))
            results, ts, score, mse, scaler = self.Model_forecasting_process(
                self.df_time[opt['idArticulo']],  **hyper)
            # ts = pd.DataFrame(ts, index=ts.index)

            ts_ = scaler.inverse_transform(ts)
            ts = pd.DataFrame(ts_, index=ts.index)

            ts_pred = scaler.inverse_transform(results.fittedvalues.to_frame())
            ts_pred = pd.DataFrame(ts_pred, index=ts.index)
            rmse_scl = scaler.inverse_transform([[np.sqrt(mse)]])[0][0]
            _ = show_results_r2(ts,
                                ts_pred,
                                opt['idArticulo'], rmse_scl, score_name='RMSE')


class ARIMAOptimizer_depricated:
    def __init__(self, df_time, iterations, data_path, model='ARIMA', ):
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
            r_min = 0
            r_max = 6
            ar = trial.suggest_int('ar', r_min, r_max)
            ii = trial.suggest_int('ii', r_min, r_max)
            ma = trial.suggest_int('ma', r_min, r_max)
            pred = total_forecasting(self.df_time[[idArticulo]], ar, ii, ma)
            score = r2_score(
                self.df_time[idArticulo], pred.apply(lambda x: round(x, 0)))
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
            result = arima_forecasting(
                self.df_time[[opt['idArticulo']]],  **hyper)
            r2 = show_results_r2(self.df_time, result.fittedvalues. apply(
                lambda x: round(x, 0)), opt['idArticulo'])

    def print_optimizer_results(self):
        for study in self.studies:
            data = [trial.value for trial in study['study'].trials]
            show_optimizer_results(data, study['idArticulo'])
