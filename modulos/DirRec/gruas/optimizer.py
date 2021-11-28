

import os
import pandas as pd
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
from sklearn.model_selection import train_test_split
from xgboost import XGBRegressor
optuna.logging.disable_default_handler()
TEST_SIZE = 0.2
RANDOM_STATE_TEST = 0


def DT_forecasting(self, ts, n_lags, max_depth, random_state, ccp_alpha):
    X, y, scaler = make_timeserie(ts.copy(), n_lags)

    model = DecisionTreeRegressor(
        max_depth=max_depth, random_state=random_state, ccp_alpha=ccp_alpha)
    # Test 20% del total
    mape, score_r2 = cross_validation_ts_mape_r2(
        model, X, y, test_size=TEST_SIZE)
    return model, X, y, score_r2, mape, scaler


def DT_score_cv(self, ts, n_lags, max_depth, random_state, ccp_alpha, cv=6):
    X, y, scaler = make_timeserie(ts.copy(), n_lags)

    # Retiramos el test 20%
    X_train, y_train = split_data_train(X, y, test_size=TEST_SIZE)

    model = DecisionTreeRegressor(
        max_depth=max_depth, random_state=random_state, ccp_alpha=ccp_alpha)

    # Validamos con el val 20% del total*80%
    mape, score_r2 = cross_validation_ts_mape_r2(
        model, X_train, y_train, test_size=TEST_SIZE)

    return score_r2, mape

######### XGBoost #############


def XGB_forecasting(self, ts, n_lags, max_depth, random_state, gamma, n_estimators):
    X, y, scaler = make_timeserie(ts.copy(), n_lags)

    model = XGBRegressor(
        max_depth=max_depth, random_state=random_state, gamma=gamma, n_estimators=n_estimators)
    # Test 20% del total
    mape, score_r2 = cross_validation_ts_mape_r2(
        model, X, y, test_size=TEST_SIZE)
    return model, X, y, score_r2, mape, scaler


def XGB_score_cv(sefl, ts, n_lags, max_depth, random_state, gamma, n_estimators, cv=6):
    X, y, scaler = make_timeserie(ts.copy(), n_lags)

    # Retiramos el test 20%
    X_train, y_train = split_data_train(X, y, test_size=TEST_SIZE)

    model = XGBRegressor(
        max_depth=max_depth, random_state=random_state, gamma=gamma, n_estimators=n_estimators)

    # Validamos con el val 20% del total*80%
    mape, score_r2 = cross_validation_ts_mape_r2(
        model, X_train, y_train, test_size=TEST_SIZE)

    return score_r2, mape

## Desicion Tree Optimizer ######


# general Model
class MLOptimizer:
    def __init__(self, df_time, iterations, data_path, model='model', subpath=None):
        self.df_time = df_time
        self.results = pd.DataFrame()
        self.iterations = iterations
        self.SEED = 5050
        self.idArticulos = df_time.columns.tolist()
        self.model_name = model
        self.DATA_PATH = data_path
        self.studies = []
        self.subpath = subpath
        # Model_forecasting_process = <functions>
        # Model_score_cv = <functions>

        # lock = threading.Lock()
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

    def save_results(self, idArticulo, study):
        model, X, y, score, rmse, scaler = self.Model_forecasting_process(
            self.df_time[idArticulo], **study.best_params)
        row = {'idArticulo': idArticulo, 'hyper': study.best_params,
               'r2_test': score, 'mape_test': rmse, 'model': self.model_name}
        self.results = self.results.append(row, ignore_index=True)

    def print_results(self):
        opts = pd.read_csv(os.path.join(
            self.result_path(), f'{self.model_name}.csv'))
        for opt in opts.to_dict(orient='records'):
            hyper = json.loads(opt['hyper'].replace("'", '"'))
            model, X, y, score, mape, scaler = self.Model_forecasting_process(
                self.df_time[opt['idArticulo']],  **hyper)
            y_fit = pd.DataFrame(model.predict(
                X), index=X.index, columns=y.columns)
            _ = show_results_r2(scaler.inverse_transform(y),
                                scaler.inverse_transform(y_fit),
                                opt['idArticulo'], mape, score_name='MAPE')

    def print_optimizer_results(self):
        for study in self.studies:
            data = [trial.value for trial in study['study'].trials]
            show_optimizer_results(data, study['idArticulo'])

# Desition Trees


class DTOptimizer(MLOptimizer):
    model = 'DT'
    Model_forecasting_process = DT_forecasting
    Model_score_cv = DT_score_cv

    def optuna_optimizer(self, idArticulo):
        def objective(trial):
            r_min = 2
            r_max = 6
            n_lags = trial.suggest_int('n_lags', r_min, r_max)
            max_depth = trial.suggest_int('max_depth', r_min, r_max)
            random_state = trial.suggest_int('random_state', 100, 3000)
            ccp_alpha = trial.suggest_uniform('ccp_alpha', 0.01, .1)
            score, mape = self.Model_score_cv(
                self.df_time[idArticulo], n_lags, max_depth, random_state, ccp_alpha)
            return mape

        study = optuna.create_study(
            direction='minimize', sampler=optuna.samplers.TPESampler(seed=self.SEED))
        study.optimize(objective, n_trials=self.iterations,)
        self.studies.append({'study': study, 'idArticulo': idArticulo})
        self.save_results(idArticulo, study)


## XGBoost Optimizer ######

class XGBOptimizer(MLOptimizer):
    model = 'xgb'
    Model_forecasting_process = XGB_forecasting
    Model_score_cv = XGB_score_cv

    def optuna_optimizer(self, idArticulo):
        def objective(trial):
            r_min = 2
            r_max = 6
            n_lags = trial.suggest_int('n_lags', r_min, r_max)
            max_depth = trial.suggest_int('max_depth', r_min, r_max)
            random_state = trial.suggest_int('random_state', 100, 3000)
            gamma = trial.suggest_uniform('gamma', 0.0, 1.0)
            n_estimators = trial.suggest_int('n_estimators', 1, 10)
            score, mape = self.Model_score_cv(self.df_time[idArticulo],
                                              n_lags=n_lags,
                                              max_depth=max_depth,
                                              random_state=random_state,
                                              gamma=gamma,
                                              n_estimators=n_estimators)
            return mape

        study = optuna.create_study(
            direction='minimize', sampler=optuna.samplers.TPESampler(seed=self.SEED))
        study.optimize(objective, n_trials=self.iterations)
        self.save_results(idArticulo, study)
        self.studies.append({'study': study, 'idArticulo': idArticulo})
