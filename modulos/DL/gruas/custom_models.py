import tensorflow as tf
from tensorflow.keras import backend as K
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Activation, Flatten, BatchNormalization, LSTM, Reshape, Input, Lambda, Bidirectional
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Conv1D, Add, Concatenate
from tensorflow.keras.utils import to_categorical
from tensorflow.keras import optimizers, Model
from tensorflow.keras.optimizers import SGD, RMSprop, Adam
from tensorflow.keras.callbacks import ModelCheckpoint, LearningRateScheduler, TensorBoard, EarlyStopping
import tensorflow.keras as keras
from sklearn.model_selection import KFold
from sklearn.metrics import r2_score
from sklearn.metrics import mean_squared_error
import uuid
import numpy as np
import pandas as pd
import os


def r2_coeff_det(y_true, y_pred):
    SS_res = K.sum(K.square(y_true-y_pred))
    SS_tot = K.sum(K.square(y_true - K.mean(y_true)))
    return (1 - SS_res/(SS_tot + K.epsilon()))

# Custom cross validation


def cross_val_score_dl(model, X, y, test_size=0.2):
    """
    Cross validation for deep learning models
    """
    # results = []
    # kf = KFold(n_splits=cv, shuffle=True, random_state=42)

    # tf.keras.backend.clear_session()
    forecasts = []
    # model.init_model()
    for i_last in range(-int(test_size*len(y)), 0):
        model.init_model()
        # if hyperparameters is not None:
        #     model = modelclass((X.iloc[0].shape[0], 1), **hyperparameters)
        # else:
        #     model = modelclass((X.iloc[0].shape[0], 1),
        #                        dbsource='xyz', summary=False)
        train_x = X.iloc[:i_last]
        train_y = y.iloc[:i_last]
        val_x = pd.DataFrame(X.iloc[i_last]).T
        val_y = pd.DataFrame(y.iloc[i_last]).T

        model.train_model(train_x, train_y, val_x, val_y,
                          verbose=0, callback=True, checkpoint=True)
        pred = model.model.predict(val_x).flatten()
        forecasts.append(pred[0])
        # tf.keras.backend.clear_session()

        # ['loss', 'mean_absolute_percentage_error', 'mean_squared_error', 'r2_coeff_det']
        # results.append(model.model.evaluate(
        #     val_x, val_y, verbose=0)[-2])  # take the mse
    # tf.keras.backend.clear_session()
    # [val +1 for val in forecasts])

    mse = mean_squared_error(y.iloc[-int(test_size*len(y)):], forecasts)
    r2_ = r2_score(y.iloc[-int(test_size*len(y)):], forecasts)

    return mse, r2_


class ForecastinModel:
    """
    loss: MAE
    """

    def __init__(self, input_shape, n_lstm_cells=1, saved_file=None, summary=True, n_epochs=10, random_state=42, lr=1e-4, optimizer='Adam'):
        self.random_state = random_state
        self.n_epochs = n_epochs
        self.lr = lr
        self.opt = optimizer
        self.input_shape = input_shape
        self.n_lstm_cells = n_lstm_cells
        self.saved_file = saved_file
        self.summary = summary
        self.init_model()

    def init_model(self):
        self.model_forecasting(self.input_shape,
                               n_lstm_cells=self.n_lstm_cells,
                               saved_file=self.saved_file,
                               summary=self.summary)

    def saved_path(self, trail_name):
        return './models/' + str(trail_name) + '.h5'

    def optimizer(self, learning_rate=1e-4):
        if self.opt == 'Adam':
            return Adam(learning_rate=learning_rate)
        elif self.opt == 'SGD':
            return SGD(learning_rate=learning_rate)
        else:
            return RMSprop(learning_rate=learning_rate)

    def model_forecasting(self, input_shape, n_lstm_cells=1, saved_file=None, summary=True):
        tf.random.set_seed(self.random_state)
        n_steps = input_shape[0]
        # construyendo modelo
        x = Input(input_shape)
        l1 = LSTM(n_lstm_cells, activation='relu')(x)
        l1 = Flatten()(l1)
        l2 = Flatten()(x)
        l2 = Dense(1, use_bias=False, activation='relu')(l2)
        out = Add()([l1, l2])
        self.model = Model(inputs=x, outputs=out)
        metrics = []
        metrics += [tf.keras.metrics.MeanAbsolutePercentageError()]
        metrics += [tf.keras.metrics.MeanAbsoluteError()]
        metrics += [r2_coeff_det]
        # MeanSquaredError  mean_absolute_percentage_error    MeanAbsoluteError
        loss = tf.keras.losses.MeanSquaredError()
        self.model.compile(optimizer=self.optimizer(learning_rate=self.lr), loss=loss,
                           metrics=metrics)
        if saved_file is not None:
            self.load_model(saved_file)
            # model.load_model(saved_file)
        if summary:
            self.model.summary()

    def load_model(self, saved_file):
        saved_file = self.saved_path(saved_file)
        self.model.load_weights(saved_file)
        # model.load_model(saved_file)

    def train_model(self, X_train, y_train, X_test, y_test, epochs=None, batch_size=32, workers=1, verbose='auto', callback=False, checkpoint=False, logs=False):
        if epochs is None:
            epochs = self.n_epochs

        logdir = "./logs"
        self.trial_name = uuid.uuid4().hex

        tboard_callback = TensorBoard(log_dir=logdir)
        saved_path = './models/'+str(self.trial_name)+'.h5'
        model_checkpoint = ModelCheckpoint(
            saved_path, monitor='val_loss', verbose=verbose, save_best_only=True, save_weights_only=True)
        earlyStopping = EarlyStopping(
            monitor='val_loss', patience=10, min_delta=0)
        custom_callback = [earlyStopping]
        if logs:
            custom_callback += [tboard_callback]
        if checkpoint:
            custom_callback += [model_checkpoint]
        tf.random.set_seed(self.random_state)
        self.history = self.model.fit(X_train, y_train,
                                      validation_data=[X_test, y_test],
                                      epochs=epochs,
                                      #   initial_epoch=self.n_epochs,
                                      callbacks=custom_callback if callback else None,
                                      workers=workers,
                                      verbose=verbose,
                                      )
        # self.n_epochs = self.n_epochs + epoch_add
        self.load_model(self.trial_name)
        os.system(f"rm {saved_path}")
