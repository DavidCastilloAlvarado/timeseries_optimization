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
import numpy as np


def r2_coeff_det(y_true, y_pred):
    SS_res = K.sum(K.square(y_true-y_pred))
    SS_tot = K.sum(K.square(y_true - K.mean(y_true)))
    return (1 - SS_res/(SS_tot + K.epsilon()))

# Custom cross validation


def cross_val_score_dl(modelclass, X, y, hyperparameters=None, cv=10, scoring='r2', epochs=100, ):
    """
    Cross validation for deep learning models
    """
    results = []
    kf = KFold(n_splits=cv, shuffle=True, random_state=42)
    for train_idx, val_idx in kf.split(X, y):
        if hyperparameters is not None:
            model = modelclass((X.iloc[0].shape[0], 1), **hyperparameters)
        else:
            model = modelclass((X.iloc[0].shape[0], 1),
                               dbsource='xyz', summary=False)
        train_x = X.iloc[train_idx]
        train_y = y.iloc[train_idx]
        val_x = X.iloc[val_idx]
        val_y = y.iloc[val_idx]
        model.train_model(train_x, train_y, val_x, val_y,
                          epochs=epochs, verbose=0, callback=True)
        # ['loss', 'mean_absolute_percentage_error', 'mean_squared_error', 'r2_coeff_det']
        results.append(model.model.evaluate(
            val_x, val_y, verbose=0)[-1])  # take the r2
    return np.array(results)


class ForecastinModel:
    """
    loss: MAE
    """

    def __init__(self, input_shape, n_lstm_cells=1, saved_file=None, summary=True, dbsource="modelxyz", random_state=42, lr=1e-4, optimizer='Adam'):
        self.random_state = random_state
        self.n_epochs = 0
        self.dbsource = dbsource
        self.lr = lr
        self.opt = optimizer
        self.model = self.model_forecasting(input_shape,
                                            n_lstm_cells=n_lstm_cells,
                                            saved_file=saved_file,
                                            summary=summary)

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
        model = Model(inputs=x, outputs=out)
        metrics = []
        metrics += [tf.keras.metrics.MeanAbsolutePercentageError()]
        metrics += [tf.keras.metrics.MeanSquaredError()]
        metrics += [r2_coeff_det]
        # MeanSquaredError  mean_absolute_percentage_error    MeanAbsoluteError
        loss = tf.keras.losses.MeanAbsoluteError()
        model.compile(optimizer=self.optimizer(learning_rate=self.lr), loss=loss,
                      metrics=metrics)
        if (saved_file):
            model.load_weights(saved_file)
            try:
                # model.load_model(saved_file)
                print("Pesos cargados")
            except:
                print("No se puede cargar los pesos")
        if summary:
            model.summary()
        return model

    def train_model(self, X_train, y_train, X_test, y_test, epochs=10, batch_size=32, workers=4, verbose='auto', callback=False, logs=False):
        dbsource = self.dbsource
        logdir = "logs/"+dbsource

        epoch_add = epochs
        tboard_callback = TensorBoard(log_dir=logdir)
        model_checkpoint = ModelCheckpoint(
            './model/'+dbsource+'/LSTMAR_grua{val_loss:.1f}.hdf5', monitor='val_loss', verbose=verbose, save_best_only=True)
        earlyStopping = EarlyStopping(
            monitor='val_r2_coeff_det', patience=2, min_delta=0)
        custom_callback = [earlyStopping]
        if logs:
            custom_callback += [tboard_callback]
        tf.random.set_seed(self.random_state)
        self.history = self.model.fit(X_train, y_train,
                                      validation_data=[X_test, y_test],
                                      epochs=self.n_epochs+epoch_add,
                                      initial_epoch=self.n_epochs,
                                      callbacks=custom_callback if callback else None,
                                      workers=workers,
                                      verbose=verbose,
                                      )
        self.n_epochs = self.n_epochs + epoch_add
