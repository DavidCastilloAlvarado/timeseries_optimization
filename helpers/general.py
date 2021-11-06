import re
import plotly.express as px


def hellomodule():
    print("Hello from module")


def homogenizar_str(text):
    text = text.lower()
    text = str(text).rstrip().lstrip()
    text = text.replace(" ", "")
    pattern = r'(?![a-zA-Z0-9_ ].).'
    text = re.sub(pattern, '', text)
    return text


def print_valores_en_fecha(data, x_label="Periodo", y_label="MoImporte", leyenda="idArticulo"):
    data = data.sort_values(by=x_label)
    snplot = sns.catplot(x=x_label, y=y_label, hue=leyenda, kind="point",
                         data=data[[x_label, y_label, leyenda]], height=5,
                         aspect=2.5, title="{} VS {}".format(y_label, x_label), palette=sns.color_palette("colorblind"))
    for ax in snplot.axes.flat[:2]:
        ax.tick_params(axis='x', labelrotation=90)


def print_linea_de_tiempo_producto(data, columna, height=2000):
    # df = df_clean#px.data.stocks(indexed=True)-1
    fig = px.line(data, facet_col=columna, facet_col_wrap=3,
                  height=height, facet_row_spacing=0.03)
    fig.show()


def generar_time_series(data, steps=3, dropna=False):
    data = data.copy()
    delays = [data.shift(t_delay) for t_delay in range(1, steps+1)]
    for ti, delayTable in enumerate(delays, 1):
        data = data.join(delayTable, rsuffix="_"+str(ti))

    if dropna:
        data = data.dropna()
    return data


def cv_score_model(data, producto, modelo, scoring='r2', cv=10):
    data = data.filter(regex=producto).reset_index()
    data['mes'] = data['Periodo'].apply(lambda x: x.month)
    X = data.drop(columns=['Periodo', producto])
    Y = data[producto]
    score = cross_val_score(modelo, X, Y, cv=10, scoring='r2').mean()
    return score


def print_forecasting_results(data, ARIMAfit, fecha_inicio_prediccion, title, producto, ahead=True, dynamic=True):
    def calcular_r2(dato_r, pred, producto, fecha_inicio_prediccion):
        init_date = datetime.datetime.strptime(
            fecha_inicio_prediccion, '%Y-%m-%d').date()
        dato_r2 = dato_r.reset_index().set_index('Periodo').loc[init_date:, :]
        dato_r2 = pd.DataFrame(pred).join(dato_r2)
        print(dato_r2)
        r2val = r2_score(dato_r2[producto], dato_r2['predicted_mean'])
        return r2val
    # One Step Ahead Prediction
    predict = ARIMAfit.get_prediction()
    predict_ci = predict.conf_int()
    # Dynamic predictions
    predict_dy = ARIMAfit.get_prediction(dynamic=fecha_inicio_prediccion)
    predict_dy_ci = predict_dy.conf_int()

    # Graph
    fig, ax = plt.subplots(figsize=(15, 4))
    ax.set(title=title, xlabel='Date', ylabel='Valores escalados')

    # Plot data points
    data.plot(ax=ax, style='o', label='Valores reales')

    # Plot predictions

    # Plot 'One-step-ahead forecast'
    if ahead:
        tabla_pred = predict.predicted_mean.loc[fecha_inicio_prediccion:]
        tabla_pred.plot(ax=ax, style='r--', label='One-step-ahead forecast')
        ci = predict_ci.loc[fecha_inicio_prediccion:]
        ax.fill_between(ci.index, ci.iloc[:, 0],
                        ci.iloc[:, 1], color='r', alpha=0.1)

    r2val = calcular_r2(data, tabla_pred, producto, fecha_inicio_prediccion)
    print(f" {producto} - El valor del r2 es : {r2val} ")

    # Plot 'Dynamic forecast'
    if dynamic:
        predict_dy.predicted_mean.loc[fecha_inicio_prediccion:].plot(
            ax=ax, style='g', label='Dynamic forecast')
        ci = predict_dy_ci.loc[fecha_inicio_prediccion:]
        ax.fill_between(ci.index, ci.iloc[:, 0],
                        ci.iloc[:, 1], color='g', alpha=0.1)

    legend = ax.legend(loc='upper left')


def test_stationarity(timeseries, producto, window=6, ):
    """('Results of Dickey-Fuller Test:')
    https://www.statsmodels.org/dev/generated/statsmodels.tsa.stattools.adfuller.html#statsmodels.tsa.stattools.adfuller
    """
    dftest = adfuller(timeseries.iloc[:, 0], maxlag=10, autolag='t-stat')
    dfoutput = pd.Series(dftest[0:4], index=[
                         'Test Statistic', 'p-value', '#Lags Used', 'Number of Observations Used'])
    for key, value in dftest[4].items():
        dfoutput['Critical Value (%s)' % key] = value
    dfoutput['idArticulo'] = producto
    dfoutput['H0'] = 'fail' if dftest[1] < 0.05 else 'pass'
    dfoutput['Unit root'] = 'No' if dftest[1] < 0.05 else 'Yes'
    return dfoutput
    #print (dfoutput)
    return dfoutput
