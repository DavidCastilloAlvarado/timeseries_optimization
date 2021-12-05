# Time Series models Benchmark

## Introducción:

Crear una línea base comparativa de varios modelos de machine learing sobre una misma base de datos, usando una estrategia de optimización por cada producto y modelo mediante la herramienta OPTUNA.

## Modelos:

---

Trabajaremos con los diguientes modelos de machine learning

1. ARIMA
2. Regresión lineal
3. Recursive strategy (1 steps ahead)

   a. Decision trees
   b. XGBOOST

4. DirRec strategy (3 steps ahead)

   a. Decision trees
   b.XGBOOST

5. Deep learning

   a.Modelo de redes neuronales recurrentes

## Steps:

---

1. Seleccionar los productos con los datos más saludables y completos sobre la línea de tiempo (months)
2. Preprocesamiento de los datos

   1. Eliminar los valores ceros en los extremos de la serie de tiempo.
   2. Estandarizar los datos de cada serie de tiempo usando x´ = (x - u)/s
   3. Generar los N-pasos hacia el pasado para alimentar a los modelos de predicción

3. Model optimization by moduls:

   1. Inicializar el modelo objetivo
   2. Optimizar el modelo (referencia: MSE loss)
   3. Validación cruzada de los datos (**entrenamiento** + **validación**)
   4. Comparativo solo con los datos de **testeo**
   5. Almacenar los hiperparámetros y las metricas
   6. Imprimir gráficas de valores reales y de entrenamiento
   7. Imprimir tabla de resultados

4. Comparación por mapa de calor según el MSE estandarizado por modelo y producto.
5. Gráficas de Rendimiento

   1. Un paso hacia adelante

      a. Producto vs MSE - Modelo

   2. Autocorrelación con gráficas aCF

      b. ACF por producto

# Referencias:

1. [Funcion de perdida MSE](https://scikit-learn.org/stable/modules/model_evaluation.html#:~:text=3.3.4.4.%20Mean%20squared%20error)
2. [score R2](https://scikit-learn.org/stable/modules/model_evaluation.html#:~:text=3.3.4.8.%20R%C2%B2%20score%2C%20the%20coefficient%20of%20determination)
3. [Optuna - ejemplo](https://optuna.readthedocs.io/en/stable/reference/generated/optuna.trial.Trial.html#optuna.trial.Trial.suggest_int)
4. [Redes Neuronales LSTM - tensorflow](https://www.tensorflow.org/api_docs/python/tf/keras/layers/LSTM)
5. [Series de tiempo - entrenamiento y testeo](https://otexts.com/fpp2/accuracy.html)
6. [White noise](https://otexts.com/fpp2/wn.html)
7. [Autocorrelación](https://otexts.com/fpp2/autocorrelation.html)
8. [Ljung–Box test](https://en.wikipedia.org/wiki/Ljung%E2%80%93Box_test)
