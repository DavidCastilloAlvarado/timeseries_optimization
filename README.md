# Time Series models Benchmark

## Introduction:

Create a base line where several machine learning models
could get over the same data, using an optimization strategy with OPTUMA optimizer.

## Models:

---

We going to work with the following machine learning models.

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

1. Select the product with more healthy data over the time line (months)
2. Preprocess the data according the model
3. Model optimization by moduls:

   1. Init target model
   2. Store hyperparámeters
   3. Optimize by models (referencia: R2 score)
   4. Optimization Graph
   5. Optimization and training

4. comparison chart over R2 score by model and product
5. comparison chart

   1. One step ahead

      a. Product vs R2 - model

   2. Two steps ahead

      a. Product vs R2 - model
