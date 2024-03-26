from pyspark_data_preprocessing import join_dataframe
import numpy as np
from pyspark.sql import SparkSession
from pyspark.ml.classification import RandomForestClassifier
from pyspark.ml.feature import VectorAssembler
from pyspark.ml.evaluation import BinaryClassificationEvaluator
from pyspark.ml.tuning import CrossValidator, ParamGridBuilder
from pyspark.ml import Pipeline

def random_forest_model(debug):

    spark = SparkSession.builder.appName("Pyspark Random Forest Classifier").getOrCreate()
    train_df, test_df = join_dataframe()
    exclude_cols = ['TARGET', 'SK_ID_CURR', 'SK_ID_BUREAU', 'SK_ID_PREV', 'index']

    assembler = VectorAssembler(inputCols=[col for col in train_df.columns if col not in exclude_cols], outputCol='features')
    df_train_assembled = assembler.transform(train_df)
    df_test_assembled = assembler.transform(test_df)

    rfc = RandomForestClassifier(featuresCol='features',labelCol='TARGET',maxIter=10)
    param_grid = ParamGridBuilder().addGrid(rfc.maxDepth,[4,5,6]).addGrid(rfc.maxBins,[16,32,64]).build()

    evaluator = BinaryClassificationEvaluator(labelCol='TARGET')

    cv = CrossValidator(estimator=rfc,estimatorParamMaps=param_grid,evaluator=evaluator,numFolds=5,seed=210)
    cv_model = cv.fit(df_train_assembled)

    if not debug:
        predictions = cv_model.transform(df_test_assembled)
    return predictions, cv_model.bestModel

# Get hyperparameters of best model:
predictions, best_model = random_forest_model(debug=False)
param_dict = {param.name: value for param, value in zip(best_model.extractParamMap().keys(), best_model.extractParamMap().values())}
print(param_dict)

# Get predictions for test dataframe:
predictions_pandas = predictions.toPandas()
predictions_pandas.to_csv('submission_result\\pyspark_random_forest.csv')
