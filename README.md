This is a project focusing on predicting customer's repayment ablilities as a part of Kaggle competition.
Data for this project can be downloaded on this website: https://www.kaggle.com/competitions/home-credit-default-risk/data

I used both Python and Pyspark for this project.

For Python, data was processed and OPTUNA library is used to tune various hyperparameters for machine learning algorithms, including LightGBM, XGBoost and CatBoost. The models were then trained with selected hyperparameters and their performance was evaluated by using ROC AUC score between the predicted probability and the observed target. The achieved public scores were as follows: LightGBM: 0.79117, CatBoost: 0.78779, and XGBoost: 0.78172. 

For Pyspark, data was processed and hyperparameters were selected from search space using ParamGridBuiler and CrossValidator for the Random Forest Classifer and GBT Classifier. Then the model with the best performance was trained with test dataframe to predict the target variable.
