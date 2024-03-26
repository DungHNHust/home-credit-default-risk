import optuna
from optuna.integration import CatBoostPruningCallback
import catboost as cb
import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score
from data_preprocessing import application_df,bureau_and_balance_df,credit_card_balance_df,installment_payments_df,pop_cash_df,previous_application_df
import gc,re
from timeit import default_timer as timer

def prepare_data():
    df = application_df()
    bureau_and_balance = bureau_and_balance_df()
    bureau_and_balance = bureau_and_balance.rename(columns = lambda x:re.sub('[^A-Za-z0-9_]+', '', x))
    df = df.join(bureau_and_balance, on ='SK_ID_CURR', how='left')
    del bureau_and_balance
    gc.collect()

    previous_application = previous_application_df()
    previous_application = previous_application.rename(columns = lambda x:re.sub('[^A-Za-z0-9_]+', '', x))
    df = df.join(previous_application, on ='SK_ID_CURR', how = 'left')
    del previous_application
    gc.collect()

    pop_cash = pop_cash_df()
    pop_cash = pop_cash.rename(columns = lambda x:re.sub('[^A-Za-z0-9_]+', '', x))
    df = df.join(pop_cash, on ='SK_ID_CURR', how='left')
    del pop_cash
    gc.collect()

    installment_payments = installment_payments_df()
    installment_payments = installment_payments.rename(columns = lambda x:re.sub('[^A-Za-z0-9_]+', '', x))
    df = df.join(installment_payments, on='SK_ID_CURR', how='left')
    del installment_payments
    gc.collect()

    credit_card_balance = credit_card_balance_df()
    credit_card_balance = credit_card_balance.rename(columns = lambda x:re.sub('[^A-Za-z0-9_]+', '', x))
    df = df.join(credit_card_balance, on = 'SK_ID_CURR', how = 'left')
    del credit_card_balance
    gc.collect()

    train_df = df[df['TARGET'].notnull()]
    # test_df = df[df['TARGET'].isnull()]
    train_df = train_df.rename(columns = lambda x:re.sub('[^A-Za-z0-9_]+', '', x))
    # test_df = test_df.rename(columns = lambda x:re.sub('[^A-Za-z0-9_]+', '', x))
    x_train = train_df.drop(columns=['TARGET','SK_ID_CURR'])
    y_train = train_df['TARGET']
    del df, train_df
    gc.collect()
    return x_train, y_train

x_train, y_train = prepare_data()
cols_with_inf = x_train.columns[x_train.isin([np.inf, -np.inf]).any()].tolist()
x_train = x_train.drop(columns=cols_with_inf)

def objective(trial, X, y):
  param = {
        "objective": trial.suggest_categorical("objective", ["Logloss", "CrossEntropy"]),
        "colsample_bylevel": trial.suggest_float("colsample_bylevel", 0.01, 0.1, log=True),
        "depth": trial.suggest_int("depth", 1, 12),
        "boosting_type": trial.suggest_categorical("boosting_type", ["Ordered", "Plain"]),
        "bootstrap_type": trial.suggest_categorical(
            "bootstrap_type", ["Bayesian", "Bernoulli", "MVS"]
        ),
        "used_ram_limit": "10gb",
        "eval_metric": "AUC"
    }

  if param["bootstrap_type"] == "Bayesian":
        param["bagging_temperature"] = trial.suggest_float("bagging_temperature", 0, 10)
  elif param["bootstrap_type"] == "Bernoulli":
        param["subsample"] = trial.suggest_float("subsample", 0.1, 1, log=True)

  cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=210)
  cv_scores = np.empty(5)
  pruning_callback = CatBoostPruningCallback(trial,'AUC')

  for idx, (train_idx, valid_idx) in enumerate(cv.split(X,y)):
      gbm = cb.CatBoostClassifier(**param)
      X_train, X_valid = X.iloc[train_idx], X.iloc[valid_idx]
      y_train, y_valid = y.iloc[train_idx], y.iloc[valid_idx]
      gbm.fit(X_train,
        y_train,
        eval_set=[(X_valid, y_valid)],
        verbose=0,
        early_stopping_rounds=100,
        callbacks=[pruning_callback])
      pruning_callback.check_pruned()

      preds = gbm.predict_proba(X_valid)[:,1]
      cv_scores[idx] = roc_auc_score(y_valid, preds)

  return np.mean(cv_scores)

study = optuna.create_study(direction='maximize', study_name='CatBoost Classifier')
func = lambda trial: objective(trial,x_train,y_train)
study.optimize(func,n_trials=100)

print(f"\tBest value (rmse): {study.best_value:.5f}")
print(f"\tBest params:")

for key, value in study.best_params.items():
    print(f"\t\t{key}: {value}")