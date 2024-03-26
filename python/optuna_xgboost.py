# XGBoost hyperparams tuning:
import numpy as np
import optuna
import pandas as pd
from sklearn.model_selection import StratifiedKFold
from optuna_integration import XGBoostPruningCallback
from sklearn.metrics import roc_auc_score
from data_preprocessing import application_df,bureau_and_balance_df,credit_card_balance_df,installment_payments_df,pop_cash_df,previous_application_df
import gc,re
from timeit import default_timer as timer
import xgboost as xgb

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
  pruning_callback = optuna.integration.XGBoostPruningCallback(trial, "validation-auc")
  param_grid = {
      'verbosity':0,
      'booster': trial.suggest_categorical('booster',['gbtree','gblinear','dart']),
      "objective": "binary:logistic",
      "eval_metric": "auc",
      'seed':210
  }
  if param_grid['booster'] == 'gblinear':
     param_grid['lambda'] = trial.suggest_float('lambda', 0.0, 1.0)
     param_grid['alpha'] = trial.suggest_float('alpha', 0.0, 1.0)

  if param_grid['booster'] == 'gbtree' or param_grid['booster']=='dart':
    param_grid["max_depth"] = trial.suggest_int("max_depth", 3, 12)
    param_grid['subsample'] = trial.suggest_float('subsample',0.5,1.0)
    param_grid["eta"] = trial.suggest_float("eta", 1e-8, 1.0, log=True)
    param_grid["gamma"] = trial.suggest_float("gamma", 1e-8, 1.0, log=True)
    param_grid["grow_policy"] = trial.suggest_categorical("grow_policy", ["depthwise", "lossguide"])
    param_grid['lambda'] = trial.suggest_float('lambda', 0.0, 1.0)
    param_grid['alpha'] = trial.suggest_float('lambda', 0.0, 1.0)
    param_grid['max_delta_step'] = trial.suggest_int("max_delta_step", 1, 10)

  if param_grid["booster"] == "dart":
    param_grid["sample_type"] = trial.suggest_categorical("sample_type", ["uniform", "weighted"])
    param_grid["normalize_type"] = trial.suggest_categorical("normalize_type", ["tree", "forest"])
    param_grid["rate_drop"] = trial.suggest_float("rate_drop", 1e-8, 1.0, log=True)
    param_grid["skip_drop"] = trial.suggest_float("skip_drop", 1e-8, 1.0, log=True)

  cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=210)
  cv_scores = np.empty(5)

  for idx, (train_idx,valid_idx) in enumerate(cv.split(X,y)):
        X_train, X_valid = X.iloc[train_idx], X.iloc[valid_idx]
        y_train, y_valid = y.iloc[train_idx], y.iloc[valid_idx]
        dtrain = xgb.DMatrix(X_train, label=y_train)
        dvalid = xgb.DMatrix(X_valid, label=y_valid)
        xgboost_train = xgb.train(param_grid, dtrain, evals=[(dvalid, "validation")], callbacks=[pruning_callback])
        # model = xgb.XGBClassifier(**param_grid)
        # model.fit(
        #     X_train,y_train,
        #     eval_set=[(X_valid,y_valid)],
        #     # eval_metric='auc',
        #     early_stopping_rounds=100,
        #     verbose=False,
        #     callbacks=[XGBoostPruningCallback(trial, "test-auc")]
        # )
        preds = xgboost_train.predict(dvalid)
        class_1_proba = preds
        # preds = model.predict_proba(X_valid)[:,1]
        cv_scores[idx] = roc_auc_score(y_valid,class_1_proba)
  return np.mean(cv_scores)

study = optuna.create_study(direction='maximize', study_name='XGBoost Classifier')
func = lambda trial: objective(trial,x_train,y_train)
study.optimize(func,n_trials=250)

print(f"\tBest value (auc): {study.best_value:.5f}")
print(f"\tBest params:")

for key, value in study.best_params.items():
    print(f"\t\t{key}: {value}")