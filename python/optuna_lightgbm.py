import numpy as np
import lightgbm as lgb
import optuna
import pandas as pd
from sklearn.model_selection import StratifiedKFold
from optuna.integration import LightGBMPruningCallback
from sklearn.metrics import roc_auc_score
from data_preprocessing import application_df,bureau_and_balance_df,credit_card_balance_df,installment_payments_df,pop_cash_df,previous_application_df
import gc,re
from timeit import default_timer as timer

# Preparing dataset:
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
    x_train = train_df.drop(columns='TARGET')
    y_train = train_df['TARGET']
    del df, train_df
    gc.collect()
    return x_train, y_train

x_train, y_train = prepare_data()

def objective(X,y,trial):
    param_grid = {
        "n_estimators": trial.suggest_categorical("n_estimators", [10000,20000]),
        "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.3),
        "num_leaves": trial.suggest_int("num_leaves", 20, 100, step=10),
        "max_depth": trial.suggest_int("max_depth", 3, 12),
        'colsample_bytree': trial.suggest_float('colsample_by_tree', 0.8, 1.0),
        'subsample': trial.suggest_float('subsample',0.8,1.0),
        'reg_alpha': trial.suggest_float('reg_alpha', 0.0, 0.2),
        'reg_lambda': trial.suggest_float('reg_lambda', 0.0, 0.2),
        'min_split_gain': trial.suggest_float('min_split_gain',0.0,0.05),
        'min_child_weight': trial.suggest_int('min_child_weight',10,50)
    }

    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=210)
    cv_scores = np.empty(5)

    for idx, (train_idx,valid_idx) in enumerate(cv.split(X,y)):
        X_train, X_valid = X.iloc[train_idx], X.iloc[valid_idx]
        y_train, y_valid = y.iloc[train_idx], y.iloc[valid_idx]
        model = lgb.LGBMClassifier(objective='binary',**param_grid)
        model.fit(
            X_train,y_train,
            eval_set=[(X_valid,y_valid)],
            eval_metric='auc',
            callbacks=[lgb.early_stopping(stopping_rounds=100),LightGBMPruningCallback(trial, 'auc')]
        )
        preds = model.predict_proba(X_valid)[:,1]
        cv_scores[idx] = roc_auc_score(y_valid,preds)
    return np.mean(cv_scores)

study = optuna.create_study(direction='maximize', study_name='LGBM Classifier')
func = lambda trial: objective(x_train,y_train,trial)
study.optimize(func,n_trials=100)

print(f"\tBest value (auc): {study.best_value:.5f}")
print(f"\tBest params:")

for key, value in study.best_params.items():
    print(f"\t\t{key}: {value}")