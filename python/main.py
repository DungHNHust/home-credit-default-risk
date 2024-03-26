import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import KFold, StratifiedKFold
import re
import lightgbm as lgb
from lightgbm import LGBMClassifier
pd.set_option('display.max_rows', None)
pd.pandas.set_option('display.max_columns', None)
import time
from contextlib import contextmanager
import gc
import catboost as cb
import xgboost as xgb
from data_preprocessing import application_df,bureau_and_balance_df,credit_card_balance_df,installment_payments_df,pop_cash_df,previous_application_df

def lightgbm_kfold(df, num_folds, stratified, debug):
    train_df = df[df['TARGET'].notnull()]
    test_df = df[df['TARGET'].isnull()]
    train_df = train_df.rename(columns = lambda x:re.sub('[^A-Za-z0-9_]+', '', x))
    test_df = test_df.rename(columns = lambda x:re.sub('[^A-Za-z0-9_]+', '', x))

    if stratified == True:
        folds = StratifiedKFold(n_splits=num_folds, shuffle=True, random_state=10)
    else:
        folds = KFold(n_splits=num_folds, shuffle=True, random_state=10)
    
    # Out-of-fold predictions:
    oof_preds = np.zeros(train_df.shape[0])
    # Submission predictions:
    sub_preds = np.zeros(test_df.shape[0])
    # Feature importance dataframe:
    feature_importance_df = pd.DataFrame()

    features = [col for col in train_df.columns if col not in ['TARGET','SK_ID_CURR','SK_ID_BUREAU','SK_ID_PREV','index']]

    for n_fold, (train_index, valid_index) in enumerate(folds.split(train_df[features],train_df['TARGET'])):
        x_train, y_train = train_df[features].iloc[train_index], train_df['TARGET'].iloc[train_index]
        x_valid, y_valid = train_df[features].iloc[valid_index], train_df['TARGET'].iloc[valid_index]

        clf = LGBMClassifier(nthread=4,
            n_estimators=10000,
            learning_rate=0.03482523180661756,
            num_leaves=80,
            colsample_bytree=0.9739734801657047,
            subsample=0.8900817399906373,
            max_depth=5,
            reg_alpha=0.17942672790698105,
            reg_lambda=0.03496900238298586,
            min_split_gain=0.028509160728525926,
            min_child_weight=45,
            silent=-1,
            verbose=-1)
        clf.fit(x_train, y_train, eval_set=[(x_train, y_train),(x_valid, y_valid)], eval_metric='auc', callbacks=[lgb.early_stopping(stopping_rounds=100)])
        oof_preds[valid_index] = clf.predict_proba(x_valid, num_iteration=clf.best_iteration_)[:,1]
        sub_preds += clf.predict_proba(test_df[features],num_iteration=clf.best_iteration_)[:,1]/folds.get_n_splits()

        fold_importance_df = pd.DataFrame()
        fold_importance_df["feature"] = features
        fold_importance_df["importance"] = clf.feature_importances_
        fold_importance_df["fold"] = n_fold + 1
        feature_importance_df = pd.concat([feature_importance_df, fold_importance_df], axis=0)
        print('Fold %2d AUC : %.6f' % (n_fold + 1, roc_auc_score(y_valid, oof_preds[valid_index])))
        del clf, x_train, y_train, x_valid, y_valid
        gc.collect()
    
    print('Full AUC score %.6f' % roc_auc_score(train_df['TARGET'], oof_preds))
    # Write submission file and plot feature importance
    submission_file = 'submission_result\\lightgbm.csv'
    if not debug:
        test_df['TARGET'] = sub_preds
        test_df[['SK_ID_CURR', 'TARGET']].to_csv(submission_file, index= False)
    feature_importance_display(feature_importance_df)
    return feature_importance_df

def xgboost_kfold(df, num_folds, stratified, debug):
    train_df = df[df['TARGET'].notnull()]
    test_df = df[df['TARGET'].isnull()]
    train_df = train_df.rename(columns = lambda x:re.sub('[^A-Za-z0-9_]+', '', x))
    test_df = test_df.rename(columns = lambda x:re.sub('[^A-Za-z0-9_]+', '', x))
    # This only applies to XGBoost and CatBoost 
    train_cols_with_inf = train_df.columns[train_df.isin([np.inf, -np.inf]).any()].tolist()
    train_df = train_df.drop(columns=train_cols_with_inf)
    test_cols_with_inf = test_df.columns[test_df.isin([np.inf, -np.inf]).any()].tolist()
    test_df = test_df.drop(columns=test_cols_with_inf)
    if stratified == True:
        folds = StratifiedKFold(n_splits=num_folds, shuffle=True, random_state=10)
    else:
        folds = KFold(n_splits=num_folds, shuffle=True, random_state=10)
    
    # Out-of-fold predictions:
    oof_preds = np.zeros(train_df.shape[0])
    # Submission predictions:
    sub_preds = np.zeros(test_df.shape[0])
    # Feature importance dataframe:
    feature_importance_df = pd.DataFrame()

    features = [col for col in train_df.columns if col not in ['TARGET','SK_ID_CURR','SK_ID_BUREAU','SK_ID_PREV','index']]

    for n_fold, (train_index, valid_index) in enumerate(folds.split(train_df[features],train_df['TARGET'])):
        x_train, y_train = train_df[features].iloc[train_index], train_df['TARGET'].iloc[train_index]
        x_valid, y_valid = train_df[features].iloc[valid_index], train_df['TARGET'].iloc[valid_index]
        clf = xgb.XGBClassifier(booster = 'gbtree',
                max_depth = 6,
                subsample = 0.8535350061067392,
                eta = 0.3919545719350727,
                gamma = 0.00014679689204015153,
                grow_policy = 'depthwise',
                max_delta_step = 4, eval_metric = 'auc', early_stopping_rounds = 100, reg_lambda = 0.9953186998101056)
        clf.fit(x_train, y_train,eval_set=[(x_train, y_train),(x_valid, y_valid)])
        
        oof_preds[valid_index] = clf.predict_proba(x_valid
                                                #    , num_iteration=clf.best_iteration_
                                                   )[:,1]
        sub_preds += clf.predict_proba(test_df[features]
                                    #    ,num_iteration=clf.best_iteration_
                                       )[:,1]/folds.get_n_splits()

        fold_importance_df = pd.DataFrame()
        fold_importance_df["feature"] = features
        fold_importance_df["importance"] = clf.feature_importances_
        fold_importance_df["fold"] = n_fold + 1
        feature_importance_df = pd.concat([feature_importance_df, fold_importance_df], axis=0)
        print('Fold %2d AUC : %.6f' % (n_fold + 1, roc_auc_score(y_valid, oof_preds[valid_index])))
        del clf, x_train, y_train, x_valid, y_valid
        gc.collect()
    
    print('Full AUC score %.6f' % roc_auc_score(train_df['TARGET'], oof_preds))
    # Write submission file and plot feature importance
    submission_file = 'submission_result\\xgboost.csv'
    if not debug:
        test_df['TARGET'] = sub_preds
        test_df[['SK_ID_CURR', 'TARGET']].to_csv(submission_file, index= False)
    feature_importance_display(feature_importance_df)
    return feature_importance_df

def catboost_kfold(df, num_folds, stratified, debug):
    train_df = df[df['TARGET'].notnull()]
    test_df = df[df['TARGET'].isnull()]
    train_df = train_df.rename(columns = lambda x:re.sub('[^A-Za-z0-9_]+', '', x))
    test_df = test_df.rename(columns = lambda x:re.sub('[^A-Za-z0-9_]+', '', x))
    # This only applies to XGBoost and CatBoost 
    train_cols_with_inf = train_df.columns[train_df.isin([np.inf, -np.inf]).any()].tolist()
    train_df = train_df.drop(columns=train_cols_with_inf)
    test_cols_with_inf = test_df.columns[test_df.isin([np.inf, -np.inf]).any()].tolist()
    test_df = test_df.drop(columns=test_cols_with_inf)
    if stratified == True:
        folds = StratifiedKFold(n_splits=num_folds, shuffle=True, random_state=10)
    else:
        folds = KFold(n_splits=num_folds, shuffle=True, random_state=10)
    
    # Out-of-fold predictions:
    oof_preds = np.zeros(train_df.shape[0])
    # Submission predictions:
    sub_preds = np.zeros(test_df.shape[0])
    # Feature importance dataframe:
    feature_importance_df = pd.DataFrame()

    features = [col for col in train_df.columns if col not in ['TARGET','SK_ID_CURR','SK_ID_BUREAU','SK_ID_PREV','index']]

    for n_fold, (train_index, valid_index) in enumerate(folds.split(train_df[features],train_df['TARGET'])):
        x_train, y_train = train_df[features].iloc[train_index], train_df['TARGET'].iloc[train_index]
        x_valid, y_valid = train_df[features].iloc[valid_index], train_df['TARGET'].iloc[valid_index]
        clf = cb.CatBoostClassifier(objective='Logloss',colsample_bylevel=0.05445698706446003,depth=4,boosting_type='Plain',bootstrap_type='MVS',eval_metric='AUC')
        clf.fit(x_train, y_train,eval_set=[(x_train, y_train),(x_valid, y_valid)],early_stopping_rounds=100)

        oof_preds[valid_index] = clf.predict_proba(x_valid)[:,1]
        sub_preds += clf.predict_proba(test_df[features])[:,1]/folds.get_n_splits()
        fold_importance_df = pd.DataFrame()
        fold_importance_df["feature"] = features
        fold_importance_df["importance"] = clf.feature_importances_
        fold_importance_df["fold"] = n_fold + 1
        feature_importance_df = pd.concat([feature_importance_df, fold_importance_df], axis=0)
        print('Fold %2d AUC : %.6f' % (n_fold + 1, roc_auc_score(y_valid, oof_preds[valid_index])))
        del clf, x_train, y_train, x_valid, y_valid
        gc.collect()
    
    print('Full AUC score %.6f' % roc_auc_score(train_df['TARGET'], oof_preds))
    # Write submission file and plot feature importance
    submission_file = 'submission_result\\catboost.csv'
    if not debug:
        test_df['TARGET'] = sub_preds
        test_df[['SK_ID_CURR', 'TARGET']].to_csv(submission_file, index= False)
    feature_importance_display(feature_importance_df)
    return feature_importance_df

# Display feature importances
def feature_importance_display(feature_importance_df):
    # Select 30 features with highest importance
    columns = feature_importance_df[['feature', 'importance']].groupby('feature').mean().sort_values(by='importance', ascending = False)[:30].index
    best_features = feature_importance_df.loc[feature_importance_df['feature'].isin(columns)]
    plt.figure(figsize=(9,16))
    sns.barplot(data=best_features.sort_values(by='importance', ascending = False), x = 'importance', y = 'feature')
    plt.title('Features with importance (average over folds)')
    plt.tight_layout()

@contextmanager
def timer(title):
    t0 = time.time()
    yield
    print(f'{title} - finished after {time.time()-t0}s.')


def main(debug = False):
    df = application_df()
    with timer('Processing bureau and balance dataframe'):
        bureau_and_balance = bureau_and_balance_df()
        bureau_and_balance = bureau_and_balance.rename(columns = lambda x:re.sub('[^A-Za-z0-9_]+', '', x))
        print(f'Bureau and balance dataframe shape: {bureau_and_balance.shape}')
        df = df.join(bureau_and_balance, on ='SK_ID_CURR', how='left')
        del bureau_and_balance
        gc.collect()
    with timer('Processing previous application dataframe'):
        previous_application = previous_application_df()
        previous_application = previous_application.rename(columns = lambda x:re.sub('[^A-Za-z0-9_]+', '', x))
        print(f'Previous application dataframe shape: {previous_application.shape}')
        df = df.join(previous_application, on ='SK_ID_CURR', how = 'left')
        del previous_application
        gc.collect()
    with timer('Processing pop-cash balance'):
        pop_cash = pop_cash_df()
        pop_cash = pop_cash.rename(columns = lambda x:re.sub('[^A-Za-z0-9_]+', '', x))
        print(f'Pop-cash balance dataframe shape: {pop_cash.shape}')
        df = df.join(pop_cash, on ='SK_ID_CURR', how='left')
        del pop_cash
        gc.collect()
    with timer('Processing installments payments'):
        installment_payments = installment_payments_df()
        installment_payments = installment_payments.rename(columns = lambda x:re.sub('[^A-Za-z0-9_]+', '', x))
        print(f'Installment payments dataframe shape: {installment_payments.shape}')
        df = df.join(installment_payments, on='SK_ID_CURR', how='left')
        del installment_payments
        gc.collect()
    with timer('Processing credit card payments'):
        credit_card_balance = credit_card_balance_df()
        credit_card_balance = credit_card_balance.rename(columns = lambda x:re.sub('[^A-Za-z0-9_]+', '', x))
        print(f'Credit card balance dataframe: {credit_card_balance.shape}')
        df = df.join(credit_card_balance, on = 'SK_ID_CURR', how = 'left')
        del credit_card_balance
        gc.collect()
    with timer('Run models training with K-folds'):
        lightgbm_feature_importance = lightgbm_kfold(df, num_folds=5,stratified=False, debug=debug)
        catboost_feature_importance = catboost_kfold(df, num_folds=5,stratified=False, debug=debug)
        xgboost_feature_importance = xgboost_kfold(df, num_folds=5,stratified=False, debug=debug)

if __name__ == '__main__':
    with timer('Model run'):
        main()