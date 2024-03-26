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

# Missing values in each table:
def missing_values_table(df, df_name):
    
    # Calculate missing values
    missing_val = df.isnull().sum()
    missing_val_percent = df.isnull().sum() / len(df) * 100.0
    
    # Create missing values table
    missing_val_table = pd.concat([missing_val, missing_val_percent], axis=1)
    missing_val_table = missing_val_table.rename(columns={0: 'Missing Values', 1: 'Percent of Total Values'})
    missing_val_table = missing_val_table[missing_val_table.iloc[:, 1] != 0].sort_values('Percent of Total Values', ascending=False).round(2)
    
    print(f'\n{df_name} dataframe has {df.shape[1]} columns and {missing_val_table.shape[0]} columns with missing values')
    
    return missing_val_table

# One-hot encoder (get dummies) for categorical columns:
def one_hot_encoding(df):
    df_original_cols = list(df.columns)
    cat_col = [col for col in df.columns if df[col].dtype == 'object']
    df = pd.get_dummies(df, columns = cat_col, dummy_na = True)
    new_col = [col for col in df.columns if col not in df_original_cols]
    return df, new_col

# Transform installment_payments dataframe:
def installment_payments_df():
    installment_payments = pd.read_csv('data\\installments_payments.csv')
    installment_payments, ins_pay_cat_col = one_hot_encoding(installment_payments)
    installment_payments['PAYMENT_PERCENTAGE'] = installment_payments['AMT_PAYMENT']/installment_payments['AMT_INSTALMENT']
    installment_payments['DAYS_PAST_DUE'] = np.maximum(installment_payments['DAYS_ENTRY_PAYMENT']-installment_payments['DAYS_INSTALMENT'],0)
    installment_payments['DAYS_BEFORE_DUE'] = np.maximum(installment_payments['DAYS_INSTALMENT']-installment_payments['DAYS_ENTRY_PAYMENT'],0)
    feature_agg = {
        'PAYMENT_PERCENTAGE':['mean','var'],
        'DAYS_PAST_DUE':['mean','max'],
        'DAYS_BEFORE_DUE':['mean','max'],
        'NUM_INSTALMENT_VERSION':['nunique'],
        'AMT_PAYMENT': ['max', 'min', 'mean', 'sum'],
        'AMT_INSTALMENT': ['max', 'mean', 'sum']
    }
    for col in ins_pay_cat_col:
        feature_agg[col] = ['mean']
    installment_payments_agg = installment_payments.groupby('SK_ID_CURR').agg(feature_agg)
    installment_payments_agg.columns = pd.Index(['INSTALL_PAYMENT' + i[0] + "_" + i[1].upper() for i in installment_payments_agg.columns])
    installment_payments_agg['INSTALL_COUNT'] = installment_payments.groupby('SK_ID_CURR').size()
    del installment_payments
    gc.collect()
    return installment_payments_agg

# Transfrom credit_card_balance dataframe:
def credit_card_balance_df():
    credit_card_balance = pd.read_csv('data\\credit_card_balance.csv')
    credit_card_balance, credit_card_cat_cols = one_hot_encoding(credit_card_balance)
    credit_card_balance.drop(columns=['SK_ID_PREV'],inplace=True,axis=1)
    feature_agg = {}
    for col in credit_card_balance.columns:
        if col in credit_card_cat_cols: feature_agg[col] = ['mean']
        else: feature_agg[col] = ['max','mean','var']
    credit_card_balance_agg = credit_card_balance.groupby('SK_ID_CURR').agg(feature_agg)
    credit_card_balance_agg.columns = pd.Index(['CREDIT_CARD_' + i[0] + "_" + i[1].upper() for i in credit_card_balance_agg.columns])
    credit_card_balance_agg['CREDIT_CARD_COUNT'] = credit_card_balance_agg.groupby('SK_ID_CURR').size()
    del credit_card_balance
    gc.collect()
    return credit_card_balance_agg

# Transform pop_cash_balance dataframe:
def pop_cash_df():
    pop_cash_balance = pd.read_csv('data\\POS_CASH_balance.csv')
    pop_cash_balance, pop_cash_cat_cols = one_hot_encoding(pop_cash_balance)

    # Numerical features aggregation:
    feature_agg = {
        'MONTHS_BALANCE':['mean','max','size'],
        'SK_DPD':['mean','max'],
        'SK_DPD_DEF':['mean','max']
    }
    # For categorical features (now have been one-hot encoded), take the mean of these columns:
    for col in pop_cash_cat_cols:
        feature_agg[col] = ['mean']
    # Aggregate the features:
    pop_cash_agg = pop_cash_balance.groupby('SK_ID_CURR').agg(feature_agg)
    pop_cash_agg.columns = pd.Index(['POP_CASH' + i[0] + "_" + i[1].upper() for i in pop_cash_agg.columns])
    pop_cash_agg['POS_COUNT'] = pop_cash_balance.groupby('SK_ID_CURR').size()
    del pop_cash_balance
    gc.collect()
    return pop_cash_agg

# Transform previous_application dataframe:
def previous_application_df():
    previous_application = pd.read_csv('data\\previous_application.csv')
    previous_application['DAYS_FIRST_DRAWING'].replace(365243, np.nan, inplace=True)
    previous_application['DAYS_FIRST_DUE'].replace(365243,np.nan,inplace=True)
    previous_application['DAYS_LAST_DUE_1ST_VERSION'].replace(365243,np.nan,inplace=True)
    previous_application['DAYS_LAST_DUE'].replace(365243,np.nan,inplace=True)
    previous_application['DAYS_TERMINATION'].replace(365243,np.nan,inplace=True)
    previous_application, previous_application_cat_col = one_hot_encoding(previous_application)
    # Percent of amount credit vs amount client asked for
    previous_application['CREDIT_APP_PERCENT'] = previous_application['AMT_CREDIT']/previous_application['AMT_APPLICATION']
    numerical_feature_agg = {
        'CREDIT_APP_PERCENT': ['max', 'mean'],
        'HOUR_APPR_PROCESS_START': ['max', 'mean'],
        'RATE_DOWN_PAYMENT': ['max', 'mean'],
        'DAYS_DECISION': ['max', 'mean'],
        'CNT_PAYMENT': ['mean', 'sum'],
        'AMT_ANNUITY': ['max', 'mean'],
        'AMT_APPLICATION': ['max', 'mean'],
        'AMT_CREDIT': ['max', 'mean'],
        'AMT_DOWN_PAYMENT': ['max', 'mean'],
        'AMT_GOODS_PRICE': ['max', 'mean']
    }
    categorical_feature_agg = {}
    for col in previous_application_cat_col:
        categorical_feature_agg[col]=['mean']
    previous_application_agg = previous_application.groupby('SK_ID_CURR').agg({**numerical_feature_agg,**categorical_feature_agg})
    previous_application_agg.columns = pd.Index(['PREV_APP' + e[0] + "_" + e[1].upper() for e in previous_application_agg.columns])

    # For all applications with 'NAME_CONTRACT_STATUS' = 'Approved', aggregate all numerical features
    pre_app_approved = previous_application[previous_application['NAME_CONTRACT_STATUS_Approved']==1]
    pre_app_approved_agg = pre_app_approved.groupby('SK_ID_CURR').agg(numerical_feature_agg)
    pre_app_approved_agg.columns = pd.Index(['PRE_APP_APPROVED_'+i[0]+'_'+i[1].upper() for i in pre_app_approved_agg.columns])
    previous_application_agg.join(pre_app_approved_agg,how='left',on='SK_ID_CURR')
    # For all applications with 'NAME_CONTRACT_STATUS' = 'Refused', aggregate all numerical features
    pre_app_refused = previous_application[previous_application['NAME_CONTRACT_STATUS_Refused']==1]
    pre_app_refused_agg = pre_app_refused.groupby('SK_ID_CURR').agg(numerical_feature_agg)
    pre_app_refused_agg.columns = pd.Index(['PRE_APP_REFUSED_'+i[0]+'_'+i[1].upper() for i in pre_app_refused_agg.columns])
    previous_application_agg.join(pre_app_refused_agg,how='left',on='SK_ID_CURR')
    del pre_app_approved, pre_app_refused, pre_app_approved_agg, pre_app_refused_agg
    gc.collect()
    return previous_application_agg

# Transform bureau and bureau_balance dataframe:
def bureau_and_balance_df():
    bureau = pd.read_csv('data\\bureau.csv')
    bureau_balance = pd.read_csv('data\\bureau_balance.csv')
    bureau, bureau_cat_col = one_hot_encoding(bureau)
    bureau_balance, bureau_balance_cat_col = one_hot_encoding(bureau_balance)

    # Bureau_balance dataframe:
    feature_agg = {'MONTHS_BALANCE':['size','min','max']}
    for col in bureau_balance_cat_col: feature_agg[col] = ['mean']
    bureau_balance_agg = bureau_balance.groupby('SK_ID_BUREAU').agg(feature_agg)
    bureau_balance_agg.columns = pd.Index([i[0]+'_'+i[1].upper() for i in bureau_balance_agg.columns])
    bureau = bureau.join(bureau_balance_agg, how = 'left', on = 'SK_ID_BUREAU')
    bureau.drop(columns=['SK_ID_BUREAU'],inplace = True)
    del bureau_balance,bureau_balance_agg
    gc.collect()

    numerical_feature_agg = {
            'DAYS_CREDIT_UPDATE': ['mean'],
            'DAYS_CREDIT_ENDDATE': ['mean'],
            'DAYS_CREDIT': ['var','mean'],
            'MONTHS_BALANCE_MIN': ['min'],
            'MONTHS_BALANCE_MAX': ['max'],
            'MONTHS_BALANCE_SIZE': ['mean', 'sum'],
            'AMT_CREDIT_MAX_OVERDUE': ['mean'],
            'AMT_CREDIT_SUM': [ 'mean', 'sum'],
            'AMT_CREDIT_SUM_DEBT': [ 'mean', 'sum'],
            'AMT_CREDIT_SUM_OVERDUE': ['mean'],
            'AMT_CREDIT_SUM_LIMIT': ['mean', 'sum'],
            'CREDIT_DAY_OVERDUE': ['mean'],
            'CNT_CREDIT_PROLONG': ['sum'],
            'AMT_ANNUITY': ['max', 'mean']
    }
    cat_feature_agg = {}
    for col in bureau_balance_cat_col: cat_feature_agg[col + "_" + 'MEAN'] = ['mean']
    for col in bureau_cat_col: cat_feature_agg[col] = ['mean']

    bureau_agg = bureau.groupby('SK_ID_CURR').agg({**numerical_feature_agg, **cat_feature_agg})
    bureau_agg.columns = pd.Index(['BUREAU_BALANCE_' + i[0] + "_" + i[1] for i in bureau_agg.columns])

    # Aggregate numerical features for all "ACTIVE" credits:
    bureau_active = bureau[bureau['CREDIT_ACTIVE_Active']==1]
    bureau_active_agg = bureau_active.groupby('SK_ID_CURR').agg(numerical_feature_agg)
    bureau_active_agg.columns = pd.Index(['ACTIVE_' + i[0] + '_' + i[1].upper() for i in bureau_active_agg.columns])
    bureau_agg = bureau_agg.join(bureau_active_agg, on = 'SK_ID_CURR', how = 'left')
    del bureau_active,bureau_active_agg
    gc.collect()

    # Aggregate numerical features for all "CLOSED" credits:
    bureau_closed = bureau[bureau['CREDIT_ACTIVE_Closed']==1]
    bureau_closed_agg = bureau_closed.groupby('SK_ID_CURR').agg(numerical_feature_agg)
    bureau_closed_agg.columns = pd.Index(['CLOSED_' + i[0] + '_' + i[1].upper() for i in bureau_closed_agg.columns])
    bureau_agg = bureau_agg.join(bureau_closed_agg, on = 'SK_ID_CURR', how = 'left')
    del bureau_closed,bureau_closed_agg
    gc.collect()
    return bureau_agg

# Transform application_train dataframe
def application_df():
    train_df = pd.read_csv('data\\application_train.csv')
    test_df = pd.read_csv('data\\application_test.csv')
    df = pd.concat([train_df,test_df])

    df = df[df['CODE_GENDER']!='XNA']
    df['DAYS_EMPLOYED'].replace(365243,np.nan,inplace=True)
    df['FLAG_OWN_CAR'] = df['FLAG_OWN_CAR'].map({'Y': 1, 'N': 0})
    df['FLAG_OWN_REALTY'] = df['FLAG_OWN_REALTY'].map({'Y': 1, 'N': 0})
    df['CODE_GENDER'] = df['CODE_GENDER'].map({'M':1,'F':0})

    df['ANNUITY_CREDIT_RATIO'] = df['AMT_ANNUITY']/df['AMT_CREDIT']
    df['CREDIT_GOODS_PRICE_RATIO'] = df['AMT_CREDIT']/df['AMT_GOODS_PRICE']
    df['CREDIT_INCOME_RATIO'] = df['AMT_CREDIT']/df['AMT_INCOME_TOTAL']
    df['EMPLOYED_TO_BIRTH_RATIO'] = df['DAYS_EMPLOYED']/df['DAYS_BIRTH']
    df['OWN_CAR_BIRTH_RATIO'] = df['OWN_CAR_AGE']*365/df['DAYS_BIRTH']
    df['OWN_CAR_EMPLOYED_RATIO'] = df['OWN_CAR_AGE']*365/df['DAYS_EMPLOYED']
    df['REGISTRATION_TO_BIRTH_RATIO'] = df['DAYS_REGISTRATION']/df['DAYS_BIRTH']
    df['PHONE_CHANGE_TO_BIRTH_RATIO'] = df['DAYS_LAST_PHONE_CHANGE']/df['DAYS_BIRTH']
    df['PHONE_CHANGE_TO_REGISTRATION'] = df['DAYS_LAST_PHONE_CHANGE']/df['DAYS_REGISTRATION']
    df['ID_PUBLISH_TO_REGISTRATION'] = df['DAYS_ID_PUBLISH']/df['DAYS_REGISTRATION']
    # Add 1 to denominator to avoid dividing by 0
    df['ANNUITY_INCOME_RATIO'] = df['AMT_ANNUITY']/(1+df['AMT_INCOME_TOTAL'])
    df['INCOME_DIVIDE_BY_CHILD'] = df['AMT_INCOME_TOTAL']/(1+df['CNT_CHILDREN'])
    # All columns for FLAG_DOCUMENT:
    flag_doc_cols = [col for col in df.columns if 'FLAG_DOCUMENT_' in col]
    other_flag_cols = [col for col in df.columns if ('FLAG_' in col) & ('FLAG_DOCUMENT_' not in col)]
    df['NEW_DOC_IND_KURT'] = df[flag_doc_cols].kurtosis(axis=1)
    df['NEW_LIVE_IND_SUM'] = df[other_flag_cols].sum(axis=1)

    df, df_cat_col = one_hot_encoding(df)
    drop_cols = ['FLAG_DOCUMENT_2','FLAG_DOCUMENT_4','FLAG_DOCUMENT_5','FLAG_DOCUMENT_6','FLAG_DOCUMENT_7','FLAG_DOCUMENT_8','FLAG_DOCUMENT_9','FLAG_DOCUMENT_10', 'FLAG_DOCUMENT_11','FLAG_DOCUMENT_12','FLAG_DOCUMENT_13','FLAG_DOCUMENT_14','FLAG_DOCUMENT_15','FLAG_DOCUMENT_16','FLAG_DOCUMENT_17','FLAG_DOCUMENT_18','FLAG_DOCUMENT_19','FLAG_DOCUMENT_20','FLAG_DOCUMENT_21']
    df = df.drop(columns=drop_cols)
    gc.collect()
    return df