from pyspark.sql import SparkSession
from pyspark.ml.feature import OneHotEncoder, StringIndexer
from pyspark.sql import functions as F
from pyspark.ml import Pipeline
from pyspark.sql.types import StringType
from pyspark.sql.types import FloatType

spark = SparkSession.builder.appName("Home Credit default risk").getOrCreate()

def one_hot_encoding(df):
    original_cols = df.columns

    cat_cols = [col for col, dtype in df.dtypes if dtype.startswith('string')]

    indexers = [StringIndexer(inputCol=col, outputCol=col + "_idx",handleInvalid= 'keep') for col in cat_cols]
    indexed_df = df
    for indexer in indexers:
        indexed_df = indexer.fit(indexed_df).transform(indexed_df)

    encoder = OneHotEncoder(inputCols=[col + "_idx" for col in cat_cols], outputCols=[col + "_vec" for col in cat_cols])
    encoded_df = encoder.fit(indexed_df).transform(indexed_df)

    encoded_df = encoded_df.select(original_cols + [col for col in encoded_df.columns if col.endswith("_vec")])

    new_cols = [col for col in encoded_df.columns if col not in original_cols]

    return encoded_df, new_cols

def installment_payments_df():
    installment_payments = spark.read.csv('..\data\installments_payments.csv',header=True,inferSchema=True)
    installment_payments_encoded, installment_payments_cat_col = one_hot_encoding(installment_payments)
    installment_payments_encoded = installment_payments_encoded.withColumn('PAYMENT_PERCENTAGE',F.col('AMT_PAYMENT') / F.col('AMT_INSTALMENT'))
    installment_payments_encoded = installment_payments_encoded.withColumn('DAYS_PAST_DUE', F.when(F.col('DAYS_ENTRY_PAYMENT') - F.col('DAYS_INSTALMENT') > 0, F.col('DAYS_ENTRY_PAYMENT') - F.col('DAYS_INSTALMENT')).otherwise(0))
    installment_payments_encoded = installment_payments_encoded.withColumn('DAYS_BEFORE_DUE',F.when(F.col('DAYS_INSTALMENT')-F.col('DAYS_ENTRY_PAYMENT')>0,F.col('DAYS_INSTALMENT')-F.col('DAYS_ENTRY_PAYMENT')).otherwise(0))
    installment_payments_agg = installment_payments_encoded.groupBy('SK_ID_CURR').agg(
        F.mean('PAYMENT_PERCENTAGE').alias('PAYMENT_PERCENTAGE_MEAN'),
        F.variance('PAYMENT_PERCENTAGE').alias('PAYMENT_PERCENTAGE_VARIANCE'),
        F.mean('DAYS_PAST_DUE').alias('DAYS_PAST_DUE_MEAN'),
        F.mean('NUM_INSTALMENT_VERSION').alias('NUM_INSTALMENT_VERSION_MEAN'),
        F.mean('AMT_PAYMENT').alias('AMT_PAYMENT_MEAN'),
        F.mean('AMT_INSTALMENT').alias('AMT_INSTALMENT_MEAN'),
        F.max('DAYS_PAST_DUE').alias('DAYS_PAST_DUE_MAX'),
        F.max('DAYS_BEFORE_DUE').alias('DAYS_BEFORE_DUE_MAX'),
        F.max('AMT_PAYMENT').alias('AMT_PAYMENT_MAX'),
        F.max('AMT_INSTALMENT').alias('AMT_INSTALMENT_MAX'),
        F.sum('AMT_PAYMENT').alias('AMT_PAYMENT_SUM'),
        F.sum('AMT_INSTALMENT').alias('AMT_INSTALMENT_SUM'),
        F.min('AMT_PAYMENT').alias('AMT_PAYMENT_MIN'),
        F.countDistinct('NUM_INSTALMENT_VERSION').alias('NUM_INSTALMENT_VERSION_COUNT'),
        F.count("*").alias('INSTALL_COUNT')
        )

    return installment_payments_agg

def credit_card_balance_df():
    credit_card_balance = spark.read.csv('data\\credit_card_balance.csv',header=True,inferSchema=True)
    credit_card_balance_encoded, credit_card_cat_col = one_hot_encoding(credit_card_balance)
    credit_card_balance_encoded = credit_card_balance_encoded.drop('SK_ID_PREV')

    credit_card_balance_agg = credit_card_balance_encoded.groupBy('SK_ID_CURR').agg(F.count('*').alias('CREDIT_CARD_COUNT'))
    return credit_card_balance_agg

def pop_cash_df():
    pop_cash = spark.read.csv('data\\POS_CASH_balance.csv',header=True,inferSchema=True)
    pop_cash_encoded, pop_cash_cat_col = one_hot_encoding(pop_cash)

    pop_cash_agg = pop_cash_encoded.groupBy('SK_ID_CURR').agg(
                                                              F.mean('MONTHS_BALANCE').alias('MONTHS_BALANCE_MEAN'),
                                                              F.mean('SK_DPD').alias('SK_DPD_MEAN'),
                                                              F.mean('SK_DPD_DEF').alias('SK_DPD_DEF_MEAN'),
                                                              F.max('SK_DPD_DEF').alias('SK_DPD_DEF_MAX'),
                                                              F.max('MONTHS_BALANCE').alias('MONTHS_BALANCE_MAX'),
                                                              F.max('SK_DPD').alias('SK_DPD_MAX'),
                                                              F.count('*').alias('POS_COUNT'))

    return pop_cash_agg

def previous_application_df():
    previous_application = spark.read.csv('data\\previous_application.csv',header=True,inferSchema=True)
    previous_application = previous_application.replace(365243,None)
    previous_application_encoded, previous_application_cat_col = one_hot_encoding(previous_application)
    previous_application_encoded = previous_application_encoded.withColumn('CREDIT_APP_PERCENT', F.col('AMT_CREDIT')/F.col('AMT_APPLICATION'))

    previous_application_agg = previous_application_encoded.groupBy('SK_ID_CURR').agg(
                                                                                          F.mean('CREDIT_APP_PERCENT').alias('CREDIT_APP_PERCENT_MEAN'),
                                                                                          F.mean('HOUR_APPR_PROCESS_START').alias('HOUR_APPR_PROCESS_START_MEAN'),
                                                                                          F.mean('RATE_DOWN_PAYMENT').alias('RATE_DOWN_PAYMENT_MEAN'),
                                                                                          F.mean('DAYS_DECISION').alias('DAYS_DECISION_MEAN'),
                                                                                          F.mean('CNT_PAYMENT').alias('CNT_PAYMENT_MEAN'),
                                                                                          F.mean('AMT_ANNUITY').alias('PREV_AMT_ANNUITY_MEAN'),
                                                                                          F.mean('AMT_APPLICATION').alias('AMT_APPLICATION_MEAN'),
                                                                                          F.mean('AMT_CREDIT').alias('AMT_CREDIT_MEAN'),
                                                                                          F.mean('AMT_DOWN_PAYMENT').alias('AMT_DOWN_PAYMENT_MEAN'),
                                                                                          F.mean('AMT_GOODS_PRICE').alias('AMT_GOODS_PRICE_MEAN'),
                                                                                          F.max('CREDIT_APP_PERCENT').alias('CREDIT_APP_PERCENT_MAX'),
                                                                                          F.max('HOUR_APPR_PROCESS_START').alias('HOUR_APPR_PROCESS_START_MAX'),
                                                                                          F.max('RATE_DOWN_PAYMENT').alias('RATE_DOWN_PAYMENT_MAX'),
                                                                                          F.max('DAYS_DECISION').alias('DAYS_DECISION_MAX'),
                                                                                          F.max('AMT_ANNUITY').alias('PREV_AMT_ANNUITY_MAX'),
                                                                                          F.max('AMT_APPLICATION').alias('AMT_APPLICATION_MAX'),
                                                                                          F.max('AMT_CREDIT').alias('AMT_CREDIT_MAX'),
                                                                                          F.max('AMT_DOWN_PAYMENT').alias('AMT_DOWN_PAYMENT_MAX'),
                                                                                          F.max('AMT_GOODS_PRICE').alias('AMT_GOODS_PRICE_MAX'),
                                                                                          F.sum('CNT_PAYMENT').alias('CNT_PAYMENT_SUM')
                                                                                          )

    return previous_application_agg

def bureau_and_balance_df():
    bureau = spark.read.csv('data\\bureau.csv', header=True, inferSchema=True)
    bureau_balance = spark.read.csv('data\\bureau_balance.csv', header=True, inferSchema=True)

    bureau_encoded, bureau_cat_col = one_hot_encoding(bureau)
    bureau_balance_encoded, bureau_balance_cat_col = one_hot_encoding(bureau_balance)

    bureau_balance_agg = bureau_balance_encoded.groupBy('SK_ID_BUREAU').agg(
                                                                                F.min('MONTHS_BALANCE').alias('MONTHS_BALANCE_MIN'),
                                                                                F.max('MONTHS_BALANCE').alias('MONTHS_BALANCE_MAX'),
                                                                                F.count('MONTHS_BALANCE').alias('MONTHS_BALANCE_SIZE')
        )

    bureau = bureau.join(bureau_balance_agg, on='SK_ID_BUREAU', how='left')
    bureau = bureau.drop('SK_ID_BUREAU')

    bureau_agg=bureau.groupBy('SK_ID_CURR').agg(F.mean('DAYS_CREDIT_UPDATE').alias('DAYS_CREDIT_UPDATE_MEAN'),
                                                    F.mean('DAYS_CREDIT_ENDDATE').alias('DAYS_CREDIT_ENDDATE_MEAN'),
                                                    F.mean('DAYS_CREDIT').alias('DAYS_CREDIT_MEAN'),
                                                    F.mean('MONTHS_BALANCE_SIZE').alias('MONTHS_BALANCE_SIZE_MEAN'),
                                                    F.mean('AMT_CREDIT_MAX_OVERDUE').alias('AMT_CREDIT_MAX_OVERDUE_MEAN'),
                                                    F.mean('AMT_CREDIT_SUM').alias('AMT_CREDIT_SUM_MEAN'),
                                                    F.mean('AMT_CREDIT_SUM_DEBT').alias('AMT_CREDIT_SUM_DEBT_MEAN'),
                                                    F.mean('AMT_CREDIT_SUM_OVERDUE').alias('AMT_CREDIT_SUM_OVERDUE_MEAN'),
                                                    F.mean('AMT_CREDIT_SUM_LIMIT').alias('AMT_CREDIT_SUM_LIMIT_MEAN'),
                                                    F.mean('CREDIT_DAY_OVERDUE').alias('CREDIT_DAY_OVERDUE_MEAN'),
                                                    F.mean('AMT_ANNUITY').alias('BUREAU_AMT_ANNUITY_MEAN'),
                                                    F.variance('DAYS_CREDIT').alias('DAYS_CREDIT_VAR'),
                                                    F.sum('MONTHS_BALANCE_SIZE').alias('MONTHS_BALANCE_SIZE_SUM'),
                                                    F.sum('AMT_CREDIT_SUM').alias('AMT_CREDIT_SUM_SUM'),
                                                    F.sum('AMT_CREDIT_SUM_DEBT').alias('AMT_CREDIT_SUM_DEBT_SUM'),
                                                    F.sum('AMT_CREDIT_SUM_LIMIT').alias('AMT_CREDIT_SUM_LIMIT_SUM'),
                                                    F.max('AMT_ANNUITY').alias('BUREAU_AMT_ANNUITY_MAX')
                                                    )
    return bureau_agg

def application_df():
    train_df = spark.read.csv('data\\application_train.csv', header=True, inferSchema=True)
    test_df = spark.read.csv('data\\application_test.csv', header=True, inferSchema=True)

    df = train_df.unionByName(test_df, allowMissingColumns=True)
    df = df.filter(df['CODE_GENDER'] != 'XNA')

    df = df.withColumn('DAYS_EMPLOYED', F.when(df['DAYS_EMPLOYED'] == 365243, None).otherwise(df['DAYS_EMPLOYED']))

    df = df.withColumn('FLAG_OWN_CAR', F.when(df['FLAG_OWN_CAR'] == 'Y', 1).otherwise(0))
    df = df.withColumn('FLAG_OWN_REALTY', F.when(df['FLAG_OWN_REALTY'] == 'Y', 1).otherwise(0))

    df = df.withColumn('CODE_GENDER', F.when(df['CODE_GENDER'] == 'M', 1).otherwise(0))

    df = df.withColumn('ANNUITY_CREDIT_RATIO', df['AMT_ANNUITY'] / df['AMT_CREDIT'])
    df = df.withColumn('CREDIT_GOODS_PRICE_RATIO', df['AMT_CREDIT'] / df['AMT_GOODS_PRICE'])
    df = df.withColumn('CREDIT_INCOME_RATIO', df['AMT_CREDIT'] / df['AMT_INCOME_TOTAL'])
    df = df.withColumn('EMPLOYED_TO_BIRTH_RATIO', df['DAYS_EMPLOYED'] / df['DAYS_BIRTH'])
    df = df.withColumn('OWN_CAR_BIRTH_RATIO', (df['OWN_CAR_AGE'] * 365) / df['DAYS_BIRTH'])
    df = df.withColumn('OWN_CAR_EMPLOYED_RATIO', (df['OWN_CAR_AGE'] * 365) / df['DAYS_EMPLOYED'])
    df = df.withColumn('REGISTRATION_TO_BIRTH_RATIO', df['DAYS_REGISTRATION'] / df['DAYS_BIRTH'])
    df = df.withColumn('PHONE_CHANGE_TO_BIRTH_RATIO', df['DAYS_LAST_PHONE_CHANGE'] / df['DAYS_BIRTH'])
    df = df.withColumn('PHONE_CHANGE_TO_REGISTRATION', df['DAYS_LAST_PHONE_CHANGE'] / df['DAYS_REGISTRATION'])
    df = df.withColumn('ID_PUBLISH_TO_REGISTRATION', df['DAYS_ID_PUBLISH'] / df['DAYS_REGISTRATION'])
    df = df.withColumn('ANNUITY_INCOME_RATIO', df['AMT_ANNUITY'] / (1 + df['AMT_INCOME_TOTAL']))
    df = df.withColumn('INCOME_DIVIDE_BY_CHILD', df['AMT_INCOME_TOTAL'] / (1 + df['CNT_CHILDREN']))

    other_flag_cols = [col for col in df.columns if ('FLAG_' in col) and ('FLAG_DOCUMENT_' not in col)]
    df = df.withColumn('NEW_LIVE_IND_SUM', sum(df[col] for col in other_flag_cols))

    drop_cols = ['FLAG_DOCUMENT_2','FLAG_DOCUMENT_4','FLAG_DOCUMENT_5','FLAG_DOCUMENT_6','FLAG_DOCUMENT_7','FLAG_DOCUMENT_8','FLAG_DOCUMENT_9','FLAG_DOCUMENT_10', 'FLAG_DOCUMENT_11','FLAG_DOCUMENT_12','FLAG_DOCUMENT_13','FLAG_DOCUMENT_14','FLAG_DOCUMENT_15','FLAG_DOCUMENT_16','FLAG_DOCUMENT_17','FLAG_DOCUMENT_18','FLAG_DOCUMENT_19','FLAG_DOCUMENT_20','FLAG_DOCUMENT_21',]
    df = df.drop(*drop_cols)

    return df

# Join dataframes together:
def join_dataframe():
    df = application_df()
    bureau_and_balance = bureau_and_balance_df()
    previous_application = previous_application_df()
    pop_cash = pop_cash_df()
    installment_payments = installment_payments_df()
    credit_card_balance = credit_card_balance_df()
    df = df.join(bureau_and_balance, on='SK_ID_CURR',how = 'left')
    df = df.join(previous_application, on='SK_ID_CURR',how = 'left')
    df = df.join(pop_cash, on='SK_ID_CURR',how = 'left')
    df = df.join(installment_payments, on='SK_ID_CURR',how = 'left')
    df = df.join(credit_card_balance, on='SK_ID_CURR',how = 'left')
    train_df = df.where(df['TARGET'].isNotNull())
    test_df = df.where(df['TARGET'].isNull())
    return train_df, test_df