# PreProcessing and Feature Selection Functions

import os
import time
import pandas as pd
import numpy as np
import pickle
import logging

from datetime import datetime, timedelta
from functools import wraps

from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.metrics import roc_auc_score, auc, precision_recall_curve, average_precision_score
from statsmodels.stats.outliers_influence import variance_inflation_factor

import xgboost as xgb

def timeit(func):
    @wraps(func)
    def timeit_wrapper(*args, **kwargs):
        start_time = time.time()
        print('start_time', time.asctime( time.localtime(start_time) ))
        result = func(*args, **kwargs)
        end_time = time.time()
        total_time = end_time - start_time
        print(f' Function {func.__name__} took {str(timedelta(seconds=total_time))}')
    return timeit_wrapper
        

def save_obj(obj, name, file_path):
    """
    This function saves any object as a .pkl file so that it can be easily read in other notebooks
    
    Parameters
    -----
    obj : Object's variable name 
        The object that needs to be saved.
    name : String
        The name that you would like to save the object as.
    file_path: String
        Directory in which you are saving the object to.
    
    Returns
    -----
        A saved .pkl file in dir and name specified.
    """
    file_path = os.path.join(file_path, name)
    with open(file_path + '.pkl', 'wb') as f:
        pickle.dump(obj, f, pickle.HIGHEST_PROTOCOL)
        
def load_obj(name, file_path):
    """
    This function loads any .pkl file
    
    Parameters
    -----
    name : String
        The name of the file that needs to be loaded.
    file_path: String
        Directory in which you are loading the object from.
    
    Returns
    -----
        An object that was saved as a .pkl file in dir and name specified.
    """
    file_path = os.path.join(file_path, name)
    with open(file_path + '.pkl', 'rb') as f:
        return pickle.load(f)

    
    
def reduce_mem_usage(df):
    """
    Optimizes memory usage of pandas dataframe
    
    Parameters
    -----
        df: pandas.DataFrame 
            dataframe to optimize
        
    Returns
    -----
        df: pandas.DataFrame 
            optimized dataframe
    """
    
    start_mem = df.memory_usage().sum() / 1024**2
    print('Memory usage of dataframe is {:.2f} MB'.format(start_mem))
    
    # Iterate through all columns
    for col in df.columns:
        
        # If column is of type 'object'
        if df[col].dtype == 'object':
            
            # Check if column can be converted to datetime
            try:
                df[col] = pd.to_datetime(df[col], infer_datetime_format=True, errors='raise')
                
            # If there are errors in converting to datetime, skip to the next column
            except ValueError:
                pass
        
        # If column is of type 'float'
        elif df[col].dtype == 'float':
            
            # Downcast column to reduce memory usage
            df[col] = pd.to_numeric(df[col], downcast='float')
        
        # If column is of type 'int'
        elif df[col].dtype == 'int':
            
            # Downcast column to reduce memory usage
            df[col] = pd.to_numeric(df[col], downcast='integer')
    
    # Optimize memory usage of datetime columns
    df.select_dtypes(include=['datetime']).apply(pd.Series.astype, dtype='datetime64[ns]')
    

    end_mem = df.memory_usage().sum() / 1024**2
    print('Memory usage after optimization is: {:.2f} MB'.format(end_mem))
    print('Decreased by {:.1f}%'.format(100 * (start_mem - end_mem) / start_mem))
    
    return df


def print_null_percentages(df):
    """
    Prints the null percentages for each column in a pandas dataframe, sorted in descending order
    
    Parameters
    -----
        df: pandas.DataFrame 
            Dataframe to iterate through
        
    Returns
    -----
        None
    """
    
    # replace other null values with NaN
    df = df.replace(['NA', 'null', 'N/A', 'nan'], np.nan)
    
    # calculate null percentages for each column
    null_percentages = df.isna().mean() * 100
    
    # Sort null percentages in descending order
    null_percentages_sorted = null_percentages.sort_values(ascending=True)
    
    # Print out null percentages for each column
    print("Columns that have null values:")
    [print(f"{col}: {percent:.4f}%") for col, percent in null_percentages_sorted.iteritems() if percent > 0]
        
        

def convert_date_columns(df):
    object_cols = df.select_dtypes(include=['object']).columns
    date_cols = [col for col in object_cols if pd.to_datetime(df[col], errors='coerce').notnull().any()]
    df[date_cols] = df[date_cols].apply(pd.to_datetime, format='%Y-%m-%d %H:%M:%S', errors='coerce')
    return df


def convert_data_types(df, LOGGER=logging.getLogger("dummy")):
    """
    Converts all the columns within the input pandas.DataFrame into the correct datatypes.
    
    Parameters
    -----
    df: pandas.DataFrame
        Input DataFrame that has incorrect datatypes
        
    Returns
    -----
    df: pandas.DataFrame
        Output DataFrame with the corrected datatypes in each column
    """
    start_time = time.time()
    print('start_time', time.asctime( time.localtime(start_time) ))
    
    # Get column data types
    dtypes = df.dtypes.to_dict()
    
    # Get boolean column names
    bool_cols = [col for col, dtype in dtypes.items() if dtype == bool]
    
    # Convert boolean columns
    if bool_cols:
        df[bool_cols] = df[bool_cols].astype(bool)
        LOGGER.info(f"Converted the following: {bool_cols} into type Boolean")
    
    # Convert numeric columns
    numeric_cols = [col for col, dtype in dtypes.items() if dtype in (int, float)]
    if numeric_cols:
        df[numeric_cols] = df[numeric_cols].apply(pd.to_numeric, errors='coerce')
        LOGGER.info(f"Converted the following: {numeric_cols} into type Numerical")
    
    # Convert datetime columns
    datetime_cols = [col for col, dtype in dtypes.items() if pd.api.types.is_datetime64_any_dtype(dtype)]
    if datetime_cols:
        df[datetime_cols] = df[datetime_cols].apply(pd.to_datetime, errors='coerce')
        LOGGER.info(f"Converted the following: {datetime_cols} into type Datetime")
    
    # Convert remaining object columns to string
    object_cols = [col for col, dtype in dtypes.items() if dtype == object and col not in bool_cols and col not in numeric_cols and col not in datetime_cols]
    if object_cols:
        df[object_cols] = df[object_cols].astype(str)
        LOGGER.info(f"Converted the following: {object_cols} into type Object")
        
    time_elapsed = time.time() - start_time
    print("Time taken to convert datatypes: {}".format( str(timedelta(seconds=time_elapsed)) ) )
    
    LOGGER.info("Time taken to convert datatypes: {}".format( str(timedelta(seconds=time_elapsed)) ) )
    
    return df


def separate_column_dtypes(df, LOGGER=logging.getLogger("dummy")):
    """
    Categorises each column name into their respective data types to make EDA and feature engineering more efficient.
    
    Parameters
    -----
    df: pandas.DataFrame
        Input DataFrame that requires column dtype categorisation
    
    Returns:
    numeric_cols: List[String]
        A list of all the Numerical column names
    bool_cols: List[String]
        A list of all the Boolean column names
    datetime_cols: List[String]
        A list of all the Datetime column names
    object_cols: List[String]
        A list of all the Object column names
    """
    start_time = time.time()
    print('start_time', time.asctime( time.localtime(start_time) ))
    
    # Get the dtypes of each column
    dtypes = df.dtypes.to_dict()
    
    # Initialize empty lists for each data type
    numeric_cols = [col for col, dtype in dtypes.items() if pd.api.types.is_numeric_dtype(dtype)]
    bool_cols = [col for col, dtype in dtypes.items() if pd.api.types.is_bool_dtype(dtype)]
    datetime_cols = [col for col, dtype in dtypes.items() if pd.api.types.is_datetime64_any_dtype(dtype)]
    object_cols = [col for col, dtype in dtypes.items() if pd.api.types.is_object_dtype(dtype)]
    
    print(f"Columns categorised into the following datatypes:") 
    print(f"----- Numerical ----- \n {numeric_cols}")
    print(f"----- Boolean ----- \n {bool_cols}")
    print(f"----- DateTime ----- \n {datetime_cols}")
    print(f"----- Object ----- \n {object_cols}")
    
    time_elapsed = time.time() - start_time
    print("Time taken to categorises datatypes: {}".format( str(timedelta(seconds=time_elapsed)) ) )
    
    LOGGER.info(f"""Columns categorised into the following datatypes: 
    ----- Numerical -----
    {numeric_cols}
    ----- Boolean -----
    {bool_cols} 
    ----- DateTime -----
    {datetime_cols} 
    ----- Object -----
    {object_cols}""")

    return numeric_cols, bool_cols, datetime_cols, object_cols


def vif_cal(selected_features, df):
    """
    Function calculates VIF values for selected_features
    
    Parameters
    -----
    selected_features : list
        Selected feature names for calculating VIF factors
    df: pandas DataFrame
        Dataframe containing features
        
    Returns
    -----
    vif : DataFrame
        VIF values for each feature
    """
    
    data = [[variance_inflation_factor(df.iloc[:,:].values, i) for i in range(df.shape[1])], selected_features]
    vif = pd.DataFrame(data=data, index=['VIF Factor','Feature']).transpose()
    
    
    return vif


@timeit
def recalc_vif(selected_features, df, vif):
    """
    Function Identifies the highest VIF Factor from the vif table and removes the feature from the list and DataFrame and
    recalculates the VIF Factors
    
    Parameters
    -----
    selected_features : list
        Selected feature names for calculating VIF factors
    df: pandas DataFrame
        Dataframe containing features
    vif : pandas DataFrame
        DataFrame containing the VIF Factor and Features
        
    Returns
    -----
    vif : pandas DataFrame
        VIF values for each feature
    df : pandas DataFrame
        DataFrame containing the modelling dataset
    selected_features : list
        list of features needed for RFECV
    """
    
    feature_to_drop = vif.loc[vif['VIF Factor']== vif['VIF Factor'].max(), 'Feature'].values[0]
    feature_max = vif['VIF Factor'].max()
    df.drop(feature_to_drop, axis=1, inplace=True)
    selected_features.remove(feature_to_drop)
    vif = vif_cal(selected_features, df)
    print("Feature that was dropped: {} ({:.3f})".format(feature_to_drop, feature_max))
    
    return selected_features, df, vif

def rfe_cv(X, y, step, n_splits, params):
    """
    Completes recursive feature elimination with k-fold cross validation. Step features are removed at each round 
    
    Parameters
    -----
    X : pandas DataFrame
        DataFrame containing model input features
    y : pandas Series
        Series containing target feature
    step : int
        Number of features removed at each round 
    n_splits : int
        Number of folds for cross validation
    params : dict
        Training model parameters
        
    Returns : dict
        Cross validation results for each step of features
    -----
    """
    count = 0 
    
    feature_names = list(X.columns)
    n_features = len(X.columns)
    
    results = {}
    
    start_time = time.time()

    while n_features >= 1:
        
        loop_time = time.time()
        
        kf = StratifiedKFold(n_splits=n_splits)
        X = X[feature_names]

        xgb_importance = []
        roc_auc_average = 0
        precision_recall_auc_average = 0
        log_loss_average = 0
        precision_average = 0

        for train_index, test_index in kf.split(X, y):

            X_train_i = X.iloc[train_index]
            y_train_i = y.iloc[train_index]
            dtrain = xgb.DMatrix(X_train_i, y_train_i)
            del X_train_i
            del y_train_i

            X_test_i = X.iloc[test_index]
            y_test_i = y.iloc[test_index]
            dtest = xgb.DMatrix(X_test_i, y_test_i)
            del X_test_i
            del y_test_i

            evallist = [(dtrain, 'train'), (dtest, 'test')]
            del dtrain
            del dtest
            
            xgb_model = xgb.train(params=params, dtrain=evallist[0][0], num_boost_round=100, evals=evallist, early_stopping_rounds=20, maximize=False, verbose_eval=False)
            
            predictions = xgb_model.predict(evallist[1][0])

            # Feature importances
            feature_importances_dict = xgb_model.get_score(importance_type='gain')

            total_importance = sum(feature_importances_dict.values())
            xgb_importance_dict_norm = {k:v/total_importance for k, v in feature_importances_dict.items()}
            xgb_importance.append(xgb_importance_dict_norm)

            # ROC AUC Metric
            roc_auc_average += roc_auc_score(evallist[1][0].get_label(), predictions)

            # Precision & Recall AUC Metric
            precision, recall, thresholds = precision_recall_curve(evallist[1][0].get_label(), predictions)
            precision_recall_auc_average += auc(recall, precision)

            # Log Loss Metric
            log_loss_average += float(xgb_model.attributes()['best_score'])

            # Precision Metric
            precision_average += average_precision_score(evallist[1][0].get_label(), predictions)
            
        xgb_importance_avg = pd.DataFrame(xgb_importance).fillna(0).mean(axis=0).sort_values(ascending=False)
        feature_names = list(xgb_importance_avg.index)

        results[n_features] = {'feature_importance': xgb_importance_avg,
                               'roc_auc_average': roc_auc_average/n_splits,
                               'precision_recall_auc_average': precision_recall_auc_average/n_splits,
                               'log_loss_average': log_loss_average/n_splits,
                               'precision_average': precision_average/n_splits
                              }
        
        print('----- Features: ', n_features, ' -----')
        print('roc_auc_average: ', roc_auc_average/n_splits)
        print('precision_recall_auc_average: ', precision_recall_auc_average/n_splits)
        print('log_loss_average:', log_loss_average/n_splits)
        print('precision_average:', precision_average/n_splits)
        print('step runtime:', time.time() - loop_time)
        print('\n')

        n_features = n_features - step
        feature_names = feature_names[0:n_features] 

    print('Runtime: ', time.time() - start_time)
    
    return results


def model_performance(X, y, params):
    """
    Prints model performance using stratified k-folds
    
    Parameters
    -----
    X : pandas DataFrame
        Input features
    params : dictionary
        Model parameters
    """
    start_time = time.time()

    kf = StratifiedKFold(n_splits=3)

    xgb_importance = []
    roc_auc_average = 0
    precision_recall_auc_average = 0
    log_loss_average = 0
    precision_average = 0

    for train_index, test_index in kf.split(X, y):

        X_train_i = X.iloc[train_index]
        y_train_i = y.iloc[train_index]
        dtrain = xgb.DMatrix(X_train_i, y_train_i)
        del X_train_i
        del y_train_i

        X_test_i = X.iloc[test_index]
        y_test_i = y.iloc[test_index]
        dtest = xgb.DMatrix(X_test_i, y_test_i)
        del X_test_i
        del y_test_i

        evallist = [(dtrain, 'train'), (dtest, 'test')]
        del dtrain
        del dtest

        xgb_model = xgb.train(params=params, dtrain=evallist[0][0], num_boost_round=100, evals=evallist, early_stopping_rounds=20, maximize=False, verbose_eval=False)

        predictions = xgb_model.predict(evallist[1][0])

        # Feature importances
        feature_importances_dict = xgb_model.get_score(importance_type='gain')

        total_importance = sum(feature_importances_dict.values())
        xgb_importance_dict_norm = {k:v/total_importance for k, v in feature_importances_dict.items()}
        xgb_importance.append(xgb_importance_dict_norm)

        # ROC AUC Metric
        roc_auc_average += roc_auc_score(evallist[1][0].get_label(), predictions)

        # Precision & Recall AUC Metric
        precision, recall, thresholds = precision_recall_curve(evallist[1][0].get_label(), predictions)
        precision_recall_auc_average += auc(recall, precision)

        # Log Loss Metric
        log_loss_average += float(xgb_model.attributes()['best_score'])

        # Precision Metric
        precision_average += average_precision_score(evallist[1][0].get_label(), predictions)

    xgb_importance_avg = pd.DataFrame(xgb_importance).fillna(0).mean(axis=0).sort_values(ascending=False)
    feature_names = list(xgb_importance_avg.index)


    print('----- Features: ', len(X.columns), ' -----')
    print('roc_auc_average: ', roc_auc_average/3)
    print('precision_recall_auc_average: ', precision_recall_auc_average/3)
    print('log_loss_average:', log_loss_average/3)
    print('precision_average:', precision_average/3)
    print('\n')

    print('Runtime: ', time.time() - start_time)

