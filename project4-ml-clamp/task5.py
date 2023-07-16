import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score, precision_score, confusion_matrix, recall_score
from sklearn.feature_selection import RFE
# from task4 import ModelMetrics

def train_model_return_scores(train_df_path,test_df_path) -> pd.DataFrame:
    # TODO: Load and preprocess the train and test dfs
    # Train a sklearn model using training data at train_df_path 
    # Use any sklearn model and return the test index and model scores
    target_col = 'class'
    train = pd.read_csv(train_df_path)
    test = pd.read_csv(test_df_path)
    df = pd.read_csv('CLAMP_Train.csv')

    # train = df.head(int(df.shape[0] * 0.7))
    # test = df.tail(int(df.shape[0] * 0.3))
    for col in train.columns:
        train[col].fillna(train[col].dropna().mean(), inplace=True)
    train.dropna(inplace=True, axis=1)

    for col in test.columns:
        test[col].fillna(test[col].dropna().mean(), inplace=True)
    test.dropna(inplace=True, axis=1)
    X_train = train[[x for x in train.columns if x != target_col]]
    X_test = test[[x for x in test.columns if x != target_col]]
    Y_train = train[target_col]
    # Y_test = test[target_col]
    
    model = RandomForestClassifier()
    model.fit(X_train, Y_train)
    model_pred_bin = model.predict(X_test)
    # TODO: output dataframe should have 2 columns
    # index : this should be the row index of the test df 
    # malware_score : this should be your model's output for the row in the test df
    test_scores = pd.DataFrame(data={'index':X_test.index,'malware_score':model_pred_bin})
    return test_scores 

