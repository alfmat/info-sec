import pandas as pd
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score, precision_score, confusion_matrix, recall_score
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.feature_selection import RFE

class ModelMetrics:
    def __init__(self, model_type:str,train_metrics:dict,test_metrics:dict,feature_importance_df:pd.DataFrame):
        self.model_type = model_type
        self.train_metrics = train_metrics
        self.test_metrics = test_metrics
        self.feat_imp_df = feature_importance_df
        self.feat_name_col = "Feature"
        self.imp_col = "Importance"


    def add_train_metric(self,metric_name:str,metric_val:float):
        self.train_metrics[metric_name] = metric_val

    def add_test_metric(self,metric_name:str,metric_val:float):
        self.test_metrics[metric_name] = metric_val
    
    def calc_naive_metrics(self, assume: int, target_col:str, train_dataset:pd.DataFrame, test_dataset:pd.DataFrame):
        guess_train = pd.Series([assume] * train_dataset.shape[0])
        guess_test = pd.Series([assume] * test_dataset.shape[0])
        actual_train = train_dataset[target_col]
        actual_test = test_dataset[target_col]

        self.train_metrics['accuracy'] = round(accuracy_score(actual_train, guess_train), 4)
        self.train_metrics['recall'] = round(recall_score(actual_train, guess_train),4)
        self.train_metrics['precision'] = round(precision_score(actual_train, guess_train),4)
        self.train_metrics['fscore'] = round(f1_score(actual_train, guess_train),4)

        self.test_metrics['accuracy'] = round(accuracy_score(actual_test, guess_test),4)
        self.test_metrics['recall'] = round(recall_score(actual_test, guess_test),4)
        self.test_metrics['precision'] = round(precision_score(actual_test, guess_test),4)
        self.test_metrics['fscore'] = round(f1_score(actual_test, guess_test),4)

    def calc_model_metrics(self, model: LogisticRegression | RandomForestClassifier | GradientBoostingClassifier, train_dataset:pd.DataFrame, test_dataset:pd.DataFrame, target_col:str):
        guess_train_proba = model.predict_proba(train_dataset[[x for x in train_dataset.columns if x != target_col]])
        guess_test_proba = model.predict_proba(test_dataset[[x for x in test_dataset.columns if x != target_col]])
        guess_train_bin = model.predict(train_dataset[[x for x in train_dataset.columns if x != target_col]])
        guess_test_bin = model.predict(test_dataset[[x for x in test_dataset.columns if x != target_col]])
        actual_train = train_dataset[target_col]
        actual_test = test_dataset[target_col]

        self.train_metrics['accuracy'] = round(accuracy_score(actual_train, guess_train_bin),4)
        self.train_metrics['recall'] = round(recall_score(actual_train, guess_train_bin),4)
        self.train_metrics['precision'] = round(precision_score(actual_train, guess_train_bin),4)
        self.train_metrics['fscore'] = round(f1_score(actual_train, guess_train_bin),4)
        self.train_metrics['roc_auc'] = round(roc_auc_score(actual_train, guess_train_proba[:,1]),4)

        TN, FP, FN, TP = confusion_matrix(actual_train, guess_train_bin).ravel()
        
        FPR = FP/(FP+TN)
        FNR = FN/(TP+FN)

        self.train_metrics['fpr'] = round(FPR,4)
        self.train_metrics['fnr'] = round(FNR,4)


        self.test_metrics['accuracy'] = round(accuracy_score(actual_test, guess_test_bin),4)
        self.test_metrics['recall'] = round(recall_score(actual_test, guess_test_bin),4)
        self.test_metrics['precision'] = round(precision_score(actual_test, guess_test_bin),4)
        self.test_metrics['fscore'] = round(f1_score(actual_test, guess_test_bin),4)
        self.test_metrics['roc_auc'] = round(roc_auc_score(actual_test, guess_test_proba[:,1]),4)

        TN, FP, FN, TP = confusion_matrix(actual_test, guess_test_bin).ravel()
        
        FPR = FP/(FP+TN)
        FNR = FN/(TP+FN)

        self.test_metrics['fpr'] = round(FPR,4)
        self.test_metrics['fnr'] = round(FNR,4)
        pass

    def __str__(self): 
        output_str = f"MODEL TYPE: {self.model_type}\n"
        output_str += f"TRAINING METRICS:\n"
        for key in sorted(self.train_metrics.keys()):
            output_str += f"  - {key} : {self.train_metrics[key]:.4f}\n"
        output_str += f"TESTING METRICS:\n"
        for key in sorted(self.test_metrics.keys()):
            output_str += f"  - {key} : {self.test_metrics[key]:.4f}\n"
        if self.feat_imp_df is not None:
            output_str += f"FEATURE IMPORTANCES:\n"
            for i in self.feat_imp_df.index:
                output_str += f"  - {self.feat_imp_df[self.feat_name_col][i]} : {self.feat_imp_df[self.imp_col][i]:.4f}\n"
        return output_str


def calculate_naive_metrics(train_dataset:pd.DataFrame, test_dataset:pd.DataFrame, target_col:str, naive_assumption:int) -> ModelMetrics:
    # TODO: Write the necessary code to calculate accuracy, recall, precision and fscore given a train and test dataframe
    # and a train and test target series and naive assumption 
    train_metrics = {
        "accuracy" : 0,
        "recall" : 0,
        "precision" : 0,
        "fscore" : 0
        }
    test_metrics = {
        "accuracy" : 0,
        "recall" : 0,
        "precision" : 0,
        "fscore" : 0
        }
    naive_metrics = ModelMetrics("Naive",train_metrics,test_metrics,None)
    naive_metrics.calc_naive_metrics(naive_assumption, target_col, train_dataset, test_dataset)
    return naive_metrics

def calculate_logistic_regression_metrics(train_dataset:pd.DataFrame, test_dataset:pd.DataFrame, target_col:str, logreg_kwargs) -> tuple[ModelMetrics,LogisticRegression]:
    # TODO: Write the necessary code to train a logistic regression binary classifiaction model and calculate accuracy, recall, precision, fscore, 
    # false positive rate, false negative rate and area under the reciever operator curve given a train and test dataframe and train and test target series 
    # and keyword arguments for the logistic regrssion model
    model = LogisticRegression(**logreg_kwargs)
    train_metrics = {
        "accuracy" : 0,
        "recall" : 0,
        "precision" : 0,
        "fscore" : 0,
        "fpr" : 0,
        "fnr" : 0,
        "roc_auc" : 0
        }
    test_metrics = {
        "accuracy" : 0,
        "recall" : 0,
        "precision" : 0,
        "fscore" : 0,
        "fpr" : 0,
        "fnr" : 0,
        "roc_auc" : 0
        }
    # TODO: Use RFE to select the top 10 features 
    # make sure the column of feature names is named Feature
    # and the column of importances is named Importance 
    # and the dataframe is sorted by ascending ranking then decending absolute value of Importance
    rfe_model = RFE(model, n_features_to_select=10, verbose=0)
    rfe_model.fit(X=train_dataset[[x for x in train_dataset.columns if x != target_col]], y=train_dataset[target_col])
    log_reg_importance = pd.DataFrame(data={'Feature': rfe_model.get_feature_names_out(), 'Importance': rfe_model.estimator_.coef_[0]})
    log_reg_importance.sort_values(inplace=True, ascending=False, by=['Importance'])
    log_reg_importance.reset_index(inplace=True)

    log_reg_metrics = ModelMetrics("Logistic Regression",train_metrics,test_metrics,log_reg_importance)
    model.fit(X=train_dataset[[x for x in train_dataset.columns if x != target_col]], y=train_dataset[target_col])
    log_reg_metrics.calc_model_metrics(model, train_dataset, test_dataset, target_col)
    return log_reg_metrics,model

def calculate_random_forest_metrics(train_dataset:pd.DataFrame, test_dataset:pd.DataFrame, target_col:str, rf_kwargs) -> tuple[ModelMetrics,RandomForestClassifier]:
    # TODO: Write the necessary code to train a random forest binary classification model and calculate accuracy, recall, precision, fscore, 
    # false positive rate, false negative rate and area under the reciever operator curve given a train and test dataframe and train and test 
    # target series and keyword arguments for the random forest model
    model = RandomForestClassifier(**rf_kwargs)
    train_metrics = {
        "accuracy" : 0,
        "recall" : 0,
        "precision" : 0,
        "fscore" : 0,
        "fpr" : 0,
        "fnr" : 0,
        "roc_auc" : 0
        }
    test_metrics = {
        "accuracy" : 0,
        "recall" : 0,
        "precision" : 0,
        "fscore" : 0,
        "fpr" : 0,
        "fnr" : 0,
        "roc_auc" : 0
        }
    # TODO: Reminder DONT use RFE for rf_importance
    # make sure the column of feature names is named Feature
    # and the column of importances is named Importance 
    # and the dataframe is sorted by decending absolute value of Importance
    model.fit(X=train_dataset[[x for x in train_dataset.columns if x != target_col]], y=train_dataset[target_col])
    rf_importance = pd.DataFrame(data={'Feature': model.feature_names_in_, 'Importance':model.feature_importances_})
    rf_importance.sort_values(inplace=True, ascending=False,by=['Importance'])
    rf_importance = rf_importance.head(10)
    rf_importance.reset_index(inplace=True)
    rf_metrics = ModelMetrics("Random Forest",train_metrics,test_metrics,rf_importance)
    rf_metrics.calc_model_metrics(model, train_dataset, test_dataset, target_col)
    return rf_metrics,model

def calculate_gradient_boosting_metrics(train_dataset:pd.DataFrame, test_dataset:pd.DataFrame, target_col:str, gb_kwargs) -> tuple[ModelMetrics,GradientBoostingClassifier]:
    # TODO: Write the necessary code to train a gradient boosting binary classification model and calculate accuracy, recall, precision, fscore, 
    # false positive rate, false negative rate and area under the reciever operator curve given a train and test dataframe and train and test 
    # target series and keyword arguments for the gradient boosting model
    model = GradientBoostingClassifier(**gb_kwargs)
    train_metrics = {
        "accuracy" : 0,
        "recall" : 0,
        "precision" : 0,
        "fscore" : 0,
        "fpr" : 0,
        "fnr" : 0,
        "roc_auc" : 0
        }
    test_metrics = {
        "accuracy" : 0,
        "recall" : 0,
        "precision" : 0,
        "fscore" : 0,
        "fpr" : 0,
        "fnr" : 0,
        "roc_auc" : 0
        }
    # TODO: Reminder DONT use RFE for gb_importance
    # make sure the column of feature names is named Feature
    # and the column of importances is named Importance 
    # and the dataframe is sorted by decending absolute value of Importance
    model.fit(X=train_dataset[[x for x in train_dataset.columns if x != target_col]], y=train_dataset[target_col])
    gb_importance = pd.DataFrame(data={'Feature': model.feature_names_in_, 'Importance':model.feature_importances_})
    gb_metrics = ModelMetrics("Gradient Boosting",train_metrics,test_metrics,gb_importance)
    gb_importance.sort_values(inplace=True, ascending=False,by=['Importance'])
    gb_importance = gb_importance.head(10)
    gb_importance.reset_index(inplace=True)
    gb_metrics = ModelMetrics("Gradient Boosting",train_metrics,test_metrics,gb_importance)
    gb_metrics.calc_model_metrics(model, train_dataset, test_dataset, target_col)
    return gb_metrics,model

# if __name__ == '__main__':
#     df = pd.read_csv('CLAMP_Train.csv')
#     for col in df.columns:
#         df[col].fillna(df[col].dropna().mean(), inplace=True)
#     df.dropna(inplace=True, axis=1)
#     train_dataset = df.head(int(df.shape[0] * 0.8))
#     test_dataset = df.tail(int(df.shape[0] * 0.2))
#     calculate_logistic_regression_metrics(train_dataset, test_dataset, 'class', {})