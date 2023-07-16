import numpy as np
import pandas as pd
import sklearn.preprocessing
import sklearn.decomposition
import sklearn.model_selection

def train_test_split(  dataset: pd.DataFrame,
                       target_col: str, 
                       test_size: float,
                       stratify: bool,
                       random_state: int) -> tuple[pd.DataFrame,pd.DataFrame,pd.Series,pd.Series]:
    # TODO: Write the necessary code to split a dataframe into a Train and Test feature dataframe and a Train and Test 
    # target series 
    if not stratify:
        X_train, X_test, y_train, y_test = sklearn.model_selection.train_test_split(dataset[[x for x in dataset.columns if x != target_col]], dataset[target_col],test_size=test_size, random_state=random_state)
    else:
        X_train, X_test, y_train, y_test = sklearn.model_selection.train_test_split(dataset[[x for x in dataset.columns if x != target_col]], dataset[target_col],test_size=test_size, stratify=dataset[target_col], random_state=random_state)
    return X_train, X_test, y_train, y_test

class PreprocessDataset:
    def __init__(self, 
                 train_features:pd.DataFrame, 
                 test_features:pd.DataFrame,
                 one_hot_encode_cols:list[str],
                 min_max_scale_cols:list[str],
                 n_components:int,
                 feature_engineering_functions:dict
                 ):
        # TODO: Add any state variables you may need to make your functions work
        self.train_features = train_features
        self.test_features = test_features
        self.one_hot_encode_cols = one_hot_encode_cols
        self.min_max_scale_cols = min_max_scale_cols
        self.n_components = n_components
        self.feature_engineering_functions = feature_engineering_functions
        return
    
    def one_hot_encode_columns_train(self) -> pd.DataFrame:
        # TODO: Write the necessary code to create a dataframe with the categorical column names in 
        # the variable one_hot_encode_cols "one hot" encoded 
        df_ohe = self.train_features[self.one_hot_encode_cols].copy()
        ohe = sklearn.preprocessing.OneHotEncoder(sparse=False)
        one_hot_encoded_dataset = pd.DataFrame(ohe.fit_transform(df_ohe), 
            columns=ohe.get_feature_names_out(self.one_hot_encode_cols),index=self.train_features.index)

        return pd.concat([one_hot_encoded_dataset, self.train_features.drop(self.one_hot_encode_cols, axis=1)], axis=1)

    def one_hot_encode_columns_test(self) -> pd.DataFrame:
        # TODO: Write the necessary code to create a dataframe with the categorical column names in 
        # the variable one_hot_encode_cols "one hot" encoded 
        df_ohe = self.test_features[self.one_hot_encode_cols].copy()
        ohe = sklearn.preprocessing.OneHotEncoder(sparse_output=False, handle_unknown='ignore')
        ohe.fit(self.train_features[self.one_hot_encode_cols])
        one_hot_encoded_dataset = pd.DataFrame(ohe.transform(df_ohe), 
            columns=ohe.get_feature_names_out(self.one_hot_encode_cols), index=self.test_features.index)

        return pd.concat([one_hot_encoded_dataset, self.test_features.drop(self.one_hot_encode_cols, axis=1)], axis=1)
    
    def min_max_scaled_columns_train(self) -> pd.DataFrame:
        # TODO: Write the necessary code to create a dataframe with the numerical column names in 
        # the variable min_max_scale_cols scaled to the min and max of each column 
        df_mms = self.train_features[self.min_max_scale_cols].copy()
        mms = sklearn.preprocessing.MinMaxScaler()
        min_max_scaled_dataset = pd.DataFrame(mms.fit_transform(df_mms), columns=self.min_max_scale_cols,
            index=self.train_features.index)
        
        return pd.concat([min_max_scaled_dataset, self.train_features.drop(self.min_max_scale_cols, axis=1)], axis=1)

    def min_max_scaled_columns_test(self) -> pd.DataFrame:
        # TODO: Write the necessary code to create a dataframe with the numerical column names in 
        # the variable min_max_scale_cols scaled to the min and max of each column 
        df_mms = self.train_features[self.min_max_scale_cols].copy()
        mms = sklearn.preprocessing.MinMaxScaler()
        mms.fit(df_mms)
        min_max_scaled_dataset = pd.DataFrame(mms.transform(self.test_features[self.min_max_scale_cols]), columns=self.min_max_scale_cols, 
            index=self.test_features.index)
        
        return pd.concat([min_max_scaled_dataset, self.test_features.drop(self.min_max_scale_cols, axis=1)], axis=1)
    
    def pca_train(self) -> pd.DataFrame:
        # TODO: use PCA to reduce the train_df to n_components principal components
        # Name your new columns component_1, component_2 .. component_n 
        df = self.train_features.copy()
        df.dropna(inplace=True, axis=1)
        pca = sklearn.decomposition.PCA(n_components=self.n_components, random_state=0)
        cols = ['component_' + str(i) for i in range(1, self.n_components + 1)]
        pca_dataset = pd.DataFrame(pca.fit_transform(df), index=range(df.shape[0]), columns=cols)
        return pca_dataset

    def pca_test(self) -> pd.DataFrame:
        # TODO: use PCA to reduce the test_df to n_components principal components
        # Name your new columns component_1, component_2 .. component_n 
        df_test = self.test_features.copy()
        df_train = self.train_features.copy()
        df_test.dropna(inplace=True, axis=1)
        df_train.dropna(inplace=True, axis=1)
        pca = sklearn.decomposition.PCA(n_components=self.n_components, random_state=0)
        pca.fit(df_train)
        cols = ['component_' + str(i) for i in range(1, self.n_components + 1)]
        pca_dataset = pd.DataFrame(pca.transform(df_test), index=range(df_test.shape[0]), columns=cols)
        return pca_dataset

    def feature_engineering_train(self) -> pd.DataFrame:
        # TODO: Write the necessary code to create a dataframe with feature engineering functions applied 
        # from the feature_engineering_functions dict (the dict format is {'feature_name':function,})
        # each feature engineering function will take in type pd.DataFrame and return a pd.Series
        df = self.train_features.copy()
        for k,v in self.feature_engineering_functions.items():
            df[k] = v(self.train_features)
        feature_engineered_dataset = df
        return feature_engineered_dataset

    def feature_engineering_test(self) -> pd.DataFrame:
        # TODO: Write the necessary code to create a dataframe with feature engineering functions applied 
        # from the feature_engineering_functions dict (the dict format is {'feature_name':function,})
        # each feature engineering function will take in type pd.DataFrame and return a pd.Series
        df = self.test_features.copy()
        for k,v in self.feature_engineering_functions.items():
            df[k] = v(self.test_features)
        feature_engineered_dataset = df
        return feature_engineered_dataset

    def preprocess(self) -> tuple[pd.DataFrame,pd.DataFrame]:
        # TODO: Use the functions you wrote above to create train/test splits of the features and target with scaled and encoded values 
        # for the columns specified in the init function
        self.test_features = self.one_hot_encode_columns_test()
        self.test_features = self.min_max_scaled_columns_test()
        self.test_features = self.feature_engineering_test()
        self.train_features = self.one_hot_encode_columns_train()
        self.train_features = self.min_max_scaled_columns_train()
        self.train_features = self.feature_engineering_train()
        
        return self.train_features, self.test_features
    
if __name__ == '__main__':
    df = pd.read_csv('CLAMP_Train.csv')
    X_train, X_test, y_train, y_test = train_test_split(df, 'class', 0.2, False, 0)
    pre = PreprocessDataset(X_train, X_test, ['e_cp'],['e_cparhdr'],10, {'sqrt':np.sqrt})
