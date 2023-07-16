import numpy as np
import pandas as pd
import sklearn.cluster
import yellowbrick.cluster

class KmeansClustering:
    def __init__(self, 
                 train_features:pd.DataFrame,
                 test_features:pd.DataFrame,
                 random_state: int
                ):
        # TODO: Add any state variables you may need to make your functions work
        self.train_features = train_features
        self.test_features = test_features
        self.random_state = random_state

    def kmeans_train(self) -> list:
        # TODO: train a kmeans model using the training data, determine the optimal value of k (between 1 and 10) with n_init set to 10 and return a list of cluster ids 
        # corresponding to the cluster id of each row of the training data
        base_model = sklearn.cluster.KMeans(random_state=self.random_state)
        kelbow = yellowbrick.cluster.KElbowVisualizer(base_model, k=(1,10))
        kelbow.fit(self.train_features)
        kmeans = sklearn.cluster.KMeans(random_state=self.random_state, n_init=10, n_clusters=kelbow.elbow_value_)
        kmeans.fit(self.train_features)
        return kmeans.predict(self.train_features).tolist()

    def kmeans_test(self) -> list:
        # TODO: return a list of cluster ids corresponding to the cluster id of each row of the test data
        base_model = sklearn.cluster.KMeans(random_state=self.random_state)
        kelbow = yellowbrick.cluster.KElbowVisualizer(base_model, k=(1,10))
        kelbow.fit(self.train_features)
        kmeans = sklearn.cluster.KMeans(random_state=self.random_state, n_init=10, n_clusters=kelbow.elbow_value_)
        kmeans.fit(self.train_features)
        return kmeans.predict(self.test_features).tolist()

    def train_add_kmeans_cluster_id_feature(self) -> pd.DataFrame:
        # TODO: return the training dataset with a new feature called kmeans_cluster_id
        output_df = self.train_features.copy()
        output_df['kmeans_cluster_id'] = self.kmeans_train()
        return output_df

    def test_add_kmeans_cluster_id_feature(self) -> pd.DataFrame:
        # TODO: return the test dataset with a new feature called kmeans_cluster_id
        output_df = self.test_features.copy()
        output_df['kmeans_cluster_id'] = self.kmeans_test()
        return output_df