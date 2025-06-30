from statsmodels.tsa.stattools import adfuller
import pandas as pd

class adf_test:
    def __init__(self, time_series_df):
        self.time_series_df = time_series_df
        self.adf_df = self.adf_df()
        self.adf_tf_df = self.adf_tf_df()

    def adf_df(self):
        adf_dict = {}

        for column in self.time_series_df.columns:
            adf_list = []

            for i in range(3, len(self.time_series_df)):
                try:
                    adf_result = adfuller(self.time_series_df[column].tail(i+1))
                except:
                    continue
                adf_list.append(adf_result[1])

            adf_dict[f'{column}'] = adf_list
        adf_df = pd.DataFrame.from_dict(adf_dict, orient='index').T
        return adf_df

    def adf_score(self):
        adf_scores_numpy = (self.adf_df.index.to_numpy() @ self.adf_df.to_numpy())/self.adf_df.index.to_numpy().sum()
        adf_scores_dict = {self.adf_df.columns[i] : adf_scores_numpy[i] for i in range(len(adf_scores_numpy))}
        adf_scores_series = pd.Series(adf_scores_dict)
        return adf_scores_series

    def adf_tf_score(self, siginificance_level=0.05):
        adf_tf_scores_numpy = (self.adf_tf_df.index.to_numpy() @ self.adf_tf_df.to_numpy())/self.adf_tf_df.index.to_numpy().sum()
        adf_tf_scores_dict = {self.adf_tf_df.columns[i] : adf_tf_scores_numpy[i] for i in range(len(adf_tf_scores_numpy))}
        adf_tf_scores_series = pd.Series(adf_tf_scores_dict)
        return adf_tf_scores_series

    def adf_tf_df(self, siginificance_level=0.05):
        return self.adf_df < siginificance_level