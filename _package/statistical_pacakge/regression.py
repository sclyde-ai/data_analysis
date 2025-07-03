from sklearn.linear_model import LinearRegression, Ridge, Lasso, RidgeCV, LassoCV
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import seaborn as sns
from statsmodels.stats.outliers_influence import variance_inflation_factor
from statsmodels.tools.tools import add_constant
from sklearn.metrics import mean_squared_error, mean_absolute_error
import pandas as pd
import numpy as np

class my_regression():
    def __init__(self, df, explanatory_columns, dependent_columns):
        self.df = df
        self.X = df[explanatory_columns]
        self.Y = df[dependent_columns]
        self.X_row = df[explanatory_columns]
        self.Y_row = df[dependent_columns]
        self.model = LinearRegression()
        self.dummy = []
        self.exlainatory = explanatory_columns
        self.dependent = dependent_columns

        self.X_train = None
        self.X_test = None
        self.Y_train = None 
        self.Y_test = None
        self.X_all = None
        self.Y_all = None

    # data processing
    def with_dummy(self, dummy_columns):
        df_dummy = pd.get_dummies(self.df[dummy_columns], dtype=int)
        self.X = self.X.join(df_dummy)
        self.dummy = dummy_columns
        return self
    
    def standard(self):
        X_index = self.X.index
        Y_index = self.Y.index
        X_columns = self.X.columns
        Y_name = self.Y.name 
        # print(type(self.X))
        # print(type(self.Y))
        if (X_index.all() == Y_index.all()):
            X = preprocessing.scale(self.X)
            Y = preprocessing.scale(self.Y)
            self.X = pd.DataFrame(X, index=X_index, columns=X_columns)
            self.Y = pd.Series(Y, index=Y_index, name=Y_name)
            # print(type(self.X))
            # print(type(self.Y))
            # print(self.X.columns)
            # print(self.Y.columns)
        else:
            print('error')
        return self
    
    def reset(self):
        self.X = self.X_row
        self.Y = self.Y_row
        self.X_train = None
        self.X_test = None
        self.Y_train = None 
        self.Y_test = None
        self.X_all = None
        self.Y_all = None
        return self 
    
    def split(self):
        X_train, X_test, Y_train, Y_test = train_test_split(self.X, self.Y, test_size=0.3, random_state=42)
        self.X_train = X_train
        self.X_test = X_test
        self.Y_train = Y_train
        self.Y_test = Y_test
        return self
    
    # fitting
    def fit(self):
        self.model.fit(self.X,self.Y)

    def train(self):
        self.X = self.X_train
        self.Y = self.Y_train
        self.model.fit(self.X_train, self.Y_train)
        return self
    
    # selecting model
    def linear(self):
        self.model = LinearRegression()
        return self
    
    def ridge(self, alpha):
        if isinstance(alpha, list):
            self.model = RidgeCV(alpha=alpha)
        else:
            self.model = Ridge(alpha=alpha)
        return self
    
    def lasso(self, alpha):
        if isinstance(alpha, list):
            self.model = LassoCV(alpha=alpha)
        else:
            self.model = Lasso(alpha=alpha)
        return self

    # put parameters
    def intercept(self):
        return self.model.intercept_
    
    def R_2(self):
        return self.model.score(self.X, self.Y)
    
    def adjusted_R_2(self):
        adjusted_R_2 = 1 - ((1-self.model.score(self.X, self.Y))*(self.X.shape[0]-1)/(self.X.shape[0] - self.X.shape[1]-1))
        return adjusted_R_2
    
    def log_likelihood(self):
        residuals = self.Y - self.model.predict(self.X)
        sigma2 = np.var(residuals) 
        log_likelihood = -0.5 * len(self.Y) * np.log(2 * np.pi * sigma2) - (0.5 / sigma2) * np.sum(residuals**2)
        return log_likelihood
    
    def aic(self):
        aic = -2 * self.log_likelihood() + 2 * (self.X.shape[1] + 1)
        return aic
    
    def bic(self):
        bic = -2 * self.log_likelihood() + np.log(len(self.Y)) * (self.X.shape[1] + 1)
        return bic
    
    def mse(self):
        self.train()
        Y_pred = self.model.predict(self.X_test)
        mse = mean_squared_error(self.Y_test, Y_pred)
        return mse
    
    def mae(self):
        self.train()
        Y_pred = self.model.predict(self.X_test)
        mae = mean_absolute_error(self.Y_test, Y_pred)
        return mae
    
    def coef(self):
        coef_matrix = self.model.coef_
        coef_series = pd.Series(coef_matrix, index=self.X.columns)
        print(coef_series)
        return coef_series
    
    def vif(self):
        X = add_constant(self.X)
        vif = pd.DataFrame()
        vif["Variable"] = X.columns
        vif["VIF"] = [variance_inflation_factor(X.values, i) for i in range(X.shape[1])]
        return vif
    
    def corr_heatmap(self):
        corr = self.X.join(self.Y).corr()
        plt.figure()
        sns.heatmap(corr)
        plt.show()
        plt.close()
        return corr
    
    def coef_hist(self):
        plt.figure()
        plt.hist(np.abs(self.model.coef_))
        plt.show()
        plt.close()
        return

    def check(self):
        self.fit()
        print('expalainatory:', self.exlainatory)
        print('dependent:', self.dependent)
        print('dummy:', self.dummy)
        print('intercept:', self.model.intercept_)
        print('R²:', self.model.score(self.X, self.Y))
        print('log likelihood', self.log_likelihood())
        print('aic', self.aic())
        print('bic', self.bic())
        # print('vif', self.vif())
        print()

    def check_series(self):
        check_dict = {}
        check_dict['explanatory'] = self.exlainatory + self.dummy
        check_dict['R_2'] = self.R_2()
        check_dict['adjusted_R_2'] = self.adjusted_R_2()
        check_dict['aic'] = self.aic()
        check_dict['bic'] = self.bic()
        if self.X.all().all() == self.X_train.all().all():
            check_dict['mse'] = self.mse()
            check_dict['mae'] = self.mae()
        return pd.Series(check_dict)

    def split_check_series(self):
        self.train()
        train_series = self.check_series(self)
        self.test()
        test_series = self.check_series(self)
        return pd.merge(train_series.to_frame(), test_series.to_frame(), suffixes=('_train', '_test'))
