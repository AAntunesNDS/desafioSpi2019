import numpy as np
import pandas as pd
from sklearn import linear_model
from sklearn import svm
from sklearn.metrics import mean_squared_error
from math import sqrt
from sklearn.model_selection import train_test_split

columns_data = ['CRIM','ZN','INDUS',
                'CHAS','NOX','RM','AGE',
                'DIS','RAD','TAX',
                'PTRATIO','B','LSTAT',
                'MEDV']

def load_dataset():
    df = pd.read_csv('housing.data', header=None, delim_whitespace=True)
    df.columns = columns_data
    return df


def adjust_training_sets(dataframe):

    # label
    y = dataframe['MEDV']

    # features
    X = dataframe.loc[:, dataframe.columns != 'MEDV']

    # division sets
    x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=13)

    return (x_train, x_test, y_train, y_test)



def compute_rmse_regressors():
        
    x_train, x_test, y_train, y_test = adjust_training_sets(load_dataset())

    classifiers = {
        'SVM.Svr' : svm.SVR(),
        'Bayesian Rigde': linear_model.BayesianRidge(),
        'LassoLars' : linear_model.LassoLars(),
        'ARDRegression' : linear_model.ARDRegression(),
        'PassiveAgressiveRegressor' : linear_model.PassiveAggressiveRegressor(),
        'TheilSenRegressor' : linear_model.TheilSenRegressor(),
        'LinearRegression' : linear_model.LinearRegression()
    }

    response = {}

    for classifier_name, classifier in classifiers.items():
        classifier.fit(x_train, y_train)
        y_pred = classifier.predict(x_test)
        response[classifier_name] = sqrt(mean_squared_error(y_pred, y_test))

    return response


def predict_price(features_values):
    
    # convert in np array to predict
    features_values = np.array(features_values)

    # load data_dataset
    df = load_dataset()

    # label
    y = df['MEDV']

    # features
    X = df.loc[:, df.columns != 'MEDV']

    # best regression model
    clf = linear_model.LinearRegression()

    # train mondel with all datas
    clf.fit(X, y)

    # predict and return for new array
    return clf.predict(features_values)
