import numpy as np
import sys
import pandas as pd
import visuals as vs # Supplementary code
from sklearn.cross_validation import ShuffleSplit

'exec(%matplotlib inline)'

data = pd.read_csv('housing.csv')
prices = data['MEDV']
features = data.drop('MEDV', axis = 1)
    
from sklearn.metrics import r2_score

def performance_metric(y_true, y_predict):
    score = r2_score(y_true, y_predict)
    return score

from sklearn.cross_validation import train_test_split

X_train, X_test, y_train, y_test = train_test_split(features, prices, test_size=0.2, random_state=10)

from sklearn.metrics import make_scorer
from sklearn.tree import DecisionTreeRegressor
from sklearn.grid_search import GridSearchCV

def fit_model(X, y):
    cv_sets = ShuffleSplit(X.shape[0], n_iter = 10, test_size = 0.20, random_state = 0)
    regressor = DecisionTreeRegressor(random_state=0)
    params = {'max_depth':[1,2,3,4,5,6,7,8,9,10]}
    scoring_fnc = make_scorer(performance_metric)
    grid = GridSearchCV(regressor, params, cv=cv_sets, scoring=scoring_fnc)
    grid = grid.fit(X, y)
    return grid.best_estimator_

reg = fit_model(X_train, y_train)

v1 = sys.argv[1];
v2 = sys.argv[2];
v3 = sys.argv[3];
client_data = [v1,v2,v3]

price = reg.predict([client_data])

print("Predicted Selling Price of a house is : $ %.2f"%price)