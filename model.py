import numpy as np
import sys
import pandas as pd
import visuals as vs # Supplementary code
from sklearn.cross_validation import ShuffleSplit

'exec(%matplotlib inline)'

data = pd.read_csv('boston.csv')
data=data.drop('medv_',axis=1)
data=data.drop('id',axis=1)


data.drop(data[data['medv'] == 1050000].index, inplace=True)
data.drop(data[data['rm'] == 8.78].index, inplace=True)


prices = data['medv']
features = data.drop('medv', axis = 1)

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
# sys.argv[1];
v1 = 0.027
v2 = 13
v3 = 7.87
v4 = 0
v5 = 0.538
v6 = sys.argv[1]
v7 = sys.argv[2]
v8 =sys.argv[3]
v9 =sys.argv[4]
v10 = sys.argv[5]
v11 = sys.argv[6]
v12 = 390
v13 = sys.argv[7]
client_data = [v1,v2,v3,v4,v5,v6,v7,v8,v9,v10,v11,v12,v13]

price = reg.predict([client_data])

print("Predicted Selling Price of a house is : $ %.2f"%price)