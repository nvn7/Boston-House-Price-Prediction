import numpy as np
import sys
import pandas as pd
import visuals as vs # Supplementary code
from sklearn.cross_validation import ShuffleSplit

'exec(%matplotlib inline)'



data = pd.read_csv('housing.csv')
prices = data['MEDV']
features = data.drop('MEDV', axis = 1)
    

print ('Boston housing dataset has {0} data points with {1} variables each'.format(*data.shape))



minimum_price = np.min(prices)

maximum_price = np.max(prices)

mean_price = np.mean(prices)

median_price = np.median(prices)

std_price = np.std(prices)

print ("Statistics for Boston housing dataset:\n")
print ("Minimum price: ${:,.2f}".format(minimum_price))
print ("Maximum price: ${:,.2f}".format(maximum_price))
print ("Mean price: ${:,.2f}".format(mean_price))
print ("Median price ${:,.2f}".format(median_price))
print ("Standard deviation of prices: ${:,.2f}".format(std_price))


data.drop(data[data['medv'] >= 1050000].index, inplace=True)
data.drop(data[data['rm'] >= 8.70].index, inplace=True)


import matplotlib.pyplot as plt
plt.figure(figsize=(20, 5))


for i, col in enumerate(features.columns):
    plt.subplot(1, 3, i+1)
    x = data[col]
    y = prices
    plt.plot(x, y, 'o')
    plt.plot(np.unique(x), np.poly1d(np.polyfit(x, y, 1))(np.unique(x)))
    plt.title(col)
    plt.xlabel(col)
    plt.ylabel('prices')



from sklearn.metrics import r2_score

def performance_metric(y_true, y_predict):   
    score = r2_score(y_true, y_predict)
    return score

	
from sklearn.cross_validation import train_test_split


X_train, X_test, y_train, y_test = train_test_split(features, prices, test_size=0.2, random_state=10)

print ("Training and testing split was successful.")

print(features.shape[0])
print(float(X_train.shape[0]) / float(features.shape[0]))
print(float(X_test.shape[0]) / float(features.shape[0]))

vs.ModelLearning(features, prices)

vs.ModelComplexity(X_train, y_train)

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

print ("Parameter 'max_depth' is {} for the optimal model.".format(reg.get_params()['max_depth']))

reg.get_params()['max_depth']



v1 = sys.argv[1];
v2 = sys.argv[2];
v3 = sys.argv[3];
client_data = [v1,v2,v3]

price = reg.predict([client_data])

print("Predicted Selling Price of a house is : $ %.2f"%price)
