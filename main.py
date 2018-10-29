import numpy as np
import sys
import pandas as pd
import visuals as vs 
from sklearn.cross_validation import ShuffleSplit

'exec(%matplotlib inline)'



data = pd.read_csv('boston.csv')
data=data.drop('medv_',axis=1)
data=data.drop('id',axis=1)






prices = data['medv']
features = data.drop('medv', axis = 1)
    

print ('Boston housing dataset has {0} data points with {1} variables each'.format(*data.shape))

#preprocessing

minimum_price = np.min(prices)

maximum_price = np.max(prices)

mean_price = np.mean(prices)

median_price = np.median(prices)

std_price = np.std(prices)
#preprocessing of medv
first_quartile = np.percentile(prices, 25)
third_quartile = np.percentile(prices, 75)
inter_quartile = third_quartile - first_quartile

lo = first_quartile - (1.5 * inter_quartile)
ho = third_quartile + (1.5 * inter_quartile)

print ("Statistics for Boston housing dataset:\n")
print ("Minimum price: ${:,.2f}".format(minimum_price))
print ("Maximum price: ${:,.2f}".format(maximum_price))
print ("Mean price: ${:,.2f}".format(mean_price))
print ("Median price ${:,.2f}".format(median_price))
print ("Standard deviation of prices: ${:,.2f}".format(std_price))
print ("First quartile of prices: ${:,.2f}".format(first_quartile))
print ("Second quartile of prices: ${:,.2f}".format(third_quartile))
print ("Interquartile (IQR) of prices: ${:,.2f}".format(inter_quartile))
print ("First quartile of prices: ${:,.2f}".format(first_quartile))
print ("Second quartile of prices: ${:,.2f}".format(third_quartile))
print ("Interquartile (IQR) of prices: ${:,.2f}".format(inter_quartile))
print ("Lower Outlier value: ${:,.2f}".format(lo))
print ("Higher Outlier value: ${:,.2f}".format(ho))

data.drop(data[data['medv'] > ho ].index, inplace=True)
#preprocessing of rm
rm = data['rm']
first_quartile = np.percentile(rm, 25)
third_quartile = np.percentile(rm, 75)
inter_quartile = third_quartile - first_quartile

lo = first_quartile - (1.5 * inter_quartile)
ho = third_quartile + (1.5 * inter_quartile)
print ("Lower Outlier value of rm: ${:,.2f}".format(lo))
print ("Higher Outlier value of rm: ${:,.2f}".format(ho))

data.drop(data[data['rm'] >= ho].index, inplace=True)

#preprocessing of lstat
lstat = data['lstat']
first_quartile = np.percentile(lstat, 25)
third_quartile = np.percentile(lstat, 75)
inter_quartile = third_quartile - first_quartile

lo = first_quartile - (1.5 * inter_quartile)
ho = third_quartile + (1.5 * inter_quartile)
print ("Lower Outlier value of lstat: ${:,.2f}".format(lo))
print ("Higher Outlier value of lstat: ${:,.2f}".format(ho))

data.drop(data[data['lstat'] >= ho].index, inplace=True)


#preprocessing of nox
nox = data['nox']
first_quartile = np.percentile(nox, 25)
third_quartile = np.percentile(nox, 75)
inter_quartile = third_quartile - first_quartile

lo = first_quartile - (1.5 * inter_quartile)
ho = third_quartile + (1.5 * inter_quartile)
print ("Lower Outlier value of nox: ${:,.2f}".format(lo))
print ("Higher Outlier value of nox: ${:,.2f}".format(ho))

data.drop(data[data['nox'] >= ho].index, inplace=True)

prices = data['medv']
features = data.drop('medv', axis = 1)

import matplotlib.pyplot as plt
plt.figure(figsize=(20, 5))


for i, col in enumerate(features.columns):
    plt.subplot(1, 14, i+1)
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



v1 = sys.argv[8]
v2 = sys.argv[9]
v3 = sys.argv[10]
v4 = sys.argv[11]
v5 = sys.argv[12]
v6 = sys.argv[1]
v7 = sys.argv[2]
v8 =sys.argv[3]
v9 =sys.argv[4]
v10 = sys.argv[5]
v11 = sys.argv[6]
v12 = sys.argv[13]
v13 = sys.argv[7]
client_data = [v1,v2,v3,v4,v5,v6,v7,v8,v9,v10,v11,v12,v13]

price = reg.predict([client_data])
print ("R2_Score is {} for our optimal model.".format(reg.score(X_train,y_train)))

print("Predicted Selling Price of your house is : $ %.2f"%price)
