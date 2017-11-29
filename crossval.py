import pandas as pd
import numpy as np
from sklearn.utils import shuffle
from sklearn.neighbors import KNeighborsRegressor
from sklearn import svm
from math import acos, sin, cos, radians
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler
import seaborn as sns; sns.set(style="ticks", color_codes=True)
import matplotlib.pyplot as plt
from sklearn.model_selection import LeaveOneOut

geo_dist = lambda a, b: acos(sin(radians(a[0]))*sin(radians(b[0]))+cos(radians(a[0]))*cos(radians(b[0]))*cos(radians(a[1]-b[1]))) * 6378.1

df = pd.read_csv('medicoes.csv')
df = shuffle(df)
test = pd.read_csv('testLoc.csv')

pathloss = 55.59 - df.iloc[:, 2:]
pathloss.rename(columns=lambda x: "LOSS_" + x, inplace=True)
result = pd.concat([df, pathloss], axis=1)

X = df.iloc[:, 2:].values
y = df.iloc[:, 0:2].values

mean = []

from sklearn.preprocessing import RobustScaler
from sklearn.model_selection import KFold

kf = KFold(n_splits=10)
for train_index, test_index in kf.split(X):
	X_train, X_test = X[train_index], X[test_index]
	y_train, y_test = y[train_index], y[test_index]

	neigh = KNeighborsRegressor(n_neighbors = 11)
	neigh.fit(X_train, y_train)
	dist_err = np.array(list(map(lambda x: geo_dist(x[0], x[1]), zip(neigh.predict(X_test), y_test))))
	err_mean = np.sqrt(np.mean(dist_err*dist_err))
	print(err_mean)
	mean = np.append(mean, err_mean)
