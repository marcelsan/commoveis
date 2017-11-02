import pandas as pd
import numpy as np
from sklearn.utils import shuffle
from sklearn.neighbors import KNeighborsRegressor
from math import acos, sin, cos, radians


# parameters
k_list = [2, 3, 5, 7, 9]
perc_val = 0.2


df = pd.read_csv('medicoes.csv')
df = shuffle(df)
X = df.iloc[:, 2:].values
y = df.iloc[:, 0:2].values

part = int(len(X)*perc_val)
X_val = X[:part]
y_val = y[:part]
X = X[part:]
y = y[part:]

geo_dist = lambda a, b: acos(sin(radians(a[0]))*sin(radians(b[0]))+cos(radians(a[0]))*cos(radians(b[0]))*cos(radians(a[1]-b[1]))) * 6378100
err_list = np.array([])

for k in k_list:
	print("K = " + str(k))
	mean = np.array([])

	for i in range(1000):
		neigh = KNeighborsRegressor(n_neighbors = k)

		neigh.fit(X, y)

		dist_err = np.array(list(map(lambda x: geo_dist(x[0], x[1]), zip(neigh.predict(X_val), y_val))))
		err_mean = np.sqrt(np.mean(dist_err*dist_err))
		mean = np.append(mean, err_mean)

	err_list = np.append(err_list, mean.mean())
	print("DONE!")

print(err_list)
print(k_list[err_list.argmin()])

