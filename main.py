import pandas as pd
import numpy as np
from sklearn.utils import shuffle
from sklearn.neighbors import KNeighborsRegressor
from math import acos, sin, cos, radians

neigh = KNeighborsRegressor(n_neighbors=2)
df = pd.read_csv('medicoes.csv')
df = shuffle(df)
X = df.iloc[:, 2:].values
y = df.iloc[:, 0:2].values

# temporary test
part = int(len(X)/5)
X_test = X[:part]
y_test = y[:part]
X = X[part:]
y = y[part:]

neigh.fit(X, y)

# neigh.predict([[1.5]])
# Implementar função de distância geodésica (pq a terra é redonda) -> Python.
geo_dist = lambda a, b: acos(sin(radians(a[0]))*sin(radians(b[0]))+cos(radians(a[0]))*cos(radians(b[0]))*cos(radians(a[1]-b[1]))) * 6378100
# Usar a função de distância geodésica para calcular erro médio quadrático.
dist_err = np.array(list(map(lambda x: geo_dist(x[0], x[1]), zip(neigh.predict(X_test), y_test))))
sqr_mean = np.sqrt(np.mean(dist_err*dist_err))
print(sqr_mean)

