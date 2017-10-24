import pandas as pd
import numpy as np
from sklearn.utils import shuffle
from sklearn.neighbors import KNeighborsRegressor

neigh = KNeighborsRegressor(n_neighbors=2)
df = pd.read_csv('medicoes.csv')
df = shuffle(df)
X = df.iloc[:, 2:].values
y = df.iloc[:, 0:2].values
neigh.fit(X, y)


# neigh.predict([[1.5]])
# Implementar função de distância geodésica (pq a terra é redonda) -> Python.
# Usar a função de distância geodésica para calcular erro médio quadrático.