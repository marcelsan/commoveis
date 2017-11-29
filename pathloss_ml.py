import pandas as pd
import numpy as np
from sklearn.utils import shuffle
from sklearn.neighbors import KNeighborsRegressor
from sklearn import svm
from math import acos, sin, cos, radians
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler

df = pd.read_csv('medicoes.csv')
df = shuffle(df)

pathloss = 55.59 - df.iloc[:, 2:]
pathloss.rename(columns=lambda x: "LOSS_" + x, inplace=True)

X = df.iloc[:, 0:2].values
