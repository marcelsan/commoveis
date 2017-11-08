import pandas as pd
import numpy as np
from sklearn.utils import shuffle
from sklearn.neighbors import KNeighborsRegressor
from math import acos, sin, cos, radians

# PREPROCESSING

def readDataset(file_name = 'dataset', extension = 'csv'):
	dataset = None

	switcher = {
		'csv': pd.read_csv(file_name + '.csv')
	}

	dataset = switcher.get(extension.lower(), pd.read_csv(file_name + '.csv'))

	return dataset
# end

def splitAttributes(data, output_idx):
	input_idx = list(filter(lambda x: x not in output_idx, list(range(data.shape[1]))))

	X = data.iloc[:, input_idx].values
	y = data.iloc[:, output_idx].values

	return X, y
# end

def splitDataset(data, train, test, validation = 0):
	thresh = np.array([train, validation]) * data.shape[0]/(train + test + validation)

	train_data = data.iloc[:thresh[0], :].values
	test_data = data.iloc[thresh[0]:thresh[1], :].values
	validation_data = data.iloc[thresh[1]:, :].values

	return train_data, test_data, validation_data
# end

# EVALL PERFORMANCE


