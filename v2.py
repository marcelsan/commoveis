import pandas as pd
import numpy as np
from sklearn.utils import shuffle
from sklearn.neighbors import KNeighborsRegressor
from math import acos, sin, cos, radians
from models import *

# PREPROCESSING DATA

def readDataset(file_name, extension = 'csv'):
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
	thresh = np.array([train, train+test]) * data.shape[0]/(train + test + validation)

	train_data = data.iloc[:thresh[0], :].values
	test_data = data.iloc[thresh[0]:thresh[1], :].values
	validation_data = data.iloc[thresh[1]:, :].values

	return train_data, test_data, validation_data
# end

# GEOGRAPHICS FUNCTIONS

def geodesicDistance(A, b):
	# implementação de Robson (raio terrestre aproximado):
	# geo_dist = lambda x, y: 2 * 6372.8 * asin(sqrt(sin(radians((y[0]-x[0])/2)) ** 2 + cos(radians(x[0])) * cos(radians(y[0])) * sin(radians((y[1]-x[1])/2)) ** 2))
	
	# distância com raio terrestre equatorial:
	geo_dist = lambda x, y: acos(sin(radians(x[0]))*sin(radians(y[0]))+cos(radians(x[0]))*cos(radians(y[0]))*cos(radians(x[1]-y[1]))) * 6378.1

	distances = np.array(list(map(lambda x: geo_dist(x, b), A)))

	return distances
# end

# EVALL MODELS

def modelPathLoss(model, distances):
	path_loss = np.vectorize(model.pathLoss)

	return path_loss(distances)
# end

# MAIN FUNCTIONS

def main():

	models = []
	models.append(FreeSpace())
	models.append(OkumuraHata())
	models.append(Cost231Hata())
	models.append(Cost231())
	# models.append(ECC33())
	models.append(Ericsson())
	models.append(Lee())
	models.append(Sui())

	medicoes = readDataset('medicoes')
	# print("READ medicoes.csv")
	erbs = readDataset('erbs')
	# print("READ erbs.csv")
	erbs, _ = splitAttributes(erbs, [0, 1, 4, 5])

	med_coord, rssi = splitAttributes(medicoes, [x for x in range(2, 8)])
	erb_coord, eirp = splitAttributes(pd.DataFrame(erbs), [2])

	distances = np.array(list(map(lambda x: geodesicDistance(erb_coord, x), med_coord)))
	path_loss = np.transpose(eirp) - rssi

	# print(distances)
	# print(path_loss)

	for model in models:
		model_err = modelPathLoss(model, distances) - path_loss
		mean_sqr = np.sqrt(np.mean(model_err ** 2))
		print(type(model).__name__ + ": " + str(mean_sqr))
	# end

# end

main()

