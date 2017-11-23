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

def splitAttributes(data, input_idx, output_idx = None):
	if output_idx == None:
		output_idx = list(filter(lambda x: x not in input_idx, list(range(data.shape[1]))))
	# end

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

# implementação de Robson (raio terrestre aproximado):
# geo_dist = lambda x, y: 2 * 6372.8 * asin(sqrt(sin(radians((y[0]-x[0])/2)) ** 2 + cos(radians(x[0])) * cos(radians(y[0])) * sin(radians((y[1]-x[1])/2)) ** 2))

# distância com raio terrestre equatorial:
geoDist = lambda x, y: acos(sin(radians(x[0]))*sin(radians(y[0]))+cos(radians(x[0]))*cos(radians(y[0]))*cos(radians(x[1]-y[1]))) * 6378.1

euclideanDist = lambda x, y: np.sqrt(np.sum((x-y)**2))

def geodesicDistance(A, b):
	distances = np.array(list(map(lambda x: geoDist(x, b), A)))

	return distances
# end

# EVALL MODELS

def modelPathLoss(model, distances):
	path_loss = np.vectorize(model.pathLoss)

	return path_loss(distances)
# end

# FINGERPRINT FUNCTIONS

def coordPoints(size_km):
	lat_lim = [-8.08, -8.065]
	lon_lim = [-34.91, -34.887]

	left_down = np.array([lat_lim[0], lon_lim[0]])
	left_up = np.array([lat_lim[1], lon_lim[0]])

	right_down = np.array([lat_lim[0], lon_lim[1]])
	right_up = np.array([lat_lim[1], lon_lim[1]])

	# Calcula as variações em graus

	y = max(geoDist(left_down, left_up), geoDist(right_down, right_up))
	x = max(geoDist(left_down, right_down), geoDist(left_up, right_up))

	# print(y/size_km, x/size_km)

	d_lat = (size_km * (lat_lim[1] - lat_lim[0])) / y
	d_lon = (size_km * (lon_lim[1] - lon_lim[0])) / x

	lat = np.linspace(lat_lim[0], lat_lim[1], (lat_lim[1]-lat_lim[0])/d_lat)
	lon = np.linspace(lon_lim[0], lon_lim[1], (lon_lim[1]-lon_lim[0])/d_lon)

	return lat, lon
# end


def erbMatrix(model, erb_pos, lat, lon):
	ones = np.ones(len(lat))
	dist_matrix = np.matrix(list(map(lambda x: geodesicDistance(list(zip(lat, x*ones)), erb_pos), lon)))

	# print("Matriz de perda para ERB montada")

	return modelPathLoss(model, dist_matrix)
# end


def pathLossMatrix(model, erb_coord, grid):
	lat, lon = coordPoints(grid)

	lat = (lat[:-1] + lat[1:]) / 2
	lon = (lon[:-1] + lon[1:]) / 2

	# print("Coordenadas calculadas")

	matrix = list(map(lambda x: erbMatrix(model, x, lat, lon), erb_coord))

	return np.transpose(matrix), lat, lon
# end


def localizeCoordinates(matrix, path_loss):
	dif_matrix = np.ones(matrix.shape[:2])
	print(matrix.shape[:2])

	for lat in range(matrix.shape[0]):
		for lon in range(matrix.shape[1]):
			dif_matrix[lat, lon] = euclideanDist(matrix[lat, lon], path_loss)
		# end
	# end

	print(dif_matrix.size)

	lat_idx = dif_matrix.argmin() // dif_matrix.shape[1]
	lon_idx = dif_matrix.argmin() %  dif_matrix.shape[1]

	print(dif_matrix.argmin())
	print(lat_idx, lon_idx)

	return lat_idx, lon_idx
# end

# TEST FUNCTIONS

def testModels(freq):
	models = []
	models.append(FreeSpace(freq))
	# models.append(OkumuraHata(freq))
	models.append(Cost231Hata(freq))
	models.append(Cost231(freq))
	models.append(ECC33(freq))
	models.append(Ericsson(freq))
	models.append(Lee(freq))
	# models.append(Sui(freq))

	medicoes = readDataset('medicoes')
	# print("READ medicoes.csv")
	erbs = readDataset('erbs')
	# print("READ erbs.csv")

	med_coord, rssi = splitAttributes(medicoes, [0, 1])
	erb_coord, eirp = splitAttributes(erbs, [2, 3], [6])

	distances = np.array(list(map(lambda x: geodesicDistance(erb_coord, x), med_coord)))
	path_loss = np.transpose(eirp) - rssi

	# print(distances)
	# print(path_loss)

	errors = np.array([])

	for model in models:
		model_err = modelPathLoss(model, distances) - path_loss
		mean_sqr = np.sqrt(np.mean(model_err ** 2))
		print(type(model).__name__ + ": " + str(mean_sqr))

		errors = np.append(errors, mean_sqr)
	# end

	return models[errors.argmin()]
# end


def main():
	
	# Leitura dos arquivos de entrada

	medicoes = readDataset('medicoes')
	erbs = readDataset('erbs')

	med_coord, rssi = splitAttributes(medicoes, [0, 1])
	erb_coord, eirp = splitAttributes(erbs, [2, 3], [6])

	# Cálculo da matriz de perdas

	model = Lee(1800)
	grid = 5e-3

	matrix, lat, lon = pathLossMatrix(model, erb_coord, grid)

	# Estima a localização pela matriz

	error = np.array([])

	for i in range(len(med_coord)):
		coord = med_coord[i]
		path_loss = eirp - rssi[i]

		x, y = localizeCoordinates(matrix, path_loss)
		error = np.append(error, geoDist(coord, np.array([lat[x], lon[y]])))
	# end

	print(error.mean())
# end

main()


# x, y = coordPoints(20e-3)	# 20 metros
# # printa as distancias entre pontos para validar o tamanho
# print("LATITUDE - LONGITUDE")
# for i in range(x.size-1):
# 	a = np.array([x[i], y[0]])
# 	b = np.array([x[i+1], y[0]])
# 	c = np.array([x[0], y[i]])
# 	d = np.array([x[0], y[i+1]])
# 	print(geodesicDistance(a, b), geodesicDistance(c, d))
