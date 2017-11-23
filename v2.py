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

def coordPoints(size_km = 5e-3):
	lat_lim = [-8.08, -8.065]
	lon_lim = [-34.91, -34.887]

	init_step = 8e4 * size_km

	corner = np.array([lat_lim[0], lon_lim[0]])
	dx = np.array([1e-7, 0])
	dy = np.array([0, 1e-7])
	
	coord_x = corner + init_step*dx

	while geoDist(coord_x, corner) < size_km:
		coord_x += dx
	# end

	d_lat = coord_x[0] - lat_lim[0]

	# print("Distância (km) em latitude: " + str(geodesicDistance(coord_x, corner)[0]))
	# print("Passo em latidude: " + str(d_lat))

	coord_y = coord_x + init_step*dy

	while geoDist(coord_y, coord_x) < size_km:
		coord_y += dy
	# end

	d_lon = coord_y[1] - lon_lim[0]

	# print("Distância (km) em longitude: " + str(geodesicDistance(coord_y, coord_x)[0]))
	# print("Passo em longitude: " + str(d_lon))

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

	for x in range(matrix.shape[0]):
		for y in range(matrix.shape[1]):
			dif_matrix[x, y] = euclideanDist(matrix[x, y], path_loss)
		# end
	# end

	x = dif_matrix.argmin() // dif_matrix.shape[0]
	y = dif_matrix.argmin() %  dif_matrix.shape[1]

	return x, y
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

	i = 0

	coord = med_coord[i]
	path_loss = np.transpose(eirp) - rssi[i]

	x, y = localizeCoordinates(matrix, path_loss)

	print(coord)
	print(lat[x], lon[y])

	error = geoDist(coord, np.array([lat[x], lon[y]]))

	print(error)
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
