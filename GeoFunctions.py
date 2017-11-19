from math import *

def geo_dist_julia(a, b):
    return 2 * 6372.8 * asin(sqrt(sin(radians((b[0]-a[0])/2)) ** 2 + cos(radians(a[0])) * cos(radians(b[0])) * sin(radians((b[1]-a[1])/2)) ** 2))

def geo_dist (a, b): 
    return acos(sin(radians(a[0]))*sin(radians(b[0]))+cos(radians(a[0]))*cos(radians(b[0]))*cos(radians(a[1]-b[1]))) * 6378.1
