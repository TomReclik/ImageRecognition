from __future__ import print_function
import numpy as np
import scipy
from sklearn import cluster, datasets, mixture
import matplotlib.pyplot as plt
import os

#
# Global variables and change to the data directory
#

OUTPUTSIZE = 80
INPUTFILE = "TP_7deformed_damage_0.tif"
PATH = os.getcwd() + "/data"

os.chdir(PATH)

#
# Read the input file
#

im = scipy.misc.imread(INPUTFILE, flatten=True)

size_x,size_y = im.shape

pos = []

for x in range(size_x):
    for y in range(size_y):
        if(im[x,y])>20:
            im[x,y] = 255
        else:
            pos.append((x,y))

positions = np.array(pos)

#
# Find defects and calculate their centroids
#

dbscan_dataset1 = cluster.DBSCAN(eps=4, min_samples=10, metric='euclidean').fit_predict(positions)

centroids = np.zeros((len(set(dbscan_dataset1))-1,2))

for i in range(len(dbscan_dataset1)):
    if dbscan_dataset1[i]!=-1:
        centroids[dbscan_dataset1[i]] = centroids[dbscan_dataset1[i]] + positions[i]

for i in range(len(centroids)):
        centroids[i] = centroids[i] / sum(dbscan_dataset1==i)

#
# Save defects into seperate files
#

SEM = scipy.misc.imread(INPUTFILE, flatten=True)

for i in range(len(centroids)):
    OUT = str(i) + ".tif"
    xmin = max(int(centroids[i][0]-OUTPUTSIZE/2),0)
    ymin = max(int(centroids[i][1]-OUTPUTSIZE/2),0)
    xmax = min(int(centroids[i][0]+OUTPUTSIZE/2),size_x)
    ymax = min(int(centroids[i][1]+OUTPUTSIZE/2),size_y)

    D = SEM[xmin:xmax,ymin:ymax]
    scipy.misc.imsave(OUT, D)
