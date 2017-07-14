from __future__ import print_function
import numpy as np
import scipy
from sklearn import cluster, datasets, mixture
import matplotlib.pyplot as plt
import os
import json

#
# Global variables and change to the data directory
#

OUTPUTSIZE = 80
INPUTPATH = os.getcwd() + "/data/"
OUTPUTPATH = os.getcwd() + "/train"

os.chdir(OUTPUTPATH)

#
# Check if dict.json file already exists
#

if(os.path.isfile("dict.json")):
    with open('dict.json','r') as infile:
        DefectInfo = json.load(infile)
else:
    DefectInfo = dict()

#
# Create a list of input SEM pictures that haven't been processed yet
#

INPUTSEM = []

for SEM in os.listdir(INPUTPATH):
    if not SEM in DefectInfo and SEM.endswith(".tif"):
        INPUTSEM.append(SEM)

#
# Loop through all unprocessed SEM pictures
#

for INPUTFILE in INPUTSEM:

    DefectInfo[str(INPUTFILE[:-4])] = dict()

    #
    # Read the input file
    #

    SEM = scipy.misc.imread(INPUTPATH + INPUTFILE, flatten=True)

    size_x,size_y = SEM.shape

    pos = []

    for x in range(size_x):
        for y in range(size_y):
            if(SEM[x,y]<=20):
                pos.append((x,y))
            # if(SEM[x,y])>20:
            #     im[x,y] = 255
            # else:

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


    for i in range(len(centroids)):
        OUT = INPUTFILE[:-4] + str(int(centroids[i][0])) + str(int(centroids[i][1])) + ".tif"
        xmin = max(int(centroids[i][0]-OUTPUTSIZE/2),0)
        ymin = max(int(centroids[i][1]-OUTPUTSIZE/2),0)
        xmax = min(int(centroids[i][0]+OUTPUTSIZE/2),size_x)
        ymax = min(int(centroids[i][1]+OUTPUTSIZE/2),size_y)

        D = SEM[xmin:xmax,ymin:ymax]

        scipy.misc.imsave(OUT, D)

        INFO = {"x": centroids[i][0],
                "y": centroids[i][1],
                "author": "",
                "type": ""}
        DefectInfo[str(INPUTFILE[:-4])][str(i)] = INFO

with open('dict.json','w') as outfile:
    json.dump(DefectInfo, outfile)
