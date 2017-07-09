from __future__ import print_function
import numpy as np
import scipy
from sklearn import cluster, datasets, mixture
import matplotlib.pyplot as plt

"""
Read image with Pillow
"""
# from PIL import Image, ImageFilter
# im = Image.open("TP_7deformed_damage_0.tif")
# size = im.size
#
# defect = im.point(lambda i: i>20 and 255)
# # defect.save("Defect4.tif")
#
# size = defect.size
#
# defect.show()

INPUTFILE = "TP_7deformed_damage_01.tif"

"""
Read image with scipy
"""

def cluster_plot(set, colours = 'gray'):
    plt.gca().invert_yaxis()
    plt.scatter(set[:,1],set[:,0],s=4,lw=0.1, c=colours)
    plt.show()

def kernel(mat,breakup_number):
    """
    Decides if pixel remains black depending on how many pixels are black
    in its vicinity
    """
    num=0
    len_x,len_y=mat.shape
    for x in range(len_x):
        for y in range(len_y):
            if mat[x,y]!=0:
                pass
            else:
                num=num+1
    if num<breakup_number:
        return 0
    else:
        return 255

im = scipy.misc.imread(INPUTFILE, flatten=True)

size_x,size_y = im.shape

black = lambda p: p>0 and 255

pos = []

for x in range(size_x):
    for y in range(size_y):
        if(im[x,y])>0:
            im[x,y] = 255
        else:
            pos.append((x,y))

positions = np.array(pos)

dbscan_dataset1 = cluster.DBSCAN(eps=4, min_samples=10, metric='euclidean').fit_predict(positions)

print('Dataset1:')
print("Number of Noise Points: ",sum(dbscan_dataset1==-1)," (",len(dbscan_dataset1),")",sep='')

print("Number of clusters: ", len(set(dbscan_dataset1))-1)

#
# Calculate cluster centroids and their spread
#

centroids = np.zeros((len(set(dbscan_dataset1))-1,2))

MinX = np.ones(len(set(dbscan_dataset1))-1)*size_x
MinY = np.ones(len(set(dbscan_dataset1))-1)*size_y

MaxX = np.zeros(len(set(dbscan_dataset1))-1)
MaxY = np.zeros(len(set(dbscan_dataset1))-1)

for i in range(len(dbscan_dataset1)):
    if dbscan_dataset1[i]!=-1:
        MinX[dbscan_dataset1[i]] = min(MinX[dbscan_dataset1[i]],positions[i][1])
        MinY[dbscan_dataset1[i]] = min(MinY[dbscan_dataset1[i]],positions[i][0])
        MaxX[dbscan_dataset1[i]] = max(MaxX[dbscan_dataset1[i]],positions[i][1])
        MaxY[dbscan_dataset1[i]] = max(MaxY[dbscan_dataset1[i]],positions[i][0])
        centroids[dbscan_dataset1[i]] = centroids[dbscan_dataset1[i]] + positions[i]

for i in range(len(centroids)):
        centroids[i] = centroids[i] / sum(dbscan_dataset1==i)

#
# Plot rectangles around defects in original picture
#

SEM = scipy.misc.imread(INPUTFILE, flatten=True)

size_x,size_y = SEM.shape

import matplotlib.patches as patches

fig = plt.figure()
ax = fig.add_subplot(111,aspect='equal')

ax.imshow(SEM, cmap='gray')

length_x = np.zeros(len(centroids))
length_y = np.zeros(len(centroids))

for i in range(len(centroids)):
    length_x[i] = (MaxX[i]-MinX[i])*3
    length_y[i] = (MaxY[i]-MinY[i])*3
    R = patches.Rectangle((centroids[i][1]-length_x[i]/2,centroids[i][0]-length_y[i]/2),
                            length_x[i],length_y[i],fill=False,edgecolor="red")

    ax.add_patch(R)

#
# Zoom into defect
#

# DN = 0
#
# D = SEM[int(centroids[DN][0]-length_y[DN]/2):int(centroids[DN][0]+length_y[DN]/2)
#     ,int(centroids[DN][1]-length_x[DN]/2):int(centroids[DN][1]+length_x[DN]/2)]
#
# D.shape

ax.imshow(SEM,cmap='gray')

plt.show()
