import numpy as np
import matplotlib.pyplot as plt
#import cv2
#import utils as utl

#------------------------- FUNCTIONS ----------------------#
def readData(path):
    "Reads data points from a file in specified path - returns numpy array"
    with open(path, 'r') as f:
        num_data = int(f.readline().split('\n')[0])
        data_points = []
        while 1:
            point = f.readline().split('\n')[0]
            if point:
                point = list(map(float, point.split(' ')))
                data_points.append(point)
            else:
                break
    if len(data_points) != num_data:
        raise ValueError("Some lines are missing, not able to read file!")
    else:
        return np.array(data_points, dtype=np.float64)

def makeDictOfLabels(K):
    "Makes an empty dictionary which each key shows corresponding labels"
    data_dict = dict()
    for i in range(K):
        # 0 * number of features
        data_dict[i] = np.empty((0, 2), dtype=np.float64)
    return data_dict

def findLabel(point, cluster_centers):
    "Returns label of the point, based on cluster centers"
    distance = np.sum((cluster_centers - point)**2, axis=1)**(0.5)
    return np.argmin(distance)

def KMeanClustering(data, K, N, iters):
    "K-mean implementation - iters shows termination condition"
    # random selecting initial points
    index = np.random.choice(np.arange(0, N), K, replace=False)    
    # cluser points based on cluster centers
    cluster_centers = data[index]
    
    iter_val = 0
    while iter_val < iters:
        # make data dictionary
        data_dict = makeDictOfLabels(K)
        # assign labels
        for i in range(data.shape[0]):
            label = findLabel(data[i], cluster_centers)
            data_dict[label] = np.append(data_dict[label], data[i].reshape(1, 2), axis=0)
        
        iter_val += 1
        # plot points:
        plotter(data_dict, cluster_centers, iter_val, K)
        # find new cluster centers
        for i in range(K):
            cluster_centers[i] = np.sum(data_dict[i], axis=0) / (data_dict[i].shape[0])
    
def plotter(data_dict, cluster_centers, iter_val, K):
    "Plots points according to their labels"
    plt.figure()
    for i in range(K):
        plt.scatter(data_dict[i][:, 0], data_dict[i][:, 1], s=10)
        plt.scatter(cluster_centers[i, 0], cluster_centers[i, 1], 
                    marker='+', c='k', s=300)
        plt.title("Labeling at iteration {}".format(iter_val))

    #plt.savefig('{}.png'.format(iter_val))

#------------------------- MAIN ---------------------------#
# reading data points
data = readData('./Points.txt')
# number of points
N = data.shape[0]

# plotting data points
plt.figure()
plt.scatter(data[:, 0], data[:, 1], s=10)
#plt.savefig('0.png')

K = 2
KMeanClustering(data, K, N, 20)
plt.show()
