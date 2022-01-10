import numpy as np
import matplotlib.pyplot as plt

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
        # each item is the index of corresponding feature and data
        data_dict[i] = []
    return data_dict

def findLabel(point, cluster_centers):
    "Returns label of the point, based on cluster centers"
    if np.isscalar(point):
        # point is scalar, so there are 1 element
        distance = np.abs(cluster_centers - point)
    else:
        # more than one element is available, so use eculidan distance
        distance = np.sum((cluster_centers - point)**2, axis=1)**(0.5)
    return np.argmin(distance)

def KMeanClustering(data, features, K, N, iters, img_name):
    "K-mean implementation - iters shows termination condition"
    # random selecting initial points
    index = np.random.choice(np.arange(0, N), K, replace=False)    
    # cluser points based on cluster centers
    cluster_centers = features[index]

    for _ in range(iters):
        # make data dictionary
        data_dict = makeDictOfLabels(K)
        # assign labels
        for i in range(data.shape[0]):
            label = findLabel(features[i], cluster_centers)
            data_dict[label].append(i)        
        # find new cluster centers
        for i in range(K):
            cluster_centers[i] = np.sum(features[data_dict[i]], axis=0) / (features[data_dict[i]].shape[0])
    
    plotter(data, data_dict, "Final labels after {} iterations".format(iters), img_name, K)

def plotter(data, data_dict, title, img_name, K):
    "Plots points according to their labels"
    plt.figure(figsize=(8,6), dpi=80)
    for i in range(K):
        plt.scatter(data[data_dict[i]][:, 0], data[data_dict[i]][:, 1], s=10)
    plt.grid(True)
    plt.title(title)
    plt.xlabel('x component')
    plt.ylabel('y component')
        
    plt.savefig('{}.jpg'.format(img_name))

#------------------------- MAIN ---------------------------#
# reading data points
data = readData('./Points.txt')
# number of points
N = data.shape[0]

# plotting data points
plt.figure(figsize=(8,6), dpi=80)
plt.scatter(data[:, 0], data[:, 1], s=10)
plt.grid(True)
plt.xlabel('x component')
plt.ylabel('y component')
plt.title('Data points plot')
plt.savefig('res01.jpg')

# two cluster
K = 2
# clustering based on (x, y) features
features = np.copy(data)
print('(x, y) clustering ...')
KMeanClustering(data, features, K, N, 15, 'res02')
# run one more time
print('Another (x, y) clustering ...')
KMeanClustering(data, features, K, N, 15, 'res03')

# clustering based on distance from origin (radius)
print('Radius clustering ...')
features = np.sqrt(np.sum(data**2, axis=1))
KMeanClustering(data, features, K, N, 5, 'res04')
print('Done!')


