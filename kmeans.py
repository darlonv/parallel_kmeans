from tqdm import tqdm
import random
import math
from operator import add

import matplotlib.pyplot as plt
import numpy as np



SEP = ','


def load_file(filename):
    points = []
    with open(filename, 'r') as f:
        lines = f.read().splitlines()

        
        for l in lines:
            include = True
            row = []
            for v in l.split(SEP):
                if v.isnumeric():  row.append(float(v))
                else: include = False
            if include: points.append(row)

        del(lines)
    return points

def minmax(points):
    max = points[0].copy()
    min = points[0].copy()

    for point in points:
        for p in range(len(point)):
            if point[p] > max[p]:
                max[p] = point[p]
            if point[p] < min[p]:
                min[p] = point[p]
    return min,max

def getrand(min, max):
    point = []
    for p in range(len(max)):
        # r = random.randint(min[p], max[p]) # int
        r = random.random()*(max[p]-min[p]) + min[p] # float
        point.append(r)

    return point

def get_random_centroids(points, n_centroids, max_tries=10):
    centroids = []
    min, max = minmax(points)
    dim = len(points[0])

    for i in range(n_centroids):
        centroids.append(getrand(min,max))
    return centroids

def select_centroids(points, n, max_tries=10):
    centroids = []
    tries = 0
    while len(centroids) < n:
        row = random.choice(points)
        if row not in centroids:
            centroids.append(row)
        else:
            tries +=1
            if tries == max_tries:
                print(f'There are at {n} different points?')
                return
    
    return centroids

def distance(p1, p2):
    s=0
    for i,j in zip(p1,p2):
        s+=(j-i)**2
    
    d = math.sqrt(s)
    return d

def closest_point(p, points):
    dists = []
    
    for point in points:
        d = distance(p, point)
        dists.append(d)
    
    closest = dists[0]
    closest_idx = 0
    
    for d in range(1,len(dists)):
        if dists[d] < closest:
            closest = dists[d]
            closest_idx = d
    
    return closest_idx

def closest_centroid(points, centroids):
    closest = []
    for p in points:
        c = closest_point(p, centroids)
        closest.append(c)
    return closest

def calc_medium_point(points):

    s = list([0]*len(points))
    for point in points:
        s = list(map(add,s, point))

    for i in range(len(s)):
        s[i] = s[i]/len(points)
    # print(s)



    return s

def calc_centroids(points, clusters, n):
    points_clusters = []
    for _ in range(n):
        points_clusters.append([])

    for p, c in zip(points, clusters):
        points_clusters[c].append(p)
    # print(points_clusters)

    centroids = []
    for p in points_clusters:
        centroids.append(calc_medium_point(p))


    return centroids

def plot2d(points, clusters, centroids):
    np_points = np.array(points)
    np_centroids=np.array(centroids)
    plt.figure()
    plt.title(f'Iteration {iter}')
    plt.scatter(x=np_points[:,0], y=np_points[:,1], c=clusters)
    plt.scatter(x=np_centroids[:,0], y=np_centroids[:,1], c='red')
    # plt.show(block=True)
    plt.show()

def kmeans(file, n_clusters=4, max_iter=100, bar=True, verbose=False, plot=False, iterplot=False):

    points = load_file(file)
    # centroids = select_centroids(points, n_clusters)
    centroids = get_random_centroids(points, n_clusters)
    # print(centroids)

    # centroids = [[2,2],[5,5]]
    # print('Centroids:',centroids)
    iterator = range(max_iter)
    if bar: iterator = tqdm(range(max_iter))
    for iter in iterator:
        if verbose:
            print(f'*** ITER {iter} ***')
        centroids_old = centroids.copy()
        clusters = closest_centroid(points, centroids)
        centroids = calc_centroids(points, clusters, n_clusters)

        if verbose:
            print('Clusters:',clusters)
            print('Centroids:',centroids)

        ## Plot
        if iterplot:
            plot2d(points, clusters)


        if centroids == centroids_old:
            # iterator.close()
            break


    if plot:
        plot2d(points, clusters, centroids)
        
