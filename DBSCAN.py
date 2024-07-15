from scipy.spatial import distance_matrix as dist
import pandas as pd, numpy as np

def dbscan(x, m= 14, e= 444, p= 1):
    dMatrix = dist(x, x, p= p)

    neighbors = []
    for i in range(len(x)):
        neighbors.append([i])
        for j in range(len(x)):         
            if dMatrix[i, j] <= e and dMatrix[i, j] != 0:
                neighbors[i].append(j)

    core_points = []
    for i in neighbors:
        rep = True
        if len(i) >= m:
            for j in core_points:
                if i[1:] == j[1:]:
                    rep = False
            if rep:
                core_points.append(i)
    
    classes = np.zeros(len(x))
    for i, cluster in enumerate(core_points):
        classes[cluster] = i+1
    
    return classes, len(core_points)

if __name__=="__main__":
    dataset = np.array(pd.read_csv("sorlie.csv", header=None))
    x, y = dataset[:, :-1], dataset[:, -1]
    # clusters, n = dbscan(x, m= 11, e= 27, p= 2)
    clusters, n = dbscan(x, m= 14, e= 444, p= 1)
    
    print(f"number of clusters: {n} (0 labels are for noise samples)")
    print(clusters)