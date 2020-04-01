import numpy as np
from sklearn.cluster import AgglomerativeClustering

def cluster(nclus, rows, cols, zs):
    print("Clustering starts")
    coord = np.array(list(zip(rows, cols, zs)))
    print("NCLUS passed to python ", nclus)
    print(coord)
    agg = AgglomerativeClustering(nclus, linkage='single').fit(coord)
    labels = agg.labels_
    labels += 1
    print("Labels are: ", labels, "size ", labels.size)
    print("Clustering ends")
    return labels.astype(dtype=np.int32)
