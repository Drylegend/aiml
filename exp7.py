import csv
import numpy as np

data = np.random.rand(100, 2)

with open('data.csv', 'w', newline='') as csvfile:
    writer = csv.writer(csvfile, delimiter=',')

    for row in data:
        writer.writerow(row)


import csv
import numpy as np
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score


data = []
print(type(data))

with open('data.csv', 'r', newline='') as csvfile:
    reader = csv.reader(csvfile, delimiter=',')
    for row in reader:
        data.append([float(x) for x in row])
        


kmeans = KMeans(n_clusters=3)
kmeans.fit(data)


labels = kmeans.labels_
centers = kmeans.cluster_centers_
silhouette = silhouette_score(data, labels)


print("Cluster labels:", labels)
print("Cluster centers:", centers)
print("Silhouette score:", silhouette)
