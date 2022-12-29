import pandas as pd
from sklearn.cluster import KMeans
import numpy as np
from sklearn.metrics import silhouette_score
import matplotlib.pyplot as plt

dataset = pd.read_csv("Live.csv")
print(dataset.head())

model = KMeans(n_clusters=4)
print(model.fit(dataset))
labels = model.predict(dataset)
np.unique(labels, return_counts = True)

silhouettes = []
ks = list(range(2, 12))
for n_cluster in ks:
    kmeans = KMeans(n_clusters=n_cluster).fit(dataset)
    label = kmeans.labels_
    sil_coeff = silhouette_score(dataset, label, metric='euclidean')
    print("For n_clusters={}, The Silhouette Coefficient is {}".format(n_cluster, sil_coeff))
    silhouettes.append(sil_coeff)

plt.figure(figsize=(12, 8))    
plt.subplot(211)
plt.scatter(ks, silhouettes, marker='x', c='r')
plt.plot(ks, silhouettes)
plt.xlabel('k')
plt.ylabel('Silhouette score')
plt.show()

dataset["labels"] = labels
print(dataset)

group_zero = dataset[dataset["labels"]==0]["num_comments"].mean()
print(group_zero)

group_one = dataset[dataset["labels"]==1]["num_comments"].mean()
print(group_one)

group_two = dataset[dataset["labels"]==2]["num_comments"].mean()
print(group_two)

group_three = dataset[dataset["labels"]==3]["num_comments"].mean()
print(group_three)

group_zero = dataset[dataset["labels"]==0]["num_shares"].mean()
print(group_zero)

group_one = dataset[dataset["labels"]==1]["num_shares"].mean()
print(group_one)

group_two = dataset[dataset["labels"]==2]["num_shares"].mean()
print(group_two)

group_three = dataset[dataset["labels"]==3]["num_shares"].mean()
print(group_three)


status_type = dataset[["status_type_photo", "status_type_video", "status_type_status"]].idxmax(axis=1)
dataset = pd.concat([dataset["labels"],status_type.rename("status_type")], axis=1)
dataset.groupby(["labels","status_type"])["status_type"].count()