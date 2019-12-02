from skimage import io
import numpy as np
from sklearn.cluster import KMeans

#  read and show original image
image = io.imread('/Users/ali/Desktop/Image/bird.png')
io.imshow(image)
io.show()

rows, cols = image.shape[0], image.shape[1]
image = image.reshape(rows * cols, 3)
# do kmeans (think of order=20 clusters or so!)
print('kmean')
clt = KMeans(n_clusters=40, max_iter=100)
model = clt.fit(image)
labels = model.labels_

# get your clusters and labels
# find clusters and labels
labels = labels.reshape(rows, cols);
clusters = model.cluster_centers_
print(clusters.shape)

# show decompressed image
image = np.zeros((rows, cols, 3), dtype=np.uint8)
for i in range(rows):
    for j in range(cols):
        image[i, j, :] = clusters[labels[i, j], :]
io.imsave('/Users/ali/Desktop/Image/compressed_image.png', image);
io.imshow(image)
io.show()
