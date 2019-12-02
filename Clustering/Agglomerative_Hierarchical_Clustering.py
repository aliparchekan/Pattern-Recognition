import itertools
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import confusion_matrix
from sklearn.cluster import AgglomerativeClustering
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.feature_selection import VarianceThreshold
from sklearn.decomposition import PCA



# Loading Dataset
train_data = np.loadtxt('/Users/ali/Desktop/DS1/DS1/Train_Data.csv', dtype=np.float32, delimiter=',')
train_labels = np.loadtxt('/Users/ali/Desktop/DS1/DS1/Train_Labels.csv', dtype=np.int32, delimiter=',')
test_data = np.loadtxt('/Users/ali/Desktop/DS1/DS1/Test_Data.csv', dtype=np.float32, delimiter=',')
test_labels = np.loadtxt('/Users/ali/Desktop/DS1/DS1/Test_Labels.csv', dtype=np.int32, delimiter=',')
class_names = ['1', '2', '3']

# Feature Selection
all_data = np.vstack((train_data,test_data))
all_data_labels=np.hstack((train_labels,test_labels))
sel = VarianceThreshold(threshold=0.90*(1-0.90))
all_data = sel.fit_transform(all_data)
all_data_size, _ = all_data.shape
_, feature_size = all_data.shape

print('all Data Samples:',all_data_size)
print('Feature Size(after feature-selection):', feature_size)


def plot_confusion_matrix(y_true, y_pred, classes,
                          normalize=False,
                          title=None,
                          cmap=plt.cm.Blues):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    if not title:
        if normalize:
            title = 'Normalized confusion matrix'
        else:
            title = 'Confusion matrix, without normalization'

    # Compute confusion matrix
    cm = confusion_matrix(y_true, y_pred)


    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')

    print(cm)

    fig, ax = plt.subplots()
    im = ax.imshow(cm, interpolation='nearest', cmap=cmap)
    ax.figure.colorbar(im, ax=ax)
    # We want to show all ticks...
    ax.set(xticks=np.arange(cm.shape[1]),
           yticks=np.arange(cm.shape[0]),
           # ... and label them with the respective list entries
           xticklabels=classes, yticklabels=classes,
           title=title,
           ylabel='True label',
           xlabel='Predicted label')

    # Rotate the tick labels and set their alignment.
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right",
             rotation_mode="anchor")

    # Loop over data dimensions and create text annotations.
    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, format(cm[i, j], fmt),
                    ha="center", va="center",
                    color="white" if cm[i, j] > thresh else "black")
    fig.tight_layout()
    plt.show()



clt = AgglomerativeClustering(n_clusters=3)

model = clt.fit(all_data)
clustering_labels = model.labels_
cluster = [[] for _ in range(3)]
mu = []
mean_distance = []
for i in range(3):
    cluster[i] = all_data[clustering_labels == i]
    current_mu = np.mean(cluster[i], axis=0)
    mu.append(current_mu)
    current_mu = current_mu.reshape((1,10))
    temp = 0
    for j, item in enumerate(cluster[i]):
        temp += np.sqrt(np.sum((item - current_mu) ** 2))
    mean_distance.append(temp / len(cluster[i]))




for i in range(len(clustering_labels)):
    clustering_labels[i] = clustering_labels[i] + 1
    if clustering_labels[i] == 2:
        clustering_labels[i] = 3
    elif clustering_labels[i] == 3:
        clustering_labels[i] = 2

plot_confusion_matrix(all_data_labels, clustering_labels, classes=class_names)

conf_matrix = confusion_matrix(all_data_labels, clustering_labels)
mean_correct_classifier_rate = 0
for i in range(3):
    mean_correct_classifier_rate += conf_matrix[i][i]
mean_correct_classifier_rate = mean_correct_classifier_rate/ len(clustering_labels)

print('mean correct classifier rate is {}'.format(mean_correct_classifier_rate))

for i in range(3):
    print('class {} mean distance is: {}'.format(i + 1, mean_distance[i]))


Nmindistance=1000

for i in range(len(all_data)):
    for j in range(len(all_data)):
        if clustering_labels[i]!=clustering_labels[j]:
            dist=np.sqrt(np.sum((all_data[i,:]-all_data[j,:])**2))
            if dist<Nmindistance:
                if dist!=0:
                   Nmindistance=dist

print('Numerator of SI was Calculated:',Nmindistance)

mindistance = [1000 for _ in range(3)]

for i in range(3):
    for j in range(len(cluster[i])):
        for k in range(len(cluster[i])):
            dist = np.sqrt(np.sum((cluster[i][j] - cluster[i][k]) ** 2))
            if dist < mindistance[i]:
                if dist != 0:
                    mindistance[i] = dist


print('Numerator of SI :',Nmindistance)
print('Denumerator of SI :',max(mindistance))
print('Separation Index:',Nmindistance/max(mindistance))
