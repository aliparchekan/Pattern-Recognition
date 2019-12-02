import numpy as np
import pickle
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
from sklearn import neighbors
from sklearn.naive_bayes import GaussianNB
from time import time


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

# Load Dataset

train_data = np.loadtxt('/Users/ali/Downloads/ReducedFashion-MNIST/Train_Data.csv', dtype=np.float32, delimiter=',')
train_label = np.loadtxt('/Users/ali/Downloads/ReducedFashion-MNIST/Train_Labels.csv', dtype=np.float32, delimiter=',')
test_data = np.loadtxt('/Users/ali/Downloads/ReducedFashion-MNIST/Test_Data.csv', dtype=np.float32, delimiter=',')
test_label = np.loadtxt('/Users/ali/Downloads/ReducedFashion-MNIST/Test_Labels.csv', dtype=np.float32, delimiter=',')


classes = ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9']

train_data = train_data / 100
test_data = test_data / 100
start = time()
knn1 = neighbors.KNeighborsClassifier(1)
model1 = knn1.fit(train_data, train_label)
test_result_knn = model1.predict(test_data)
accuracy_knn1 = sum(1 * (test_result_knn == test_label)) / len(test_label)
stop_knn = time()
print('KNN accuracy is {}'.format(accuracy_knn1))
print('took {} seconds'.format(stop_knn - start))

plot_confusion_matrix(test_label, test_result_knn, classes=classes)
start = time()
pnp = neighbors.RadiusNeighborsClassifier(30)
model_pnp = pnp.fit(train_data, train_label)
test_result_pnp = model_pnp.predict(test_data)
accuracy_pnp = sum(1 * (test_result_pnp == test_label)) / len(test_label)
stop_pnp = time()
print("parzen window accuracy is: {}".format(accuracy_pnp))
print('took {} seconds'.format(stop_pnp - start))

plot_confusion_matrix(test_label, test_result_pnp, classes=classes)
start = time()
bng = GaussianNB()
model_bng = bng.fit(train_data, train_label)
test_result_bng = model_bng.predict(test_data)
accuracy_bng = sum(1 * (test_result_bng == test_label)) / len(test_label)
stop_bng = time()

print('gaussian naive bayes accuracy is : {}'.format(accuracy_bng))
print('took {} seconds'.format(stop_bng - start))

plot_confusion_matrix(test_label, test_result_bng, classes= classes)