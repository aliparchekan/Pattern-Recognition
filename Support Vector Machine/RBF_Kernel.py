from data_loader import load_mnist
import numpy as np
from sklearn import svm
from sklearn.model_selection import GridSearchCV
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
from time import time
from sklearn.svm import LinearSVC


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

train_data = np.loadtxt('/Users/ali/Downloads/ReducedFashion-MNIST/Train_Data.csv', dtype=np.float32, delimiter=',')
train_data = train_data / 100
train_label = np.loadtxt('/Users/ali/Downloads/ReducedFashion-MNIST/Train_Labels.csv', dtype=np.float32, delimiter=',')
test_data = np.loadtxt('/Users/ali/Downloads/ReducedFashion-MNIST/Test_Data.csv', dtype=np.float32, delimiter=',')
test_data = test_data / 100
test_label = np.loadtxt('/Users/ali/Downloads/ReducedFashion-MNIST/Test_Labels.csv', dtype=np.float32, delimiter=',')


classes = ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9']


#rbf one vs one
start = time()
C_range = np.logspace(-2, 10, 4)
gamma_range = np.logspace(-9, 5, 4)
param_grid = dict(gamma = gamma_range, C = C_range)

svr = svm.SVC(kernel='rbf', decision_function_shape='ovo')
clf = GridSearchCV(svr, param_grid=param_grid)
clf.fit(train_data, train_label)
print('train duration is {}'.format(time() - start))
start = time()
test_result = clf.predict(test_data)
print('test duration is {}'.format(time() - start))
plot_confusion_matrix(test_label, test_result, classes=classes)

print('CCR is {}'.format(np.sum(1*(test_result == test_label)) / len(test_result)))
print('the best parameters are {} '.format(clf.best_params_))

#rbf one vs rest
start = time()
C_range = np.logspace(-2, 10, 4)
gamma_range = np.logspace(-9, 5, 4)
param_grid = dict(gamma = gamma_range, C = C_range)

svr = svm.SVC(kernel='rbf', decision_function_shape='ovr')
clf = GridSearchCV(svr, param_grid=param_grid)
clf.fit(train_data, train_label)
print('train duration is {}'.format(time() - start))
start = time()
test_result = clf.predict(test_data)
print('test duration is {}'.format(time() - start))
plot_confusion_matrix(test_label, test_result, classes=classes)

print('CCR is {}'.format(np.sum(1*(test_result == test_label)) / len(test_result)))
print('the best parameters are {} '.format(clf.best_params_))
