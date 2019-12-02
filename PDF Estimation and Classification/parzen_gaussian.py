import numpy as np
import pickle
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
import time
from scipy.stats import norm

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

chosen_sample_size = 100
train_sample_size, feature_size = train_data.shape
print("feature size is: {}".format(feature_size))

test_sample_size, _ = test_data.shape

each_class_length = np.zeros(10)

for i in range(10):
    each_class_length[i] = sum(1 * (train_label == i))

class_priority = each_class_length / np.sum(each_class_length)

print(class_priority)


test_result = np.zeros(test_sample_size)

start_time = time.time()
h = 1
for i in range(test_sample_size):
    values = np.zeros(10)

    for j in range(10):
        a = train_data[train_label == j]
        #b = [norm(np.linalg.norm(test_data[i] - x)/h) for x in a]
        parzen_count = sum([(np.exp(-0.5 * np.linalg.norm(test_data[i] - x)/h)/(2 * np.pi)) for x in a])
        parzen_estimate = parzen_count / (len(a) * h)
        values[j] = class_priority[j] * parzen_estimate


    test_result[i] = float(np.argmax(values))
stop_time = time.time()
duration = stop_time - start_time

print('took {} seconds'.format(duration))
plot_confusion_matrix(test_label, test_result, classes=classes)

conf_matrix = confusion_matrix(test_label, test_result)
mean_correct_classifier_rate = 0
for i in range(10):
    mean_correct_classifier_rate += conf_matrix[i][i]
mean_correct_classifier_rate = mean_correct_classifier_rate / test_sample_size

print('mean correct classifier rate is {}'.format(mean_correct_classifier_rate))
