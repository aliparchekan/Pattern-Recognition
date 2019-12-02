import numpy as np
import pickle
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
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


classes = ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9', 'not classified']
lambda_r = 0.8
lambda_s = 1

train_data = train_data / 100
test_data = test_data / 100

train_sample_size , feature_size = train_data.shape
print("feature size is: {}".format(feature_size))

test_sample_size, _ = test_data.shape

each_class_length = np.zeros(10)

for i in range(10):
    each_class_length[i] = sum(1 * (train_label == i))

class_priority = each_class_length / np.sum(each_class_length)

print(class_priority)

means = np.zeros([10, feature_size])
for i in range(10):
    means[i] = np.mean(train_data[train_label == i], axis=0)
variances = np.zeros([10, feature_size, feature_size])
for i in range(10):
    a = train_data[train_label == i,:]
    for j in a:
        temp = j - means[i]
        temp = temp.reshape(feature_size, 1)
        variances[i] += temp*temp.T
    variances[i] = variances[i] / each_class_length[i]

sigma = np.zeros([feature_size, feature_size])

for i in range(10):
    sigma += (each_class_length[i] - 1) * variances[i]
sigma = sigma / (train_sample_size - 10)

test_result = np.zeros(test_sample_size)
start_time = time()

for i in range(test_sample_size):
    values = np.zeros(10)
    temp2 = np.sqrt(np.linalg.det(sigma)) * (2 * np.pi)**(feature_size/2)
    for j in range(10):

        temp = (test_data[i,:] - means[j]).reshape(feature_size,1)

        values[j] = class_priority[j] * np.exp(-0.5 * np.matmul(np.matmul(temp.T,np.linalg.inv(sigma)),temp)) / temp2

    test_result[i] = float(np.argmax(values))
    if (values[int(test_result[i])]/sum(values) < (1 - float(lambda_r / lambda_s))):
        test_result[i] = 10
stop_time = time()
print('took {} seconds'.format(stop_time - start_time))
plot_confusion_matrix(test_label, test_result, classes=classes)

conf_matrix = confusion_matrix(test_label, test_result)
mean_correct_classifier_rate = 0
for i in range(10):
    mean_correct_classifier_rate += conf_matrix[i][i]
mean_correct_classifier_rate = mean_correct_classifier_rate/ test_sample_size

print('mean correct classifier rate is {}'.format(mean_correct_classifier_rate))
