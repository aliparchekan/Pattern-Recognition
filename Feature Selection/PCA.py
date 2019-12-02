from data_loader import load_mnist
from sklearn.naive_bayes import GaussianNB
import copy
from time import time
import numpy as np
import matplotlib.pyplot as ppl


train_data, train_label = load_mnist('/Users/ali/Downloads/Fashion-MNIST')
test_data, test_label = load_mnist('/Users/ali/Downloads/Fashion-MNIST', 't10k')



train_sample_size, feature_size = train_data.shape
test_sample_size , _ = test_data.shape
mean = np.mean(train_data, axis=0)

train_data_std = train_data - mean
covariance = (train_data_std).T.dot(train_data_std)/(train_data_std.shape[0] - 1)
cond = np.linalg.cond(covariance)

eig_vals, eig_vecs = np.linalg.eig(covariance)
print('condition number is {}'.format(cond))


ppl.plot(eig_vals)
ppl.title('eighen values')
ppl.xlabel('number')
ppl.ylabel('eighen value')
ppl.show()
eig_pairs = [(np.abs(eig_vals[i]), eig_vecs[:,i]) for i in range(len(eig_vals))]
eig_pairs.sort()
eig_pairs.reverse()
length_of_chosen_eigh = 784
for i in reversed(range(feature_size)):
    if (eig_pairs[0][0] / eig_pairs[i][0] > 1000):
        length_of_chosen_eigh = length_of_chosen_eigh - 1
print('chosen length is : {}'.format(length_of_chosen_eigh))
matrix_w = []
D = [1/np.sqrt(x) for x in eig_vals]
for i in range(length_of_chosen_eigh):
    matrix_w.append(eig_pairs[i][1].reshape(784,1))
D = np.array(D)
D = np.diag(D[:length_of_chosen_eigh])
matrix_w = np.hstack(matrix_w)
new_data = train_data_std.dot(matrix_w.dot(D))
mean2 = np.mean(test_data, axis=0)
test_data_std = test_data - mean2
new_test = test_data_std.dot(matrix_w.dot(D))

clf = GaussianNB()
clf.fit(new_data, train_label)
predicted = clf.predict(new_test)
print('accuracy is {}'.format(sum((predicted == test_label) * 1)/ test_sample_size))
clf2 = GaussianNB()
clf2.fit(train_data, train_label)
predicted = clf2.predict(test_data)
print('accuracy is {}'.format(sum((predicted == test_label) * 1)/ test_sample_size))
