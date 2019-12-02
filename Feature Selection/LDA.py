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


eig_vals, eig_vecs = np.linalg.eig(covariance)

length_of_chosen_eigh = 189


eig_pairs = [(np.abs(eig_vals[i]), eig_vecs[:,i]) for i in range(len(eig_vals))]
eig_pairs.sort()
eig_pairs.reverse()
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
mean = np.mean(new_data, axis=0)

sw = np.zeros((length_of_chosen_eigh, length_of_chosen_eigh))
sb = np.zeros((length_of_chosen_eigh, length_of_chosen_eigh))
for i in range(10):
    temp = new_data[train_label == i]
    class_mean = np.mean(temp, axis=0)
    sw +=(temp - class_mean).T.dot((temp - class_mean))
    sb += np.outer(class_mean.reshape(length_of_chosen_eigh,1) - mean.reshape(length_of_chosen_eigh, 1), class_mean.reshape(length_of_chosen_eigh, 1) - mean.reshape(length_of_chosen_eigh, 1)) * len(temp)

sp = np.matmul(np.linalg.inv(sw), sb)
sp_eigh_vals, sp_eigh_vecs = np.linalg.eigh(sp)
eig_pairs = [(np.abs(sp_eigh_vals[i]), sp_eigh_vecs[:,i]) for i in range(len(sp_eigh_vals))]
eig_pairs.sort()
eig_pairs.reverse()
ppl.figure()
sorted_eigh = [eig_pairs[i][0] for i in range(length_of_chosen_eigh)]
ppl.plot(sorted_eigh)
ppl.title('eighen values')
ppl.xlabel('number')
ppl.ylabel('eighen value')
ppl.show()

sep_measure = [sum(sorted_eigh[:i + 1]) for i in range(length_of_chosen_eigh)]
ppl.figure()
ppl.plot(sep_measure)
ppl.title('seperability measure')
ppl.xlabel('number of features')
ppl.ylabel('separibility value')

ppl.show()

cond = np.linalg.cond(sp)
print('condition number is {}'.format(cond))

length_of_chosen_eigh_2 = length_of_chosen_eigh
for i in reversed(range(length_of_chosen_eigh)):
    if (eig_pairs[0][0] / eig_pairs[i][0] > 1000):
        length_of_chosen_eigh_2 = length_of_chosen_eigh_2 - 1
print('chosen length is {}'.format(length_of_chosen_eigh_2))
matrix_w2 = []
for i in range(length_of_chosen_eigh_2):
    matrix_w2.append(eig_pairs[i][1].reshape(length_of_chosen_eigh,1))

matrix_w2 = np.hstack(matrix_w2)

new_data = new_data.dot(matrix_w2)
new_test = new_test.dot(matrix_w2)

clf = GaussianNB()
clf.fit(new_data, train_label)
predicted = clf.predict(new_test)
print('accuracy is {}'.format(sum((predicted == test_label) * 1)/ test_sample_size))
clf2 = GaussianNB()
clf2.fit(train_data, train_label)
predicted = clf2.predict(test_data)
print('accuracy is {}'.format(sum((predicted == test_label) * 1)/ test_sample_size))