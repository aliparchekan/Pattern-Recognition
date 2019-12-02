from data_loader import load_mnist
from sklearn.naive_bayes import GaussianNB
import copy
from time import time
import numpy as np
import matplotlib.pyplot as ppl
from sklearn.decomposition import PCA
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA



train_data, train_label = load_mnist('/Users/ali/Downloads/Fashion-MNIST')
test_data, test_label = load_mnist('/Users/ali/Downloads/Fashion-MNIST', 't10k')



train_sample_size, feature_size = train_data.shape
test_sample_size , _ = test_data.shape

pca_scores = []
lda_scores = []


for i in range(feature_size):
    temp = copy.deepcopy(train_data)
    test_temp = copy.deepcopy(test_data)
    pca = PCA(n_components=i + 1)
    pca.fit(temp)
    temp = pca.transform(temp)
    test_temp = pca.transform(test_temp)
    clf = GaussianNB()
    clf.fit(temp, train_label)
    result = clf.predict(test_temp)
    pca_scores.append(sum((result == test_label) * 1)/ test_sample_size)

for i in range(10):
    temp = copy.deepcopy(train_data)
    test_temp = copy.deepcopy(test_data)
    clf = LDA(n_components=i + 1)
    clf.fit(temp, train_label)
    temp = clf.transform(temp)
    test_temp = clf.transform(test_temp)
    clf = GaussianNB()
    clf.fit(temp , train_label)
    result = clf.predict(test_temp)
    lda_scores.append(sum((result == test_label) * 1) / test_sample_size)

ppl.figure()
ppl.plot(pca_scores)
ppl.show()
ppl.figure()
ppl.plot(lda_scores)
ppl.show()
