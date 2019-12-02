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

chosen_feature = []
scores = []

for i in range(100):
    print('choosing {} th feature'.format(i))
    start = time()
    values = np.zeros(feature_size)
    for j in range(feature_size):
        if j in chosen_feature:
            values[j] = -1
            continue
        temp_features = copy.deepcopy(chosen_feature)
        temp_features.append(j)
        train_sub = train_data[:,temp_features]
        test_sub = test_data[:, temp_features]
        clf = GaussianNB()
        clf.fit(train_sub, train_label)
        predicted = clf.predict(test_sub)
        values[j] = sum((predicted == test_label) * 1)/ test_sample_size
    chosen_feature.append(np.argmax(values))
    scores.append(values[chosen_feature[-1]])
    stop = time()
    print('elapsed {} seconds to choose {} th feature'.format(stop - start, i))

print('best practice is with {} features'.format(np.argmax(scores) + 1))
ppl.plot(scores)
ppl.title('correct rate vs. Features')
ppl.xlabel('feature size')
ppl.ylabel('correct rate')
ppl.show()




