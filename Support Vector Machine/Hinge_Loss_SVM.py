import numpy as np
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import matplotlib
from mpl_toolkits.mplot3d import Axes3D
from sklearn import svm

data = np.loadtxt('/Users/ali/Downloads/haberman.csv', delimiter=',')
labels = data[:, -1]
values = data[:,:-1]
labels[labels == 2] = -1

pca = PCA(n_components=2)
pca.fit(values)
new_values = pca.transform(values)

plt.figure()
colors = ['red', 'blue']

plt.scatter(new_values[:,0], new_values[:,1], c=labels, cmap= matplotlib.colors.ListedColormap(colors))
plt.show()


result = np.zeros((306, 3))
result[:new_values.shape[0], :new_values.shape[1]] = new_values
for i in range(306):
    result[i,2] = (new_values[i,0] ) * (new_values[i,1] )

fig = plt.figure()
colors = ['red', 'blue']
ax = plt.axes(projection= '3d')
ax.scatter(result[:,0], result[:,1], result[:,2], c=labels, cmap= matplotlib.colors.ListedColormap(colors))

for spine in ax.spines.values():
    spine.set_visible(False)

plt.tight_layout()
plt.show()




def svm_function(x, y):
    w = np.zeros(len(x[0]))
    l_rate = 0.0001
    epoch = 10000
    labda = 1/epoch
    for e in range(epoch):
        for i in range(len(x)):
            val = np.dot(x[i], w)
            if (y[i] * val < 1):
                w = w + l_rate * ((y[i] * x[i])- (2*labda*w))
            else:
                w = w + l_rate * (-2 * labda * w)

    return w

weights = svm_function(result, labels)
print(weights)

x_range = np.linspace(-20,30)
a,b = np.meshgrid(x_range, x_range)
N = x_range.size
plane = weights[0] * a + weights[1] * b + weights[2]


fig = plt.figure()
colors = ['red', 'blue']
ax = plt.axes(projection= '3d')
ax.scatter(result[:,0], result[:,1], result[:,2], c=labels, cmap= matplotlib.colors.ListedColormap(colors))

ax.plot_surface(np.reshape(a, (N,N)), np.reshape(b,(N,N)), np.reshape(plane, (N,N)))

for spine in ax.spines.values():
    spine.set_visible(False)


plt.tight_layout()
plt.show()

clf = svm.SVC(kernel='linear',gamma='scale')
clf.fit(result, labels)
svm_weights = clf.coef_[0]
print(svm_weights)
plane = svm_weights[0] * a + svm_weights[1] * b + svm_weights[2]

fig = plt.figure()
colors = ['red', 'blue']
ax = plt.axes(projection= '3d')
ax.scatter(result[:,0], result[:,1], result[:,2], c=labels, cmap= matplotlib.colors.ListedColormap(colors))

ax.plot_surface(np.reshape(a, (N,N)), np.reshape(b,(N,N)), np.reshape(plane, (N,N)))

for spine in ax.spines.values():
    spine.set_visible(False)


plt.tight_layout()
plt.show()
