import numpy as np
from scipy import convolve
from sklearn.neural_network import BernoulliRBM
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.datasets import fetch_mldata
from sklearn import cross_validation
from sklearn import metrics
import matplotlib.pyplot as plt

__author__ = 'Sogo'

def nudge_dataset(X, Y):
    """
    This produces a dataset 5 times bigger than the original one,
    by moving the 8x8 images in X around by 1px to left, right, down, up
    """
    direction_vectors = [
    [[0, 1, 0],
    [0, 0, 0],
    [0, 0, 0]],

    [[0, 0, 0],
    [1, 0, 0],
    [0, 0, 0]],

    [[0, 0, 0],
    [0, 0, 1],
    [0, 0, 0]],

    [[0, 0, 0],
    [0, 0, 0],
    [0, 1, 0]]]

    shift = lambda x, w: convolve(x.reshape((64, 64)), mode='constant', weights=w).ravel()
    X = np.concatenate([X] + [np.apply_along_axis(shift, 1, X, vector) for vector in direction_vectors])
    Y = np.concatenate([Y for _ in range(5)], axis=0)
    return X, Y

image_shape = (28, 28)
def plot_gallery(title, images, n_col, n_row):
    plt.figure(figsize=(2. * n_col, 2.26 * n_row))
    plt.suptitle(title, size=16)
    for i, comp in enumerate(images):
        plt.subplot(n_row, n_col, i + 1)
        vmax = max(comp.max(), -comp.min())
        plt.imshow(comp.reshape(image_shape), cmap=plt.cm.gray,
                   vmin=-vmax, vmax=vmax)
        plt.xticks(())
        plt.yticks(())
    plt.subplots_adjust(0.01, 0.05, 0.99, 0.93, 0.04, 0.)
    plt.show()

mnist = fetch_mldata('MNIST original')
X, Y = mnist.data, mnist.target
X = np.asarray( X, 'float32')
# Scaling between 0 and 1
X = (X - np.min(X, 0)) / (np.max(X, 0) + 0.0001)  # 0-1 scaling
# Convert to binary images
X = X > 0.5
print ('Input X shape', X.shape)

rbm = BernoulliRBM(n_components=200, learning_rate=0.01, batch_size=10, n_iter=10, verbose=True, random_state=None)
logistic = LogisticRegression(C=10)
print('Pipeline Activated.')
clf = Pipeline(steps=[('rbm', rbm), ('clf', logistic)])
X_train, X_test, Y_train, Y_test = cross_validation.train_test_split( X, Y, test_size=0.2, random_state=0)
print('Data Fitting:')
clf.fit(X_train, Y_train)
Y_pred = clf.predict(X_test)
print('Score:')
print(metrics.classification_report(Y_test, Y_pred))

# check components of RBM
comp = rbm.components_
plot_gallery('RBM Components', comp[:25], 5, 5)