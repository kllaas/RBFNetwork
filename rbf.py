from scipy import *
from scipy.linalg import norm, pinv

from matplotlib import pyplot as plt
import pandas as pd
import numpy as np


class RBF:
    def __init__(self, indim, numCenters, outdim):
        self.indim = indim
        self.outdim = outdim
        self.numCenters = numCenters
        self.centers = [random.uniform(-1, 1, indim) for i in xrange(numCenters)]
        self.beta = 8
        self.W = random.random((self.numCenters, self.outdim))

    def _basisfunc(self, c, d):
        return exp(-self.beta * norm(d - c) ** 2)

    def _calcAct(self, X):
        # calculate activations of RBFs
        G = zeros((X.shape[0], self.numCenters), float)
        for ci, c in enumerate(self.centers):
            for xi, x in enumerate(X):
                G[xi, ci] = self._basisfunc(c, x)
        return G

    def train(self, X, Y):
        """ X: matrix of dimensions n x indim
            y: column vector of dimension n x 1 """

        # choose random center vectors from training set
        rnd_idx = random.permutation(X.shape[0])[:self.numCenters]
        self.centers = [X[i, :] for i in rnd_idx]

        print "center", self.centers

        # calculate activations of RBFs
        G = self._calcAct(X)
        print G

        # calculate output weights (pseudoinverse)
        self.W = dot(pinv(G), Y)

    def test(self, X):
        """ X: matrix of dimensions n x indim """

        G = self._calcAct(X)
        Y = dot(G, self.W)
        return Y


def plot_test_set():
    df = pd.read_csv('/home/alexey/PycharmProjects/rbfNetwork/test_data.csv', header=None)

    y = df.iloc[1:21, 2].values
    x = df.iloc[1:21, [0, 1]].values
    z = rbf.test(x)

    class_a = []
    class_b = []
    incorrects = []

    right_answers = 0
    for i, item in enumerate(y):
        if abs(item - z[i]) <= 0.2:
            right_answers += 1

            if abs(z[i] - 1) > 0.2:
                class_b.append(x[i])
            else:
                class_a.append(x[i])
        else:
            incorrects.append(x[i])

    percents = (float(right_answers) / len(z)) * 100
    print 'test data accuracy = %f' % percents

    # plot data
    plt.figure(figsize=(12, 8))
    class_a = zip(*class_a)
    class_b = zip(*class_b)
    incorrects = zip(*incorrects)

    plt.plot(class_a[0], class_a[1], 'ro', c='b')
    plt.plot(class_b[0], class_b[1], 'ro', c='m')
    if len(incorrects) != 0:
        plt.plot(incorrects[0], incorrects[1], 'ro', c='r')

    plt.title('RBF. Accuracy=%d%% (on test data)' % percents)

    plt.show()


if __name__ == '__main__':
    df = pd.read_csv('/home/alexey/PycharmProjects/rbfNetwork/data05.csv', header=None)

    y = df.iloc[1:78, 2].values
    x = df.iloc[1:78, [0, 1]].values

    # rbf regression
    rbf = RBF(2, 60, 1)
    rbf.train(x, y)
    z = rbf.test(x)

    right_answers = 0

    for i, item in enumerate(y):
        if abs(item - z[i]) < 0.1:
            right_answers += 1

    percents = (float(right_answers) / len(z)) * 100
    print 'train data accuracy = %f' % percents

    plot_test_set()
