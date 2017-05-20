#!env/bin/python
import matplotlib.pyplot as plt
import numpy as np
from sklearn.preprocessing import StandardScaler

from lib.data import load_data


def main():
    print 'Reading data'
    X, y = load_data('data.csv')

    print 'Feature normalization'
    scaler = StandardScaler()
    X = scaler.fit_transform(X)

    print 'Plotting histograms'
    plt.figure()
    for i in range(X.shape[1]):
        x = X[:, i]
        plt.subplot(3, 2, i + 1)
        plt.title('Feature {}'.format(i))
        plt.hist(x[y == 1], bins=50)
    plt.show()


if __name__ == '__main__':
    main()

