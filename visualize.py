#!env/bin/python
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

from lib.data import load_data


def main():
    print 'Reading data'
    X, y = load_data('data.csv')

    print 'Feature normalization'
    scaler = StandardScaler()
    X = scaler.fit_transform(X)

    print 'Running PCA'
    pca = PCA(n_components=2)
    X_reduced = pca.fit(X).transform(X)

    print 'Plotting PCA in 2D'
    plt.figure()
    colors = ['navy', 'darkorange']
    for i, color in enumerate(colors):
        plt.scatter(X_reduced[y == i, 0], X_reduced[y == i, 1], color=color)
    plt.xlabel('$z_1$')
    plt.ylabel('$z_2$')
    plt.title('Principal Component Analisis of data')
    plt.show()        


if __name__ == '__main__':
    main()

