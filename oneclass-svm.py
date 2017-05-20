#!env/bin/python
from __future__ import division
from datetime import timedelta
import numpy as np
from sklearn import svm, metrics
from sklearn.externals import joblib
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from time import time
import sys

from lib.data import load_data


def main():
    # Data load
    start = time()
    X, y = load_data('data.csv')
    end = time()
    print 'Data loading done in', timedelta(seconds=end - start)

    # Data preprocessing
    start = time()
    y[y == 1] = -1  # Anomalies - Frauds
    y[y == 0] = 1  # Normal cases - Non-frauds

    # Feature normalization
    proportion_anomalies = np.sum(y == -1) / y.size
    scaler = StandardScaler()
    X = scaler.fit_transform(X)

    # Reserve all anomalies for test set
    y_anomaly = y[y == -1]
    X_anomaly = X[y == -1]
    y_normal = y[y == 1]
    X_normal = X[y == 1]

    # Train/test split
    X_train, X_test, y_train, y_test = train_test_split(X_normal, y_normal, test_size=0.2, random_state=0)
    X_test = np.concatenate((X_test, X_anomaly), axis=0)
    y_test = np.concatenate((y_test, y_anomaly), axis=0)

    end = time()
    print 'Data preprocessing done in', timedelta(seconds=end - start)

    # Training
    start = time()
    try:
        clf = joblib.load('OneClassSVM.pkl')
        end = time()
        print 'Model loaded in', timedelta(seconds=end - start)
    except IOError:
        max_iter = 10000
        if len(sys.argv) == 2:
            max_iter = int(sys.argv[1])
        print 'max_iter =', max_iter
        clf = svm.OneClassSVM(nu=proportion_anomalies, kernel='rbf', max_iter=max_iter)
        clf.fit(X_train)
        end = time()
        print 'Training done in', timedelta(seconds=end - start)
        joblib.dump(clf, 'OneClassSVM.pkl')

    # Testing
    start = time()
    predictions = clf.predict(X_test)
    precision = metrics.precision_score(y_test, predictions)
    recall = metrics.recall_score(y_test, predictions)
    f1_score = metrics.f1_score(y_test, predictions)
    end = time()
    print 'Testing done in', timedelta(seconds=end - start)
    print '\tPrecision:', precision
    print '\tRecall:', recall
    print '\tF1-score:', f1_score


if __name__ == '__main__':
    main()

