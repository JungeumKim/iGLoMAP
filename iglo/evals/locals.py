from sklearn import svm as svm_

import copy
import torch
import sys

sys.path.append("/home/kim2712/Desktop/research/Othermethods")
from PaCMAP.evaluation.evaluation import *

from sklearn.neighbors import KNeighborsClassifier, NearestNeighbors

from IPython.core.debugger import set_trace

def knn_test(X_test, y_test, X_train, y_train, n_neighbor=1):
    knn = KNeighborsClassifier(n_neighbors=n_neighbor)
    knn.fit(X_train, y_train)

    predicted = knn.predict(X_test)

    return (predicted == y_test).mean()


def knn_test_series(X_test, y_test, X_train, y_train, n_neighbors=[1, 3, 5, 10, 15, 20, 25, 30]):
    avg_accs = []
    for n_neighbor in n_neighbors:
        avg_acc = knn_test(X_test, y_test, X_train, y_train, n_neighbor)
        avg_accs.append(avg_acc)
    return avg_accs


def local_classifiers(z, y, n_neighbors=[1, 3, 5, 10, 15, 20, 25, 30], svm=True):
    n_neighbors = copy.copy(n_neighbors)

    accus = knn_eval_series(z, y, n_neighbors)

    if svm:
        svm_acc = faster_svm_eval(z, y)

        return accus, svm_acc

    return accus, 0

# device="cuda"
# z, y = get_Z(net, loader, device)
# z_test, y_test = get_Z(net, test_loader, device)

def local_msrs_test(z,y, z_test,y_test, n_neighbors=[1, 3, 5, 10, 15, 20, 25, 30], do_svm=True):

    accus, svm = local_classifiers(z, y, n_neighbors, svm=do_svm)

    test_accus = knn_test_series(z_test, y_test, z, y, n_neighbors=n_neighbors)

    if do_svm:

        clf = svm_.SVC(kernel='rbf')
        clf.fit(z, y)
        y_pred = clf.predict(z_test)

        test_svm = (y_test == y_pred).mean()
    else:

        test_svm = 0


    locals_ = {"knn_accus": accus, "svm": svm, "knn_accus_test": test_accus,
               "svm_test": test_svm, "knn_acc": accus[3], "knn_acc_test": test_accus[3]}
    return locals_


def local_msrs_np(z, y, n_neighbors=[1, 3, 5, 10, 15, 20, 25, 30], do_svm=True):
    accus, svm = local_classifiers(z, y, n_neighbors, svm=do_svm)

    locals_ = {"knn_accus": accus, "svm": svm, "knn_acc": accus[3]}

    return locals_





