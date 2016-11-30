from sklearn.neural_network import MLPClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.svm import SVC, LinearSVC
from sklearn.naive_bayes import GaussianNB
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.feature_selection import *
from sklearn.linear_model import LogisticRegression, LassoCV
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.metrics import accuracy_score
from sklearn.decomposition import PCA
import numpy as np
import pickle
import pandas as pd
import argparse


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--train_pkl', type=str,
        default="train_forstu.pickle",
        help='Directory for train pickle')
    parser.add_argument('--test_pkl', type=str,
        default="valid_forstu.pickle",
        help='Directory for test pickle')
    return parser.parse_args()


def read_pickle(file_path):
    with open(file_path, "rb") as fin:
        return pickle.load(fin, encoding="latin1")
    return None


def do_prediction(clf, X_train, X_test, y_train, y_test):
    scaler = StandardScaler()
    scaler.fit(X_train)
    X_train = scaler.transform(X_train)
    X_test = scaler.transform(X_test)
    clf.fit(X_train, y_train)
    return accuracy_score(clf.predict(X_test), y_test)


def do_selection_and_prediction(sel, clf, X_train, X_test, y_train, y_test):
    sel.fit(X_train, y_train)
    return do_prediction(clf, sel.transform(X_train), sel.transform(X_test), y_train, y_test)


if __name__ == '__main__':
    args = parse_args()
    X_train_raw, y_train = read_pickle(args.train_pkl)
    X_test_raw, y_test = read_pickle(args.test_pkl)
    classifier_names = ["Bayes", "LDA", "LSVM", "RBF SVM", "MLP", "DTree"]
    classifiers = [
        GaussianNB(),
        LinearDiscriminantAnalysis(),
        LinearSVC(C=0.025),
        SVC(gamma=0.001, C=100),
        MLPClassifier(alpha=1e-5, hidden_layer_sizes=256, random_state=1, max_iter=3000),
        ExtraTreesClassifier(),
    ]
    print("Test accuracy")
    test_res = pd.DataFrame([["{0:.2f}%".format(do_selection_and_prediction(sel, clf, X_train_raw,
            X_test_raw, y_train, y_test)*100) for clf in classifiers] for sel in selectors],
            columns=classifier_names)
    print(test_res)
