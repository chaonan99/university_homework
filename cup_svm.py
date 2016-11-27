import pickle
import argparse
from sklearn import svm


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--train_pkl', type=str,
        default="/home/chenhaonan/data/Image/cup_dataset/train_forstu.pickle",
        help='Directory for train pickle')
    parser.add_argument('--test_pkl', type=str,
        default="/home/chenhaonan/data/Image/cup_dataset/valid_forstu.pickle",
        help='Directory for test pickle')
    return parser.parse_args()


def read_pickle(file_path):
    with open(file_path, "rb") as fin:
        return pickle.load(fin, encoding="latin1")
    return None


if __name__ == '__main__':
    args = parse_args()
    train_feature, train_label = read_pickle(args.train_pkl)
    test_feature, test_label = read_pickle(args.test_pkl)
    clf = svm.LinearSVC()
    clf.fit(train_feature, train_label)
    predict_label = clf.predict(test_feature)
    accuracy = (predict_label == test_label).sum()/float(test_label.shape[0])
    print("Accuracy of SVM: %4f" % accuracy)
