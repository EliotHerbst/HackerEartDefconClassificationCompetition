from sklearn.metrics import make_scorer, accuracy_score

from tools import get_accuracy, write_to_file


def PIPELINE(estimator):
    print(get_accuracy(estimator))
    write_to_file(estimator)
