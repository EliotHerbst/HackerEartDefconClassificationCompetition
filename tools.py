import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import sklearn
from sklearn.metrics import make_scorer, accuracy_score
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
from sklearn.datasets import load_digits
from sklearn.model_selection import learning_curve, train_test_split, cross_val_score
from sklearn.model_selection import ShuffleSplit


def get_accuracy(estimator):
    df = pd.read_csv("train.csv", header=None)
    df = df.sample(frac=1).reset_index(drop=True)
    dataset = df.values
    X = dataset[:, 0:10].astype(float)

    Y = dataset[:, 10]

    X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=.2, random_state=1)

    scores = cross_val_score(estimator, X_train, y_train, scoring=make_scorer(accuracy_score), cv=10)
    return 'CV accuracy: %.3f +/- %.3f' % (np.mean(scores), np.std(scores))


def write(st):
    File_Object = open("output.csv", "a")
    File_Object.write(st + "\n")
    print(st)
    File_Object.close()


def write_to_file(estimator):
    df2 = pd.read_csv('test.csv', header=None)
    ds2 = df2.values
    df = pd.read_csv("train.csv", header=None)
    df = df.sample(frac=1).reset_index(drop=True)
    data_set = df.values
    X = data_set[:, 0:10].astype(float)

    Y = data_set[:, 10]
    X_predictions = ds2[:, 0:10]
    X_index = ds2[:, 10]

    estimator.fit(X, Y)
    preds = estimator.predict(X_predictions)

    write_to_file("ID,DEFCON_Level")

    for x in range(0, len(preds)):
        write_to_file(str(int(X_index[x])) + "," + str(int(preds[x])))


def plot_learning_curve(estimator, title, X, y, axes=None, ylim=None, cv=None,
                        n_jobs=None, train_sizes=np.linspace(.1, 1.0, 5)):
    """
    Generate 3 plots: the test and training learning curve, the training
    samples vs fit times curve, the fit times vs score curve.

    Parameters
    ----------
    estimator : object type that implements the "fit" and "predict" methods
        An object of that type which is cloned for each validation.

    title : string
        Title for the chart.

    X : array-like, shape (n_samples, n_features)
        Training vector, where n_samples is the number of samples and
        n_features is the number of features.

    y : array-like, shape (n_samples) or (n_samples, n_features), optional
        Target relative to X for classification or regression;
        None for unsupervised learning.

    axes : array of 3 axes, optional (default=None)
        Axes to use for plotting the curves.

    ylim : tuple, shape (ymin, ymax), optional
        Defines minimum and maximum yvalues plotted.

    cv : int, cross-validation generator or an iterable, optional
        Determines the cross-validation splitting strategy.
        Possible inputs for cv are:
          - None, to use the default 5-fold cross-validation,
          - integer, to specify the number of folds.
          - :term:`CV splitter`,
          - An iterable yielding (train, test) splits as arrays of indices.

        For integer/None inputs, if ``y`` is binary or multiclass,
        :class:`StratifiedKFold` used. If the estimator is not a classifier
        or if ``y`` is neither binary nor multiclass, :class:`KFold` is used.

        Refer :ref:`User Guide <cross_validation>` for the various
        cross-validators that can be used here.

    n_jobs : int or None, optional (default=None)
        Number of jobs to run in parallel.
        ``None`` means 1 unless in a :obj:`joblib.parallel_backend` context.
        ``-1`` means using all processors. See :term:`Glossary <n_jobs>`
        for more details.

    train_sizes : array-like, shape (n_ticks,), dtype float or int
        Relative or absolute numbers of training examples that will be used to
        generate the learning curve. If the dtype is float, it is regarded as a
        fraction of the maximum size of the training set (that is determined
        by the selected validation method), i.e. it has to be within (0, 1].
        Otherwise it is interpreted as absolute sizes of the training sets.
        Note that for classification the number of samples usually have to
        be big enough to contain at least one sample from each class.
        (default: np.linspace(0.1, 1.0, 5))
    """
    if axes is None:
        _, axes = plt.subplots(1, 3, figsize=(20, 5))

    axes[0].set_title(title)
    if ylim is not None:
        axes[0].set_ylim(*ylim)
    axes[0].set_xlabel("Training examples")
    axes[0].set_ylabel("Score")

    train_sizes, train_scores, test_scores, fit_times, _ = \
        learning_curve(estimator, X, y, cv=cv, n_jobs=n_jobs,
                       train_sizes=train_sizes,
                       return_times=True)
    train_scores_mean = np.mean(train_scores, axis=1)
    train_scores_std = np.std(train_scores, axis=1)
    test_scores_mean = np.mean(test_scores, axis=1)
    test_scores_std = np.std(test_scores, axis=1)
    fit_times_mean = np.mean(fit_times, axis=1)
    fit_times_std = np.std(fit_times, axis=1)

    # Plot learning curve
    axes[0].grid()
    axes[0].fill_between(train_sizes, train_scores_mean - train_scores_std,
                         train_scores_mean + train_scores_std, alpha=0.1,
                         color="r")
    axes[0].fill_between(train_sizes, test_scores_mean - test_scores_std,
                         test_scores_mean + test_scores_std, alpha=0.1,
                         color="g")
    axes[0].plot(train_sizes, train_scores_mean, 'o-', color="r",
                 label="Training score")
    axes[0].plot(train_sizes, test_scores_mean, 'o-', color="g",
                 label="Cross-validation score")
    axes[0].legend(loc="best")

    # Plot n_samples vs fit_times
    axes[1].grid()
    axes[1].plot(train_sizes, fit_times_mean, 'o-')
    axes[1].fill_between(train_sizes, fit_times_mean - fit_times_std,
                         fit_times_mean + fit_times_std, alpha=0.1)
    axes[1].set_xlabel("Training examples")
    axes[1].set_ylabel("fit_times")
    axes[1].set_title("Scalability of the model")

    # Plot fit_time vs score
    axes[2].grid()
    axes[2].plot(fit_times_mean, test_scores_mean, 'o-')
    axes[2].fill_between(fit_times_mean, test_scores_mean - test_scores_std,
                         test_scores_mean + test_scores_std, alpha=0.1)
    axes[2].set_xlabel("fit_times")
    axes[2].set_ylabel("Score")
    axes[2].set_title("Performance of the model")

    return plt


def average(listX):
    terms = len(listX)
    total = 0
    for x in range(terms):
        total += listX[x]
    return float(total) / terms


def max(listX):
    max = listX[0]
    for x in range(len(listX)):
        if listX[x] > max:
            max = listX[x]
    return max


def min(listX):
    min = listX[0]
    for x in range(len(listX)):
        if listX[x] < min:
            min = listX[x]
    return min


def score(y_true, y_pred):
    return sklearn.metrics.f1_score(y_true, y_pred, labels=None, pos_label=None, average='weighted')


def print_for_all(function, name):
    one = ''
    two = ''
    three = ''
    four = ''
    five = ''
    for x in range(len(ones[0])):
        one += str(function([ones[i][x] for i in range(len(ones))])) + ' '
    one = one[0:-1]
    for x in range(len(twos[0])):
        two += str(function([twos[i][x] for i in range(len(twos))])) + ' '
    two = two[0:-1]
    for x in range(len(threes[0])):
        three += str(function([threes[i][x] for i in range(len(threes))])) + ' '
    three = three[0:-1]
    for x in range(len(fours[0])):
        four += str(function([fours[i][x] for i in range(len(fours))])) + ' '
    four = four[0:-1]
    for x in range(len(fives[0])):
        five += str(function([fives[i][x] for i in range(len(fives))])) + ' '
    five = five[0:-1]
    print(name)
    print("One: " + one)
    print("Two: " + two)
    print("Three: " + three)
    print("Four: " + four)
    print("Five: " + five)


'''
fives = []
fours = []
threes = []
twos = []
ones = []

df = pd.read_csv("train.csv", header=None)
data_set = df.values
X = data_set[:, 0:10].astype(float)
Y = data_set[:, 10]

for x in range(len(Y)):
    if Y[x] == 1:
        ones.append(X[x])
    elif Y[x] == 2:
        twos.append(X[x])
    elif Y[x] == 3:
        threes.append(X[x])
    elif Y[x] == 4:
        fours.append(X[x])
    else:
        fives.append(X[x])

print_for_all(average, 'Averages')
print_for_all(max, 'Max')
print_for_all(min, 'Min')

'''
