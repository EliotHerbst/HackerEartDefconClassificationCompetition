import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, roc_curve
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import KFold, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import log_loss
from sklearn.model_selection import GridSearchCV


def write_to_file(st):
    File_Object = open("output.csv", "a")
    File_Object.write(st + "\n")
    print(st)
    File_Object.close()


# get dataset
dataframe = pd.read_csv("train.csv", header=None)
dataframe = dataframe.sample(frac=1).reset_index(drop=True)
dataset = dataframe.values
X = dataset[:, 0:10].astype(float)
Y = dataset[:, 10]

from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.2, random_state=0)

# standardize
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)

clf = RandomForestClassifier(n_estimators=20, min_samples_split=0.6, min_samples_leaf=0.2,max_depth=12)
'''
n_estimator = [1, 20, 50, 100, 200, 250, 300, 500, 1000, 2000]
max_depths = np.linspace(1, 32, 32, endpoint=True)
min_samples_splits = np.linspace(0.1, 1.0, 10, endpoint=True)
min_samples_leafs = np.linspace(0.1, 0.5, 5, endpoint=True)

param_grid = dict(n_estimators=n_estimator, max_depth=max_depths, min_samples_split=min_samples_splits,
                  min_samples_leaf=min_samples_leafs)

import time

grid = GridSearchCV(estimator=clf, param_grid=param_grid, cv=3, n_jobs=-1)

start_time = time.time()
grid_result = grid.fit(X, Y)
# Summarize results
print("Best: %f using %s" % (grid_result.best_score_, grid_result.best_params_))
print("Execution time: " + str((time.time() - start_time)) + ' ms')
'''
from sklearn.metrics import roc_curve, auc
from sklearn import preprocessing

# y_train = preprocessing.label_binarize(y_train, classes=[1, 2, 3, 4, 5])
# y_test = preprocessing.label_binarize(y_test, classes=[1, 2, 3, 4, 5])
'''
min_samples_splits = np.linspace(0.1, 1.0, 10, endpoint=True)
train_log_loss = []
train_accuracy = []
test_log_loss = []
test_accuracy = []
for estimator in min_samples_splits:
    rf = RandomForestClassifier(n_estimators=128, max_depth=12, min_samples_split=estimator, n_jobs=-1)
    rf.fit(X_train, y_train)
    train_pred = rf.predict(X_train)
    log_loss_train = log_loss(y_train, train_pred)
    accuracy_train = accuracy_score(y_train, train_pred)
    train_log_loss.append(log_loss_train)
    train_accuracy.append(accuracy_train)
    y_pred = rf.predict(X_test)
    log_loss_test = log_loss(y_test, y_pred)
    accuracy_test = accuracy_score(y_test, y_pred)
    test_log_loss.append(log_loss_test)
    test_accuracy.append(accuracy_test)
from matplotlib.legend_handler import HandlerLine2D

line1, = plt.plot(min_samples_splits, train_log_loss, 'b', label="Train Log-Loss")
line2, = plt.plot(min_samples_splits, test_log_loss, 'r', label='Test Log-Loss')
line3, = plt.plot(min_samples_splits, train_accuracy, 'm', label = 'Train Accuracy')
line4, = plt.plot(min_samples_splits, test_accuracy, 'y', label = 'Test Accuracy')
plt.legend(handler_map={line1: HandlerLine2D(numpoints=2)})
plt.ylabel('Score')
plt.xlabel('n_estimators')
plt.show()
'''
'''
(n_estimators=100, criterion='gini', max_depth=None, min_samples_split=2, 
min_samples_leaf=1, min_weight_fraction_leaf=0.0, max_features='auto', max_leaf_nodes=None, 
min_impurity_decrease=0.0, min_impurity_split=None, bootstrap=True, oob_score=False, 
n_jobs=None, random_state=None, verbose=0, warm_start=False, class_weight=None, ccp_alpha=0.0, 
max_samples=None)[source]¶
'''

clf.fit(X_train, y_train)
predictions = clf.predict(X_test)

print("Train Accuracy :: ", accuracy_score(y_train, clf.predict(X_train)))

print("Test Accuracy  :: ", accuracy_score(y_test, predictions))

print(" Confusion matrix ", confusion_matrix(y_test, predictions))

dataframe = pd.read_csv("test.csv", header=None)
dataset = dataframe.values
x_ids = dataset[:, 10]
test_values_x = dataset[:, 0:10].astype(float)
test_values_x = sc.transform(test_values_x)
predictions = clf.predict(test_values_x)
write_to_file("ID,DEFCON_Level")

for x in range(0, len(test_values_x)):
    write_to_file(str(int(x_ids[x])) + "," + str(int(predictions[x])))

kfold = KFold(n_splits=10, shuffle=True)
results = cross_val_score(clf, X, Y, cv=kfold)
print("Baseline: %.2f%% (%.2f%%)" % (results.mean() * 100, results.std() * 100))
