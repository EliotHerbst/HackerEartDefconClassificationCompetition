import pandas as pd
from sklearn.ensemble import AdaBoostClassifier, GradientBoostingClassifier
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split, GridSearchCV
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from sklearn.metrics import confusion_matrix, accuracy_score
from sklearn.preprocessing import StandardScaler

from tools import score

sns.set()


def write_to_file(st):
    File_Object = open("output.csv", "a")
    File_Object.write(st + "\n")
    print(st)
    File_Object.close()


dataframe = pd.read_csv("train.csv", header=None)
dataframe = dataframe.sample(frac=1).reset_index(drop=True)
dataset = dataframe.values
X = dataset[:, 0:10].astype(float)
Y = dataset[:, 10]

X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.20)

scaler = StandardScaler()
scaler.fit(X_train)

X_train = scaler.transform(X_train)
X_test = scaler.transform(X_test)
'''
# AdaBoost
clf = AdaBoostClassifier(n_estimators=4, algorithm='SAMME')
clf.fit(X_train, y_train)

from sklearn.ensemble import GradientBoostingClassifier

loss = ['deviance']
learning_rate = [0.15, 0.1, 0.05, 0.01, 0.005, 0.001]
n_estimators = [100, 250, 500, 750, 1000, 1250, 1500, 1750]
max_depth = [2, 3, 4, 5, 6, 7]

param_grid = {'max_features': [2, 3, 4, 5, 6, 7], 'subsample': [0.7, 0.75, 0.8, 0.85, 0.9, 0.95, 1]}

import time

grid = GridSearchCV(estimator=GradientBoostingClassifier(n_estimators=1000, max_depth=7, loss='deviance',
                                                         learning_rate=0.01, min_samples_split=4, min_samples_leaf=5)
                    , scoring='f1_weighted', param_grid=param_grid, cv=3,
                    n_jobs=-1)

start_time = time.time()
grid_result = grid.fit(X_train, y_train)
# Summarize results
print("Best: %f using %s" % (grid_result.best_score_, grid_result.best_params_))
print("Execution time: " + str((time.time() - start_time)) + ' ms')
'''
clf = GradientBoostingClassifier(n_estimators=1000, max_depth=7, loss='deviance', learning_rate=0.01)
clf.fit(X_train, y_train)

labels = clf.predict(X_test)
categories = [1, 2, 3, 4, 5]

mat = confusion_matrix(y_test, labels)
sns.heatmap(mat.T, square=True, annot=True, fmt='d', cbar=False,
            xticklabels=categories, yticklabels=categories)
plt.xlabel('true label')
plt.ylabel('predicted label')

plt.show()

accuracy = score(y_test, labels)
print(accuracy)

dataframe = pd.read_csv("test.csv", header=None)
dataset = dataframe.values
tests = dataset[:, 0:10].astype(float)
sc = StandardScaler()

X_tests = sc.fit_transform(tests)
x_ids = dataset[:, 10]

test_values_x = clf.predict(X_tests)
write_to_file("ID,DEFCON_Level")

for x in range(0, len(test_values_x)):
    write_to_file(str(int(x_ids[x])) + "," + str(int(test_values_x[x])))
