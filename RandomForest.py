import matplotlib.pyplot as plt
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import KFold, cross_val_score
from sklearn.preprocessing import StandardScaler


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

clf = RandomForestClassifier(n_estimators=500)

n_estimators = [1, 2, 4, 8, 16, 32, 64, 100, 200]
train_results = []
test_results = []
for estimator in n_estimators:
    rf = RandomForestClassifier(n_estimators=estimator, n_jobs=-1)
    kfold = KFold(n_splits=10, shuffle=True)
    results = cross_val_score(clf, X, Y, cv=kfold)
    train_results.append(results.mean)
    test_results.append(results.std)

line1, = plt.plot(n_estimators, train_results, 'b', label='Accuracy')
line2, = plt.plot(n_estimators, test_results, 'r', label='Standard Deviation')
plt.ylabel('Accuracy')
plt.xlabel('N-Estimators')
plt.show()

'''(n_estimators=100, criterion='gini', max_depth=None, min_samples_split=2, 
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
# write_to_file("ID,DEFCON_Level")
'''
for x in range(0, len(test_values_x)):
    write_to_file(str(int(x_ids[x])) + "," + str(int(predictions[x])))
'''
kfold = KFold(n_splits=10, shuffle=True)
results = cross_val_score(clf, X, Y, cv=kfold)
print("Baseline: %.2f%% (%.2f%%)" % (results.mean() * 100, results.std() * 100))
