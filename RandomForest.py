import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix

def write_to_file(st):
    File_Object = open("output.csv", "a")
    File_Object.write(st + "\n")
    print(st)
    File_Object.close()

# get dataset
dataframe = pd.read_csv("train.csv", header=None)
dataset = dataframe.values
X = dataset[:, 0:10].astype(float)
Y = dataset[:, 10]

from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.2, random_state=0)

# standardize
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)


clf = RandomForestClassifier()
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

