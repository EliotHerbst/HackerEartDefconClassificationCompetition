import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import pandas as pd
import seaborn as sns
from sklearn.metrics import confusion_matrix, accuracy_score
from sklearn.neighbors import KNeighborsClassifier

sns.set()


def write_to_file(st):
    File_Object = open("output.csv", "a")
    File_Object.write(st + "\n")
    print(st)
    File_Object.close()


dataframe = pd.read_csv("train.csv", header=None)
dataset = dataframe.values
X = dataset[:, 0:10].astype(float)
Y = dataset[:, 10]

X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.20)

scaler = StandardScaler()
scaler.fit(X_train)

X_train = scaler.transform(X_train)
X_test = scaler.transform(X_test)

clf = KNeighborsClassifier(n_neighbors=25)
clf.fit(X_train, y_train)

pred = clf.predict(X_test)
categories = [1, 2, 3, 4, 5]
'''
error = []

# Calculating error for K values between 1 and 40
for i in range(1, 40):
    knn = KNeighborsClassifier(n_neighbors=i)
    knn.fit(X_train, y_train)
    pred_i = knn.predict(X_test)
    error.append(np.mean(pred_i != y_test))

plt.figure(figsize=(12, 6))
plt.plot(range(1, 40), error, color='red', linestyle='dashed', marker='o',
         markerfacecolor='blue', markersize=10)
plt.title('Error Rate K Value')
plt.xlabel('K Value')
plt.ylabel('Mean Error')

plt.show()
 
mat = confusion_matrix(y_test, pred)
sns.heatmap(mat.T, square=True, annot=True, fmt='d', cbar=False,
            xticklabels=categories, yticklabels=categories)
plt.xlabel('true label')
plt.ylabel('predicted label')

plt.show()
'''
print(accuracy_score(y_test, pred))

write_to_file("ID,DEFCON_Level")
dataframe = pd.read_csv("test.csv", header=None)
dataset = dataframe.values
x_ids = dataset[:, 10]
xs = dataset[:, 0:10]
scaler = StandardScaler()
scaler.fit(xs)

xs = scaler.transform(xs)
test_values_x = clf.predict(xs)
for x in range(0, len(x_ids)):
    write_to_file(str(int(x_ids[x])) + "," + str(int(test_values_x[x])))
