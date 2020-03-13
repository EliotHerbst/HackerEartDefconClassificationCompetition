import pandas as pd
from sklearn import svm
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
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

categories = [1, 2, 3, 4, 5]

X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.2, random_state=0)

clf = svm.SVC(decision_function_shape='ovo')
clf.fit(X_train, y_train)

pred = clf.predict(X_test)

from sklearn.metrics import confusion_matrix, accuracy_score

mat = confusion_matrix(y_test, pred)
sns.heatmap(mat.T, square=True, annot=True, fmt='d', cbar=False,
            xticklabels=categories, yticklabels=categories)
plt.xlabel('true label')
plt.ylabel('predicted label')

plt.show()

print(accuracy_score(y_test, pred))
