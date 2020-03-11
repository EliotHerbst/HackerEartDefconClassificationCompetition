import pandas as pd
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis, LinearDiscriminantAnalysis
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import StandardScaler


def write_to_file(st):
    File_Object = open("output.csv", "a")
    File_Object.write(st + "\n")
    print(st)
    File_Object.close()


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

lda_model = LinearDiscriminantAnalysis()
lda_preds = lda_model.fit(X_train, y_train).predict(X_test)
lda_acc = accuracy_score(y_test, lda_preds)
print('LDA Accuracy: {}'.format(lda_acc))

qda_model = QuadraticDiscriminantAnalysis()
qda_preds = qda_model.fit(X_train, y_train).predict(X_test)
qda_acc = accuracy_score(y_test, qda_preds)
print('QDA Accuracy: {}'.format(qda_acc))

rda_model = LinearDiscriminantAnalysis(solver='lsqr', shrinkage='auto')
rda_preds = rda_model.fit(X_train, y_train).predict(X_test)
rda_acc = accuracy_score(y_test, rda_preds)
print('RDA Accuracy: {}'.format(rda_acc))

logreg_model = LogisticRegression()
logreg_preds = logreg_model.fit(X_train, y_train).predict(X_test)
logreg_acc = accuracy_score(y_test, logreg_preds)
print('Logistic Regression Accuracy: {}'.format(logreg_acc))

dataframe = pd.read_csv("test.csv", header=None)
dataset = dataframe.values
tests = dataset[:, 0:10].astype(float)
X_tests = sc.fit_transform(tests)
x_ids = dataset[:, 10]

test_values_x = logreg_model.fit(X_train, y_train).predict(X_tests)
write_to_file("ID,DEFCON_Level")

for x in range(0, len(test_values_x)):
    write_to_file(str(int(x_ids[x])) + "," + str(int(test_values_x[x])))