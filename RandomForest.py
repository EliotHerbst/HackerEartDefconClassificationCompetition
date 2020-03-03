import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score

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


regressor = RandomForestRegressor(n_estimators=20, random_state=0)
regressor.fit(X_train, y_train)
y_pred = regressor.predict(X_test)


print(pd.crosstab(y_test, y_pred, rownames=['Actual Result'], colnames=['Predicted Result']))

