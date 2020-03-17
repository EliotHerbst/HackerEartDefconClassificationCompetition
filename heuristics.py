import pandas as pd

def write_to_file(st):
    File_Object = open("output.csv", "a")
    File_Object.write(st + "\n")
    print(st)
    File_Object.close()
from Defcon_Pipeline import PIPELINE


class Heuristics:
    alpha = 0
    recursive = False

    def __init__(self):
        pass

    def fit(self, x, target):
        pass

    def predict(self, x):
        predictions = []
        for y in range(len(x)):
            vars = x[y]
            predictions.append(prediction_alg(vars))

    def get_params(self, deep=True):
        deep
        # suppose this estimator has parameters "alpha" and "recursive"
        return {"alpha": self.alpha, "recursive": self.recursive}

    def set_params(self, **parameters):
        return self


def prediction_alg(xs):
    return 1


df2 = pd.read_csv('test.csv', header=None)
ds2 = df2.values
X_index = ds2[:, 10]

write_to_file("ID,DEFCON_Level")

for x in range(0, len(X_index)):
    write_to_file(str(int(X_index[x])) + "," + str(5))