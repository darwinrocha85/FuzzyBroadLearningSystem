import numpy as np
import time
from random import seed
from sklearn.preprocessing import MinMaxScaler
from sklearn.cluster import KMeans
import csv
from skfuzzy import cmeans
import numpy as np
from scipy.linalg import lstsq
import numpy as np

import numpy as np
import skfuzzy as fuzz

class FBLS:
    def __init__(self, input_size, output_size):
        self.input_size = input_size
        self.output_size = output_size
        self.rule_base = fuzz.RuleBase()
        self.enhancement_layer = None

    def fit(self, X, y):
        for i in range(self.output_size):
            for j in range(self.input_size):
                rule = fuzz.Rule(np.full(self.input_size, j), fuzz.Gaussian(0.5, 0.1))
                self.rule_base.add_rule(rule)

        self.enhancement_layer = fuzz.Defuzzifier(np.arange(0, 1, 0.01), "centroid")

    def predict(self, X):
        fuzzy_outputs = []
        for i in range(self.output_size):
            fuzzy_output = self.rule_base.activate(X)
            fuzzy_outputs.append(fuzzy_output)

        enhanced_output = self.enhancement_layer.defuzzify(fuzzy_outputs)
        return enhanced_output


mat_path = ""


train_x = []
train_y = []
test_x = []
test_y = []

with open(mat_path + "train_archivo_1.csv", "r") as file:
    csvreader = csv.reader(file, delimiter=",")
    for row in csvreader:
        train_x.append(np.float64(row))

with open(mat_path + "train_archivo_2.csv", "r") as file:
    csvreader = csv.reader(file, delimiter=",")
    for row in csvreader:
        train_y.append(np.float64(row))

with open(mat_path + "test_archivo_1.csv", "r") as file:
    csvreader = csv.reader(file, delimiter=",")
    for row in csvreader:
        test_x.append(np.float64(row))

with open(mat_path + "test_archivo_2.csv", "r") as file:
    csvreader = csv.reader(file, delimiter=",")
    for row in csvreader:
        test_y.append(np.float64(row))

train_x = np.array(train_x)
train_y = np.array(train_y)
test_x = np.array(test_x)
test_y = np.array(test_y)

fbls = FBLS(1, 1)
fbls.fit(train_x, train_y)

y_pred = fbls.predict(test_x)

print(y_pred)
