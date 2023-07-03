from turtle import position
import numpy as np
import time
from random import seed
from sklearn.preprocessing import MinMaxScaler
from sklearn.cluster import KMeans
import csv
from skfuzzy import cmeans
import numpy as np
from scipy.linalg import lstsq
from fbls import fbls

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

C = 2**-10  # C: the regularization parameter for sparse regularization
s = 0.8  # s: the shrinkage parameter for enhancement nodes
best = 72
result = []
for NumRule in range(1, 20):  # searching range for fuzzy rules per fuzzy subsystem
    for NumFuzz in range(1, 20):  # searching range for number of fuzzy subsystems
        for NumEnhan in range(1, 20):  # searching range for enhancement nodes
            print(
                f"Fuzzy rule No. = {NumRule}, Fuzzy system No. = {NumFuzz}, Enhan. No. = {NumEnhan}"
            )
            seed(1)
            Alpha = {}
            for i in range(0,NumFuzz):
                alpha = np.random.rand(train_x.shape[1], NumRule)
                Alpha[i] = alpha
            # generating coefficients of the then part of fuzzy rules for each fuzzy system

            WeightEnhan = np.random.rand( NumFuzz * NumRule + 1, NumEnhan)  # Initializing weights connecting fuzzy subsystems with enhancement layer

            (
                NetoutTest,
                Training_time,
                Testing_time,
                TrainingAccuracy,
                TestingAccuracy,
            ) = fbls.bls_train(
                train_x,
                train_y,
                test_x,
                test_y,
                Alpha,
                WeightEnhan,
                s,
                C,
                NumRule,
                NumFuzz,
            )

            time_1 = Training_time + Testing_time
            result.append(
                [NumRule, NumFuzz, NumEnhan, TrainingAccuracy, TestingAccuracy]
            )
            if best < TestingAccuracy:
                best = TestingAccuracy


print(TestingAccuracy)
print(result)
X = [0,0,0,0,0]
for item in result:
    if item[4] > X[4]:
        X = item

print("Best result:")
print(X)
