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


class MapMinMaxApplier(object):
    def __init__(self, slope, intercept):
        self.slope = slope
        self.intercept = intercept

    def __call__(self, x):
        return x * self.slope + self.intercept

    def reverse(self, y):
        return (y - self.intercept) / self.slope


def mapminmax(x, ymin=-1, ymax=+1):
    x = np.asanyarray(x)
    xmax = x.max(axis=-1)
    xmin = x.min(axis=-1)
    if (xmax == xmin).any():
        print("some rows have no variation")
    slope = ((ymax - ymin) / (xmax - xmin))[:, np.newaxis]
    intercept = (-xmin * (ymax - ymin) / (xmax - xmin))[:, np.newaxis] + ymin
    ps = MapMinMaxApplier(slope, intercept)
    return ps(x), ps


def tansig(x):
    return np.tanh(x)


def result_tra(x):
    y = np.zeros(x.shape[0], dtype=int)
    for i in range(x.shape[0]):
        y[i] = np.argmax(x[i, :])
    return y

def encontrar_mayor_membresia(elemento, centroides):
    from scipy.spatial import distance
    from scipy.spatial.distance import cityblock
    from scipy.spatial.distance import chebyshev
    from scipy.spatial.distance import minkowski
    mayor_membresia = -1
    centroide_mayor_membresia = None
    for centroide in centroides:
        #euclides distancia = np.linalg.norm(centroide - elemento[0])
        # manhattan       distancia = cityblock(elemento[0], centroide)
        # chebyshev        distancia = chebyshev(elemento[0], centroide)
        # minkowski
        distancia = minkowski(elemento[0], centroide)
        if distancia < 0:
            distancia = distancia * -1
        membresia = 1 / (1 + distancia)
        if membresia > mayor_membresia:
            mayor_membresia = membresia
            centroide_mayor_membresia = centroide
    return centroide_mayor_membresia

def bls_train(
    train_x, train_y, test_x, test_y, Alpha, WeightEnhan, s, C, NumRule, NumFuzz
):
    std = 1
    ps = []
    y = np.zeros((train_x.shape[0], NumFuzz * NumRule))
    t_y = np.zeros((train_x.shape[0], NumRule))

    CENTER = KMeans(n_clusters=NumRule).fit(train_x).cluster_centers_

    for i in range(1,NumFuzz):
        b1 = Alpha[i]

        for j in range(train_x.shape[1]):

            ten2 = np.tile(train_x[j, :], (NumRule, 1))
            center = encontrar_mayor_membresia(ten2, CENTER)
            MF = np.exp(
                -np.power(
                    np.subtract(
                        ten2,
                        center,
                    ),
                    2,
                )
                / std
            )

            MF = np.prod(MF, axis=1)
            MF = MF / np.sum(MF)
            temP = np.dot(train_x[j, :], b1)

            t_y[j, :] = MF[0] * temP[0]

        [T1, ps1] = mapminmax(t_y, 0, 1)
        ps.append(ps1)

        y[:, NumRule * (i - 1) : NumRule * i] = T1

    H2 = np.column_stack((y, 0.1 * np.ones((y.shape[0], 1))))

    print(WeightEnhan)
    T2 = np.dot(H2, WeightEnhan)
    l2 = np.max(T2)
    l2 = s / l2

    T2 = np.tanh(T2 * l2)
    T3 = np.column_stack((y, T2))

    beta = np.dot(
        np.linalg.inv(np.dot(T3.T, T3) + np.eye(T3.shape[1]) * C), np.dot(T3.T, train_y)
    )

    #finaliza el tiempo de entrenamiento

    NetoutTrain = np.dot(T3, beta)

    yy = np.argmax(NetoutTrain, axis=1)
    train_yy = np.argmax(train_y, axis=1)
    TrainingAccuracy = np.sum(yy == train_yy) / train_yy.shape[0] * 100
    print("Training Accuracy is:", TrainingAccuracy, "%")

    # Inicio de la prueba
    yy1 = np.zeros((test_x.shape[0], NumFuzz * NumRule))

    for i in range(1,NumFuzz):
        b1 = Alpha[i]
        t_y = np.zeros((test_x.shape[0], NumRule))
        for j in range(test_x.shape[1]):

            ten2 = np.tile(test_x[j, :], (NumRule, 1))
            center = encontrar_mayor_membresia(ten2, CENTER)

            MF = np.exp(-((ten2 - center.T) ** 2) / std)
            MF = np.prod(MF, axis=1)
            MF = MF / np.sum(MF)
            t_y[j, :] = MF[0] * (test_x[j, :] @ b1)[0]

        ps1 = ps[i-1]
        scaler = MinMaxScaler()
        TT1 = scaler.fit_transform(t_y)
        del scaler
        yy1[:, NumRule * (i - 1) : NumRule * i] = TT1
        del ps1

    HH2 = np.hstack((yy1, 0.1 * np.ones((yy1.shape[0], 1))))
    m_1 = HH2 @ WeightEnhan
    TT2 = tansig(m_1 * l2)

    TT3 = np.hstack((yy1, TT2))
    NetoutTest = TT3 @ beta

    y = result_tra(NetoutTest)
    test_yy = result_tra(test_y)
    TestingAccuracy = np.count_nonzero(y == test_yy) / test_yy.shape[0]
    TT3 = []

    print("Testing Accuracy is :", TestingAccuracy * 100, "%")
    Training_time = 0
    Testing_time= 0

    return NetoutTest, Training_time, Testing_time, TrainingAccuracy, TestingAccuracy * 100


np.set_printoptions(precision=4, suppress=True)
# np.warnings.filterwarnings('ignore')

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


# Test set for wbc data set
n = np.random.permutation(test_x.shape[0])
test_x = test_x[n[:140], :]
test_y = test_y[n[:140], :]

train_y = (train_y - 1) * 2 + 1
test_y = (test_y - 1) * 2 + 1

C = 2**-30  # C: the regularization parameter for sparse regularization
s = 0.8  # s: the shrinkage parameter for enhancement nodes
best = 72
result = []
for NumRule in range(1, 5):  # searching range for fuzzy rules per fuzzy subsystem
    for NumFuzz in range(1, 4):  # searching range for number of fuzzy subsystems
        for NumEnhan in range(1, 2):  # searching range for enhancement nodes
            print(
                f"Fuzzy rule No. = {NumRule}, Fuzzy system No. = {NumFuzz}, Enhan. No. = {NumEnhan}"
            )
            seed(1)
            Alpha = {}
            for i in range(1,NumFuzz):
                alpha = np.random.rand(train_x.shape[1], NumRule)
                Alpha[i] = alpha
            # generating coefficients of the then part of fuzzy rules for each fuzzy system

            WeightEnhan = np.random.rand(
                NumFuzz * NumRule + 1, NumEnhan
            )  # Initializing weights connecting fuzzy subsystems with enhancement layer

            (
                NetoutTest,
                Training_time,
                Testing_time,
                TrainingAccuracy,
                TestingAccuracy,
            ) = bls_train(
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
                np.savez(
                    "optimal.npz",
                    TrainingAccuracy=TrainingAccuracy,
                    TestingAccuracy=TestingAccuracy,
                    NumRule=NumRule,
                    NumFuzz=NumFuzz,
                    NumEnhan=NumEnhan,
                    time=time_1,
                )


print(result)
X = [0,0,0,0,0]
for item in result:
    if item[4] > X[4]:
        X = item

print("Best result:")
print(X)
