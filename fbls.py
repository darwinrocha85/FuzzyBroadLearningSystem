import numpy as np
from sklearn.preprocessing import MinMaxScaler
from sklearn.cluster import KMeans
import numpy as np
from sklearn import preprocessing
import skfuzzy as fuzz


class MapMinMaxApplier(object):
    def __init__(self, slope, intercept):
        self.slope = slope
        self.intercept = intercept

    def __call__(self, x):
        return x * self.slope + self.intercept

    def reverse(self, y):
        return (y - self.intercept) / self.slope


class fbls():
    @staticmethod
    def bls_train(
        train_x, train_y, test_x, test_y, Alpha, WeightEnhan, s, C, NumRule, NumFuzz
    ):

        beta, TrainingAccuracy = calculated_train(train_x, train_y, Alpha, WeightEnhan, s, C, NumRule, NumFuzz)
        TestgAccuracy =calculated_test(test_x, test_y, Alpha, WeightEnhan, s,NumRule, NumFuzz, beta)

        return 0, 0, 0, TrainingAccuracy , TestgAccuracy


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
    y=[]
    for i in range(0,x.shape[0]):
        tt = np.abs(np.round(max(x[i,:])))
        if tt < 0:
            tt = tt * -1
        y.append(tt)

    y = np.transpose(y)
    return y


def encontrar_mayor_membresia(elemento, centroides):
    from scipy.spatial import distance
    from scipy.spatial.distance import cityblock
    from scipy.spatial.distance import chebyshev
    from scipy.spatial.distance import minkowski
    mayor_membresia = -1
    centroide_mayor_membresia = None
    for centroide in centroides:
        #euclides
        distancia = np.linalg.norm((centroide - elemento[0]))
        membresia = 1 / (1 + distancia)
        if membresia > mayor_membresia:
            mayor_membresia = membresia
            centroide_mayor_membresia = centroide
    return distancia, centroide_mayor_membresia



def calculated_train(train_x, train_y, Alpha, WeightEnhan, s, C, NumRule, NumFuzz):
    std = 1
    ps = []
    y = np.zeros((train_x.shape[0], NumFuzz * NumRule))

    H1 = train_x
    center_list = []
    t_y = np.zeros((train_x.shape[0], NumRule))
    for i in range(1,NumFuzz+1):
        b1 = Alpha[i-1]


        CENTER = KMeans(n_clusters=NumRule, n_init=10).fit(train_x).cluster_centers_

        for j in range(0,train_x.shape[0]):

            distances, center = encontrar_mayor_membresia(train_x[j,:], CENTER)
            MF = np.exp(-distances**2 / std)
            if np.ndim(MF) != 1:
                MF = np.prod(MF)
                MF = MF / np.sum(MF)


            t_y[j, :] = MF * np.dot(train_x[j, :], b1)


        center_list.append(center)

        T1, ps1 =  mapminmax(t_y.T,0,1)
        T1 = T1.T

        ps.append(ps1)

        y[:, NumRule * (i - 1) : NumRule * i] = T1




    H1 = []
    T1= []


    H2 = np.concatenate((y, 0.1 * np.ones((y.shape[0], 1))), axis=1)

    #print(WeightEnhan)
    T2 = H2 @ WeightEnhan
    l2 = np.max(T2)
    l2 = s / l2

    T2 = np.tanh(T2 * l2)
    T3 = np.concatenate((y, T2), axis=1)
    H2 =[]
    T2 = []

    T3_transpose = T3.T
    eye_matrix = np.eye(T3_transpose.shape[0])
    beta = np.linalg.inv(T3_transpose @ T3 + eye_matrix * C) @ (T3_transpose @ train_y)





    NetoutTrain = T3 @ beta



    T3 = []

    yy = result_tra(NetoutTrain)
    train_yy = result_tra(train_y)

    TrainingAccuracy = np.sum(yy == train_yy) / train_yy.shape[0]
    print("Training Accuracy is:", TrainingAccuracy *100, "%")
    return beta, TrainingAccuracy


 #**********************************************************************************************************************
def calculated_test(test_x, test_y, Alpha, WeightEnhan, s, NumRule, NumFuzz, beta):
    std = 1
    ps = []
    y = np.zeros((test_x.shape[0], NumFuzz * NumRule))

    H1 = test_x
    center_list = []
    t_y = np.zeros((test_x.shape[0], NumRule))
    for i in range(1,NumFuzz+1):
        b1 = Alpha[i-1]


        CENTER = KMeans(n_clusters=NumRule, n_init=10).fit(test_x).cluster_centers_

        for j in range(0,test_x.shape[0]):

            distances, center = encontrar_mayor_membresia(test_x[j,:], CENTER)
            MF = np.exp(-distances**2 / std)
            if np.ndim(MF) != 1:
                MF = np.prod(MF)
                MF = MF / np.sum(MF)


            t_y[j, :] = MF * np.dot(test_x[j, :], b1)


        center_list.append(center)

        T1, ps1 =  mapminmax(t_y.T,0,1)
        T1 = T1.T

        ps.append(ps1)

        y[:, NumRule * (i - 1) : NumRule * i] = T1




    H1 = []
    T1= []


    H2 = np.concatenate((y, 0.1 * np.ones((y.shape[0], 1))), axis=1)

    #print(WeightEnhan)
    T2 = H2 @ WeightEnhan
    l2 = np.max(T2)
    l2 = s / l2

    T2 = np.tanh(T2 * l2)
    T3 = np.concatenate((y, T2), axis=1)
    H2 =[]
    T2 = []


    NetoutTrain = T3 @ beta



    T3 = []

    yy = result_tra(NetoutTrain)
    train_yy = result_tra(test_y)

    TrainingAccuracy = np.sum(yy == train_yy) / train_yy.shape[0]
    print("Testing Accuracy is:", TrainingAccuracy *100, "%")
    return TrainingAccuracy
