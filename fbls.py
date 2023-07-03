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
        std = 1
        ps = []
        y = np.zeros((train_x.shape[0], NumFuzz * NumRule))

        H1 = train_x
        center_list = []
        for i in range(1,NumFuzz+1):
            b1 = Alpha[i-1]
            t_y = np.zeros((train_x.shape[0], NumRule))

            cntr, _, _, _, _, _, _ = fuzz.cluster.cmeans(data=train_x,c=NumRule, m=4, error=0.005, maxiter=100)
            center = cntr.T
            #CENTER = cntr.T
            #CENTER = KMeans(n_clusters=NumRule).fit(train_x).cluster_centers_

            for j in range(0,train_x.shape[0]):

                train_x_repmat = train_x[j,:]
                #center = encontrar_mayor_membresia(train_x_repmat, CENTER)
                distances = np.linalg.norm(train_x_repmat - center, axis=1)
                MF = np.exp(-distances**2 / std)
                #MF = np.exp(-np.power( np.subtract(train_x_repmat, center, dtype=np.float64), 2) / std)
                if np.ndim(MF) != 1:
                    MF = np.prod(MF)
                    MF = MF / np.sum(MF)

                for k in range(NumRule):
                    t_y[j, k] = MF[k] * np.dot(train_x[j, :], b1)


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

        #finaliza el tiempo de entrenamiento



        NetoutTrain = T3 @ beta



        T3 = []

        yy = result_tra(NetoutTrain)
        print('222222222#################')
        print(yy)
        print('#################')
        train_yy = result_tra(train_y)

        print(train_yy)
        print(np.sum(yy == train_yy))
        print(train_yy.shape[0])
        print('#################')


        TrainingAccuracy = np.sum(yy == train_yy) / train_yy.shape[0]
        print("Training Accuracy is:", TrainingAccuracy, "%")

        '''# Inicio de la prueba
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
        '''
        #return NetoutTest, Training_time, Testing_time, TrainingAccuracy, TestingAccuracy * 100
        return 0, 0, 0, 0, 0 * 100


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
    for i in range(x.shape[0]):
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
        distancia = np.linalg.norm(centroide - elemento[0])
        # manhattan       distancia = cityblock(elemento[0], centroide)
        # chebyshev        distancia = chebyshev(elemento[0], centroide)
        # minkowski      distancia = minkowski(elemento[0], centroide)
        if distancia < 0:
            distancia = distancia * -1
        membresia = 1 / (1 + distancia)
        if membresia > mayor_membresia:
            mayor_membresia = membresia
            centroide_mayor_membresia = centroide
    return centroide_mayor_membresia
