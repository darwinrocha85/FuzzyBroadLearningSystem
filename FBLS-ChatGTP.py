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

class FuzzyBroadLearningSystem:
    def __init__(self, num_rules, learning_rate=0.01, epochs=100):
        self.num_rules = num_rules
        self.learning_rate = learning_rate
        self.epochs = epochs
        self.rules = []

    def fit(self, X, y):
        # Inicialización de los parámetros del modelo
        self._initialize_parameters(X.shape[1])

        # Entrenamiento del modelo
        for epoch in range(self.epochs):
            output = self._forward_pass(X)
            self._update_parameters(X, output, y)

    def predict(self, X):
        output = self._forward_pass(X)
        return output

    def _initialize_parameters(self, num_features):
        for _ in range(self.num_rules):
            rule = {
                'centers': np.random.uniform(size=num_features),
                'sigmas': np.random.uniform(size=num_features),
                'weights': np.random.uniform()
            }
            self.rules.append(rule)

    def _forward_pass(self, X):
        output = np.zeros(X.shape[0])
        for rule in self.rules:
            memberships = self._compute_memberships(X, rule['centers'], rule['sigmas'])
            output += memberships * rule['weights']
        return output

    def _compute_memberships(self, X, centers, sigmas):
        memberships = []
        for i in range(X.shape[1]):
            membership = np.exp(-((X[:, i] - centers[i]) ** 2) / (2 * sigmas[i] ** 2))
            memberships.append(membership)
        return np.prod(memberships, axis=0)

    def _update_parameters(self, X, output, y):
        error = y.shape[0] - output
        for i in range(self.num_rules):
            memberships = self._compute_memberships(X, self.rules[i]['centers'], self.rules[i]['sigmas'])
            self.rules[i]['weights'] += self.learning_rate * np.sum(error * memberships)
            delta_centers = self.learning_rate * np.sum((error * memberships)[:, np.newaxis] * (X - self.rules[i]['centers']), axis=0)
            delta_sigmas = self.learning_rate * np.sum((error * memberships)[:, np.newaxis] * ((X - self.rules[i]['centers']) ** 2 - self.rules[i]['sigmas'] ** 2) / self.rules[i]['sigmas'] ** 3, axis=0)
            self.rules[i]['centers'] += delta_centers
            self.rules[i]['sigmas'] += delta_sigmas

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

# Crear una instancia del modelo FBLS
model = FuzzyBroadLearningSystem(num_rules=5, learning_rate=0.1, epochs=100)

# Entrenar el modelo
model.fit(train_x, train_y)

# Predecir
predictions = model.predict(test_y)

print("Predicciones:")
for i, prediction in enumerate(predictions):
    print(f"Instancia {i+1}: {prediction}")
