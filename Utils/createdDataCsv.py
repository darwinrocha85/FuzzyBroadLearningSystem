import torch
import pandas as pd
import numpy as np
import random
from sklearn.decomposition import PCA

class createdData():
    @staticmethod
    def creadtes_file(dataset_input, type_data_input="training", porcentaje_input=0.5):
        array_data = []
        pca = PCA(n_components=1)

        for bacth in dataset_input:
            face = bacth['face'].numpy()
            audio = bacth['audio'].numpy()
            text = bacth['text'].numpy()
            label = torch.argmax(bacth['label'], dim=-1).numpy()
            label = np.ravel(label)
            input = np.concatenate(([face], [audio], [text]), axis=0)
            input= pca.fit_transform(input)
            array_data.append([max(input[0]), max(input[1]),max(input[2]), max(label)])
           # array_data.append([min(face), max(audio), min(text), max(label)])


        # Obtener el tamaño de la lista y calcular la cantidad de elementos para cada porcentaje
        total_elements = len(array_data)
        porcentaje = int(porcentaje_input * total_elements)

        # Barajar (shuffle) la lista de forma aleatoria
        random.shuffle(array_data)

        # Dividir la lista en dos listas basadas en los porcentajes
        lista_1 = array_data[:porcentaje]
        lista_2 = array_data[porcentaje:]

        if type_data_input == "test":

            # Crear un DataFrame de pandas a partir del arreglo
            df = pd.DataFrame(lista_1)

            # Escribir el DataFrame en un archivo CSV
            df.to_csv('test_archivo_1.csv', index=False)

            # Crear un DataFrame de pandas a partir del arreglo
            df = pd.DataFrame(lista_2)

            # Escribir el DataFrame en un archivo CSV
            df.to_csv('test_archivo_2.csv', index=False)

        else:
            # Crear un DataFrame de pandas a partir del arreglo
            df = pd.DataFrame(lista_1)

            # Escribir el DataFrame en un archivo CSV
            df.to_csv('train_archivo_1.csv', index=False)

            # Crear un DataFrame de pandas a partir del arreglo
            df = pd.DataFrame(lista_2)

            # Escribir el DataFrame en un archivo CSV
            df.to_csv('train_archivo_2.csv', index=False)
