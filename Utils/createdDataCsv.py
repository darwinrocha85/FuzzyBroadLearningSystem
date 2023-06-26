import torch
import pandas as pd
import numpy as np
import random
from sklearn.decomposition import PCA
from Utils.enumerated import type_reduction
from sklearn.manifold import TSNE

class createdData():
    @staticmethod
    def creadtes_file(dataset_input, type_data_input="training", porcentaje_input=0.5, type_reduction_input=type_reduction.not_reduction):
        array_data = []
        pca = PCA(n_components=1)
        tsne = TSNE(n_components=1, verbose=1, perplexity=2)
        for bacth in dataset_input:
            face = bacth['face'].numpy()
            audio = bacth['audio'].numpy()
            text = bacth['text'].numpy()
            label = torch.argmax(bacth['label'], dim=-1).numpy()
            label = np.ravel(label)
            if type_reduction_input == type_reduction.not_reduction:
                input = np.concatenate((face,audio, text, label), axis=0)
                array_data.append(input)
            elif type_reduction_input == type_reduction.take_minimus:
                input = [min(face),min(audio), min(text), min(label)]
                array_data.append(input)
            elif type_reduction_input == type_reduction.PCA:
                input = np.concatenate(([face], [audio], [text]), axis=0)
                input= pca.fit_transform(input)
                array_data.append([max(input[0]), max(input[1]),max(input[2]), max(label)])
            else:
                input =np.concatenate(([face], [audio], [text]), axis=0)
                input = tsne.fit_transform(input)
                input = input.flatten()
                print(input)
                array_data.append([input[0], input[1], input[2], max(label)])





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
