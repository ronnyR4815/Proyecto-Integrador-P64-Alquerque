import cv2
import numpy as np
import pandas as pd
import os
from joblib import load

def show_labeled_images(input_folder, output_folder, kmeans, classifier):
    # Limpiar la carpeta de salida
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)
    else:
        for file in os.listdir(output_folder):
            os.remove(os.path.join(output_folder, file))
    
    # Crear las columnas para el dataset
    # column_names = [f"({x},{y})" for x in range(0, 5) for y in range(0, 5)]
    # tablero_df = pd.DataFrame(columns=column_names)

    # Procesar cada archivo en la carpeta de entrada
    for filename in os.listdir(input_folder):
        if filename.endswith(".jpg") or filename.endswith(".png"):
            file_path = os.path.join(input_folder, filename)
            img = cv2.imread(file_path)
            img_gray = cv2.imread(file_path, 0)

            # Aplicar Otsu para binarización
            _, binary_image = cv2.threshold(img_gray, 0, 255, cv2.THRESH_OTSU)

            # Etiquetado de componentes conectados
            num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(binary_image, connectivity=8)

            # SIFT para extracción de características
            sift = cv2.SIFT_create()

            predictions = []

            for label in range(1, num_labels):
                x, y, w, h, _ = stats[label]
                segment = binary_image[y:y+h, x:x+w]

                # Extraer descriptores SIFT
                _, des = sift.detectAndCompute(segment, None)
                if des is not None:
                    # Crear histograma de características
                    histogram = build_feature_histogram([des], kmeans)[0]
                    histogram = histogram.reshape(1, -1)
                    prediction = classifier.predict(histogram)[0]
                else:
                    prediction = 0

                predictions.append(prediction)

                # Dibujar rectángulo y etiqueta en la imagen
                cv2.rectangle(img, (x, y), (x+w, y+h), (0, 255, 0), 2)
                cv2.putText(img, str(prediction), (x + 20, y + 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

            # row = pd.DataFrame([predictions], columns=column_names)
            # tablero_df = pd.concat([tablero_df, row], ignore_index=True)

            # Guardar la imagen procesada en la carpeta de salida
            output_image_path = os.path.join(output_folder, filename)
            cv2.imwrite(output_image_path, img)
            print(f"Imagen {filename} guardada...")

    # tablero_df.to_csv('tableros.csv', index=False)
    # print("Dataset tableros guardado...")

def build_feature_histogram(descriptors_list, kmeans):
    histogram_length = kmeans.n_clusters
    histograms = []
    
    # Construir un histograma por cada conjunto de descriptores
    for des in descriptors_list:
        if len(des) > 0:
            features = kmeans.predict(des)
            histogram, _ = np.histogram(features, bins=np.arange(histogram_length + 1))
            histograms.append(histogram)
        else:
            histograms.append(np.zeros(histogram_length))
    
    # Convertir la lista de histogramas a un array
    return np.array(histograms)

classifier = load('0_clasificador_tableros/modeloSVM.joblib')
kmeans = load('0_clasificador_tableros/kmeans.joblib')
# Deteccion de fichas en las imagenes
# show_labeled_images("1_dataset_imagenes/dataset", "3_labelling_imagenes/labelling", kmeans, classifier)
# show_labeled_images("1_dataset_imagenes/dataset_andrew", "3_labelling_imagenes/labelling_andrew", kmeans, classifier)
# show_labeled_images("1_dataset_imagenes/dataset_sarita_1", "3_labelling_imagenes/labelling_sarita_1", kmeans, classifier)
# show_labeled_images("1_dataset_imagenes/dataset_sarita_2", "3_labelling_imagenes/labelling_sarita_2", kmeans, classifier)
# show_labeled_images("1_dataset_imagenes/dataset_osillo", "3_labelling_imagenes/labelling_osillo", kmeans, classifier)
# show_labeled_images("1_dataset_imagenes/dataset_mena", "3_labelling_imagenes/labelling_mena", kmeans, classifier)
# show_labeled_images("1_dataset_imagenes/dataset_pila_1", "3_labelling_imagenes/labelling_pila_1", kmeans, classifier)
# show_labeled_images("1_dataset_imagenes/dataset_pila_2", "3_labelling_imagenes/labelling_pila_2", kmeans, classifier)