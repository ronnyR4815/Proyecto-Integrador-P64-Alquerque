import cv2
import pandas as pd
import numpy as np
import os
from sklearn.cluster import KMeans
from sklearn.svm import SVC
from joblib import dump

def segmentarImg(input_folder, output_folder):
    x_values = []

    # Limpiar la carpeta de salida
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)
    else:
        for file in os.listdir(output_folder):
            os.remove(os.path.join(output_folder, file))

    # Leer imágenes de la carpeta de entrada
    for filename in os.listdir(input_folder):
        if filename.endswith(".jpg") or filename.endswith(".png"):
            file_path = os.path.join(input_folder, filename)
            img = cv2.imread(file_path)
            img_gray = cv2.imread(file_path, 0)

            # Aplicar Otsu para binarización
            _, binary_image = cv2.threshold(img_gray, 0, 255, cv2.THRESH_OTSU)

            # Etiquetado de blobs
            num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(binary_image, connectivity=8)
            color = (255, 255, 0)

            print(f'Numero de objetos encontrados en {filename}: {num_labels-1}')

            # Procesa cada blob detectado
            for label in range(1, num_labels):
                x, y, w, h, _ = stats[label]
                img_segment = binary_image[y:y+h, x:x+w]

                # Guardar cada imagen segmentada
                x_values.append(img_segment)
                segment_file_path = os.path.join(output_folder, f'segment_{label}_{filename}')
                # cv2.imwrite(segment_file_path, img_segment)

                # Dibujar en la imagen original para verificación
                cv2.rectangle(img, (x, y), (x+w, y+h), color, 2)
                centroid_x, centroid_y = centroids[label]
                cv2.circle(img, (int(centroid_x), int(centroid_y)), 2, color, -1)

            # Opcional: Guardar la imagen original con marcas para revisión
            cv2.imwrite(os.path.join(output_folder, f'marked_{filename}'), img)
    return x_values

def extract_sift_features(images):
    sift = cv2.SIFT_create()
    descriptors_list = []
    for img in images:
        kp, des = sift.detectAndCompute(img, None)
        descriptors_list.append(des if des is not None else [])
    return descriptors_list

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

def train_classifier(X_train, y_train):
    classifier = SVC(kernel='rbf', probability=True)
    classifier.fit(X_train, y_train)
    dump(classifier, '0_clasificador_tableros/modeloSVM.joblib')
    return classifier





x_train = segmentarImg("1_dataset_imagenes/train", "2_segmentos_imagenes/segmentedTrain")
# x_test = segmentarImg("1_dataset_imagenes/dataset", "2_segmentos_imagenes/segmentedTest")

y_train = np.array([1,1,0,2,2,2,2,0,1,1,1,1,0,2,2,1,1,0,1,1,2,2,0,2,2,
                    0,1,0,2,1,0,2,0,1,2,0,1,0,2,1,2,2,0,1,0,2,1,0,2,0,
                    1,2,1,2,1,0,0,0,0,0,2,1,2,1,2,0,0,0,0,0,1,2,1,2,1,
                    1,2,1,0,1,0,1,2,0,0,0,0,0,1,1,2,1,2,0,2,2,0,1,2,2])

# Extraer características SIFT de las imágenes
descriptors_train = extract_sift_features(x_train)
# descriptors_test = extract_sift_features(x_test)

# Clustering para crear un Bag of Words
k = 3  # Número de clústeres
all_descriptors = np.vstack([des for des in descriptors_train if len(des) > 0])
#all_descriptors = np.vstack([des for des in descriptors_train if len(des) > 0] + 
#                            [des for des in descriptors_test if len(des) > 0])
kmeans = KMeans(n_clusters=k, random_state=0)
kmeans.fit(all_descriptors)
dump(kmeans, '0_clasificador_tableros/kmeans.joblib')

# Construir histogramas de características para cada imagen
X_train = build_feature_histogram(descriptors_train, kmeans)
# X_test = build_feature_histogram(descriptors_test, kmeans)

classifier = train_classifier(X_train, y_train)