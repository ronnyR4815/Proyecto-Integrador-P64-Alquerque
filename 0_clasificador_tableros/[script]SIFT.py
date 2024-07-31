import cv2
import pandas as pd
import numpy as np
import os

def segmentarImg(input_folder):
    x_values = []

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

            print(f'Numero de objetos encontrados en {filename}: {num_labels-1}')

            # Procesa cada blob detectado
            for label in range(1, num_labels):
                x, y, w, h, _ = stats[label]
                img_segment = binary_image[y:y+h, x:x+w]

                x_values.append(img_segment)

    return x_values

def sift_csv(images):
    sift = cv2.SIFT_create()
    descriptors_list = []
    for img in images:
        kp, des = sift.detectAndCompute(img, None)
        if des is not None:
            descriptors_list.append(des.flatten().tolist())
        else:
            descriptors_list.append(np.zeros(128).tolist())

    df = pd.DataFrame({'descriptores': descriptors_list})
    df.to_csv('0_clasificador_tableros/[descriptores]SIFT.csv', index=False)
    return descriptors_list


segmentos = segmentarImg("1_dataset_imagenes/dataset")
sift_csv(segmentos)