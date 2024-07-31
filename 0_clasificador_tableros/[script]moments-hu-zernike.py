import cv2
import pandas as pd
import os
import mahotas

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

def momentos_csv(images):
    momentos = []
    for img in images:
        m = cv2.moments(img)
        momentos.append(m)

    df = pd.DataFrame(momentos)
    df.to_csv('0_clasificador_tableros/[descriptores]moments.csv', index=False)
    return momentos

def momentos_hu_csv(images):
    hu_list = []
    for img in images:
        m = cv2.moments(img)
        hu = cv2.HuMoments(m).flatten()
        hu_list.append(hu.tolist())

    df = pd.DataFrame(hu_list)
    df.to_csv('0_clasificador_tableros/[descriptores]HuMoments.csv', index=False)
    return hu_list

def momentos_zernike_csv(images, radius=10, degree=6):
    zernike_list = []
    for img in images:
        moments = mahotas.features.zernike_moments(img, radius=radius, degree=degree)
        zernike_list.append(moments)

    df = pd.DataFrame(zernike_list)
    df.to_csv('0_clasificador_tableros/[descriptores]zernikeMoments.csv', index=False)    
    return zernike_list

segmentos = segmentarImg("1_dataset_imagenes/dataset")
momentos_csv(segmentos)
momentos_hu_csv(segmentos)
momentos_zernike_csv(segmentos)