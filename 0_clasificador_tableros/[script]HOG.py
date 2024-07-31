import cv2
import pandas as pd
import os
from skimage.transform import resize
from skimage.feature import hog

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

                img_segment = resize(img_segment, (64, 32))
                x_values.append(img_segment)

    return x_values

def hog_csv(images):
    hog_list = []
    for img in images:
        # Calcular descriptores HOG
        fd, hog_image = hog(img, orientations=9, pixels_per_cell=(8, 8), cells_per_block=(2, 2), visualize=True)
        hog_list.append(fd.tolist())

    df = pd.DataFrame(hog_list)
    df.to_csv('0_clasificador_tableros/[descriptores]HOG.csv', index=False)
    
    return hog_list


segmentos = segmentarImg("1_dataset_imagenes/dataset")
hog_csv(segmentos)