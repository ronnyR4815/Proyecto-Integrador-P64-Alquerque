# Proyecto de Reconocimiento de Piezas de Tablero de Alquerque

Este proyecto se enfoca en la clasificación y reconocimiento de las piezas de un tablero de Alquerque utilizando diferentes técnicas de extracción de características y modelos de aprendizaje automático.

## Descripción del Proyecto

El objetivo principal del proyecto fue desarrollar y comparar distintos modelos de clasificación para identificar correctamente las piezas en las casillas de un tablero de Alquerque. El proceso se llevó a cabo en varias etapas, desde la segmentación del tablero hasta la evaluación de modelos de aprendizaje automático.

## Pasos Realizados

### 1. Segmentación de Tableros
- Se segmentaron los tableros de Alquerque hasta obtener 25 imágenes separadas de cada pieza del tablero.

### 2. Extracción de Descriptores
- Se utilizaron descriptores SIFT (Scale-Invariant Feature Transform) para cada pieza segmentada.

### 3. Agrupamiento con K-means
- Se agruparon los descriptores en 3 clusters utilizando el algoritmo de K-means.

### 4. Creación de Histogramas
- Se crearon histogramas con las características agrupadas por K-means para cada pieza del tablero.

### 5. Entrenamiento del Modelo SVM
- Se entrenó un modelo de Máquina de Soporte Vectorial (SVM) utilizando los histogramas de 5 tableros distintos y sus etiquetas reales.

### 6. Evaluación del Modelo SVM
- Se evaluó la precisión del modelo SVM con 100 imágenes de tableros diferentes. El modelo logró un 100% de precisión en las imágenes evaluadas.

## Comparación de Diferentes Extractores de Características

Se construyeron varios datasets basados en diferentes tipos de extractores de características:
- SIFT
- HOG (Histogram of Oriented Gradients)
- Momentos
- Momentos Hu
- Momentos Zernike

## Entrenamiento de un Modelo CNN

- Se entrenó un modelo basado en CNN (Convolutional Neural Network) con las mismas imágenes utilizadas en el modelo SVM.
- Se encontró que el clasificador de CNN era menos preciso comparado con el modelo SVM, que tenía un 100% de precisión.

## Construcción del Dataset de Movimientos

- Se creó un dataset de (100 x 25) con 100 tableros evaluados, donde cada fila representaba las 25 casillas de un tablero.
- Se extrajeron aproximadamente 130 movimientos posibles por tablero, resultando en un dataset final de 9070 movimientos.

## Modelos de Aprendizaje Automático

Se entrenaron tres modelos diferentes utilizando el dataset de movimientos:
- Neural Network: 0.98239
- Perceptron Multicapa: 0.98026
- Random Forest: 0.89968

## Proceso de Particionamiento del Dataset

- Se evaluaron las primeras 100 semillas para la partición del dataset.
- Se utilizó la mediana de las accuracies para seleccionar la semilla final de particionamiento (Semilla = 77).
- El dataset fue dividido en conjuntos de entrenamiento y prueba basados en esta semilla y almacenado en documentos separados para replicar pruebas.

## Resultados

- Los modelos entrenados obtuvieron altas precisiones, siendo el modelo de Random Forest el de menor precisión comparado con los otros dos modelos.

## Conclusiones

El proyecto demostró que el modelo SVM con descriptores SIFT fue extremadamente efectivo para la clasificación de piezas en tableros de Alquerque. Comparativamente, el modelo CNN no alcanzó la misma precisión. Los modelos de aprendizaje automático entrenados con el dataset de movimientos también mostraron altas precisiones, con la red neuronal simple obteniendo el mejor rendimiento.

## Requisitos

- Python 3.x
- OpenCV
- Scikit-learn
- Keras/TensorFlow (para modelos de redes neuronales)
- Pandas
- NumPy

## Instrucciones para Ejecutar el Proyecto

1. Clonar el repositorio.
2. Instalar las dependencias.
3. Ejecutar los scripts de segmentación y extracción de características.
4. Entrenar los modelos de clasificación.
5. Evaluar los modelos con el dataset de test.
