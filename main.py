import os
import numpy as np
import nibabel as nib
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv3D, MaxPooling3D, Flatten, Dense, Dropout
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.utils import to_categorical
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt

# Directorio que contiene los archivos NIfTI
directorio_base = 'folder_nifti'

# Listas para almacenar datos y etiquetas
datos = []
etiquetas = []
no_az = []

# Leer los archivos NIfTI en subdirectorios
for subdirectorio in os.listdir(directorio_base):
    subdirectorio_path = os.path.join(directorio_base, subdirectorio)
    if os.path.isdir(subdirectorio_path):
        # Obtener una lista de nombres de archivos en el subdirectorio
        archivos_2d = [archivo for archivo in os.listdir(subdirectorio_path) if archivo.endswith('.nii')]
        archivos_2d.sort()

        # Crear un volumen 3D apilando los slices 2D
        volumen_3d = []
        for archivo_2d in archivos_2d:
            archivo_2d_path = os.path.join(subdirectorio_path, archivo_2d)
            nifti_img = nib.load(archivo_2d_path)
            slice_2d = nifti_img.get_fdata()
            volumen_3d.append(slice_2d)
        
        # Apilar slices para formar el volumen 3D
        volumen_3d = np.array(volumen_3d)
        
        # Agregar el volumen 3D a la lista de datos
        datos.append(volumen_3d)
        
        # Agregar la etiqueta correspondiente (1 si tiene Alzheimer, 0 si no)
        ban_az = int(subdirectorio.startswith('alzheimer'))
        etiquetas.append(ban_az)
        if ban_az == 0:
            no_az.append(volumen_3d)
        

# Convertir listas en matrices numpy
datos = np.array(datos)
etiquetas = np.array(etiquetas)

# Dividir los datos en conjuntos de entrenamiento y prueba
X_train, X_test, y_train, y_test = train_test_split(datos, etiquetas, test_size=0.3333, random_state=42)

# Convertir etiquetas a formato one-hot encoding (binario)

y_train = to_categorical(y_train, num_classes=2)
y_test = to_categorical(y_test, num_classes=2)

# Definir la arquitectura de la CNN 3D
model = Sequential()

# Convolutional layer 1
model.add(Conv3D(32, kernel_size=(3, 3, 3), activation='relu', input_shape=X_train.shape[1:]))
model.add(MaxPooling3D(pool_size=(2, 2, 2)))

# Convolutional layer 2
model.add(Conv3D(64, kernel_size=(3, 3, 3), activation='relu'))
model.add(MaxPooling3D(pool_size=(2, 2, 2)))

# Convolutional layer 3
model.add(Conv3D(128, kernel_size=(3, 3, 3), activation='relu'))
model.add(MaxPooling3D(pool_size=(2, 2, 2)))

# Flatten the 3D output to feed into fully connected layers
model.add(Flatten())

# Fully connected layers
model.add(Dense(128, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(2, activation='sigmoid'))  # 2 clases: Alzheimer o no Alzheimer // sigmoid -> Binario // softmax -> convierte números en probabilidades que suman uno

model.summary()

# Compilar el modelo
model.compile(loss='categorical_crossentropy', optimizer=Adam(lr=0.0001), metrics=['accuracy'])

# Entrenar el modelo
history = model.fit(X_train, y_train, batch_size=32, epochs=10, verbose=2, validation_data=(X_test, y_test))

# Evaluar el modelo en los datos de prueba
test_loss, test_accuracy = model.evaluate(X_test, y_test)
print(f'Precisión en datos de prueba: {test_accuracy}')
print(f'Precisión en datos de prueba con porcentaje: {test_accuracy * 100:.2f}%')


# Graficar la precisión durante el entrenamiento y la validación
plt.plot(history.history['accuracy'])
plt.plot(history.history['val_accuracy'])
plt.title('Precisión del modelo')
plt.xlabel('Época')
plt.ylabel('Precisión')
plt.legend(['Entrenamiento', 'Validación'], loc='upper left')
plt.show()


# Hacer la predicción
nueva_imagen_preprocesada = no_az  # Asegúrate de definir 'preprocesar_imagen'

# Expandir las dimensiones para que coincidan con la entrada del modelo
nueva_imagen_preprocesada = np.expand_dims(nueva_imagen_preprocesada, axis=0)

# Realizar la predicción
prediccion = model.predict(X_test)

# Interpretar la predicción
umbral = 0.5  # Ajusta este umbral según tus necesidades
confianza_alzheimer = prediccion[0][1]  # Probabilidad de que sea Alzheimer
confianza_no_alzheimer = prediccion[0][0]  # Probabilidad de que no sea Alzheimer

if confianza_alzheimer > umbral:
    resultado = "Alzheimer"
    porcentaje_confianza = confianza_alzheimer * 100
else:
    resultado = "No Alzheimer"
    porcentaje_confianza = confianza_no_alzheimer * 100

print(f"Resultado: {resultado}")
print(f"Porcentaje de Confianza: {porcentaje_confianza:.2f}%")

