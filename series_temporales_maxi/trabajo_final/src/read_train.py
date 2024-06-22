import librosa
import os
import numpy as np

def read_folder(folder):
    # vemos los archivos de audio
    # carpeta del training set
    classes_folders = os.listdir(folder)                   # carpeta de cada clase
    # Creamos un diccionario para cada clase donde pondremos los audios
    data = {}
    for c in classes_folders:
        if c[0] != '.':
            data[c] = data.get(c, []) # agregamos
    # Llenamos una lista para cada clase con los datos
    for c in classes_folders:
        if c[0] != '.':
            for f in os.listdir(os.path.join(folder, c)):
                path = os.path.join(folder, c, f)
                audio, sr = librosa.load(path , sr=None)  # leemos el audio como numpy array
                duracion = librosa.get_duration(y=audio, sr=sr)
                data[c].append(audio)   # no guardamos el samplerate porque todos son iguales
    return data

if __name__ == '__main__':
    data = read_folder('./audios/')
