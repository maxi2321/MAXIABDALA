import librosa
import os
import numpy as np
from scipy.signal import ShortTimeFFT, windows


def read_folder(folder, samplerate = 48000):
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
                audio, sr = librosa.load(path , sr=samplerate)  # leemos el audio como numpy array
                # duracion = librosa.get_duration(y=audio, sr=samplerate)
                data[c].append(audio)   # no guardamos el samplerate porque todos son iguales
    return data

def create_specs(x, ws, sr):
    '''Wrapper para calcular los espectrogramas de manera sencilla
    Regresa las frecuencias, los intervalos de tiempo y el espectrograma normalizado entre 0 y 1'''
    N = len(x)
    SFT = ShortTimeFFT(windows.boxcar(ws), hop=ws, fs=sr, scale_to='magnitude')
    ts = SFT.delta_t * np.arange(N)/ws
    Sx = SFT.spectrogram(x)  # perform the STFT
    Sx_l = np.log(Sx[:])
    Sx = (Sx_l - Sx_l.min())
    Sx /= Sx.max()
    return SFT.f, ts[::ws], Sx.T  # ax 0: time ax 1: freqs

def create_serie(x, width, ws, sr):
    '''
    returns a series of spectrograms of width= width
    this series can be used as time series for a lstm
    '''
    f, ts, spec = create_specs(x, ws, sr)
    serie = []
    for i in range(spec.shape[0]//width):
        serie.append(spec[i*width:i*width+width])
    return np.array(serie)


if __name__ == '__main__':
    import matplotlib.pyplot as plt
    data = read_folder('./audios/')

    # # dibujamos algun datito
    # rng = np.random.default_rng()
    # sr = 48000
    # keys = list(data.keys())
    # fig, axs = plt.subplots(3,1, figsize = (8,8))
    # for i in range(3):
    #     idx = rng.integers(len(data[keys[i]]))
    #     signal = data[keys[i]][idx]
    #     axs[i].set_title(keys[i])
    #     axs[i].plot(np.arange(len(signal))/sr, signal)
    # axs[i].set_xlabel(f'time in s')
    # fig.tight_layout()
    # plt.show()

    # creamos un espectrograma y dibujamos
    # freq, times, spec = create_specs(data['moto'][0], ws=400, sr=16000)
    # print(times.shape, freq.shape, spec.shape)
    # plt.imshow(spec, extent=[times[0], times[-1],freq[0], freq[-1]], aspect='auto', origin='lower')
    # plt.show()
    serie = create_serie(data['colectivo'][0], width=10, ws=400, sr=16000)
    print(serie.shape)
    fig, axs = plt.subplots(1,serie.shape[0])
    for i in range(serie.shape[0]):
        axs[i].imshow(serie[i].T, aspect = 'auto')
    plt.show()
    