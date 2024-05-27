import numpy as np
import matplotlib.pyplot as plt
import librosa
import os

# uso listdir que me permite listar los archivos de un directorio
for path in os.listdir('./data/train'):
    y, sr = librosa.load('./data/train/' + path)
    Y = np.fft.fft(y)
    promedio = np.mean(np.abs(Y))
    print(path, promedio)

'''# Cargar el archivo de audio
y, sr = librosa.load(path_wave)
print(y[:5])
print(np.mean(y[:5]))
print(sr)

# Graficar la señal de audio
plt.plot(y)
plt.show()

# Calcular la transformada de Fourier
Y = np.fft.fft(y)
plt.plot(np.abs(Y))
plt.show()

promedio = np.mean(np.abs(Y))
print(promedio)'''