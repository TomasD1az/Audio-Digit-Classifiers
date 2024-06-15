import matplotlib.pyplot as plt
import numpy as np
import soundfile as sf
import librosa as lb
import os

train_dir = os.listdir('data/train')
señales = {'0': [], '1': [], '2': [], '3': [], '4': [], '5': [], '6': [], '7': [], '8': [], '9': []}
archivos = {'0': [], '1': [], '2': [], '3': [], '4': [], '5': [], '6': [], '7': [], '8': [], '9': []}

#Cargo los directorios
for path in train_dir:
    archivos[path.split('_')[0]].append('data/train/' + path)

#Cargo las señales
for n in range(10):
    for value in archivos[str(n)]:
        señales[str(n)].append(sf.read(value)[0])

#Promedio de las señales
def raw():
    promedio_raw = {'0': [], '1': [], '2': [], '3': [], '4': [], '5': [], '6': [], '7': [], '8': [], '9': []}

    for n in range(10):
        señales_stackeadas = np.stack(señales[str(n)], axis=0)
        averaged_signal = np.mean(señales_stackeadas, axis=0)
        plt.plot(averaged_signal)
        '''plt.title('Promedio de las señales de ' + str(n))
        plt.show()'''
        promedio_raw[str(n)] = averaged_signal
    return promedio_raw

#Transformada de Fourier
def dft():
    promedio_dft = {'0': [], '1': [], '2': [], '3': [], '4': [], '5': [], '6': [], '7': [], '8': [], '9': []}

    for n in range(10):
        señales_stackeadas = np.stack(señales[str(n)], axis=0)
        averaged_signal = np.mean(np.abs(np.fft.rfft(señales_stackeadas)), axis=0)
        plt.plot(averaged_signal)
        '''plt.title('Promedio de las señales de ' + str(n))
        plt.show()'''
        promedio_dft[str(n)] = averaged_signal
    return promedio_dft

#Mel Frequency Cepstral Coefficients
def mfcc():
    promedio_mfcc = {'0': [], '1': [], '2': [], '3': [], '4': [], '5': [], '6': [], '7': [], '8': [], '9': []}

    for n in range(10):
        señales_stackeadas = np.stack(señales[str(n)], axis=0)
        averaged_signal = np.mean(lb.feature.mfcc(y=señales_stackeadas), axis=0)
        plt.plot(averaged_signal)
        '''plt.title('Promedio de las señales de ' + str(n))
        plt.show()'''
        promedio_mfcc[str(n)] = averaged_signal
    return promedio_mfcc

#Comparación
def comparacion(feat):
    c = 0
    for path in train_dir:
        a, _ = sf.read('data/train/' + path)
        if feat == 'dft':
            a = np.abs(np.fft.rfft(a))
        elif feat == 'mfcc':
            a = lb.feature.mfcc(y=a)
        difs = []
        for n in range(10):
            difs.append(np.mean(np.abs((a - promedios[str(n)]) ** 2)))
        true_num = path.split('_')[0]
        pred_num = np.argmin(difs)
        if int(true_num) == pred_num:
            c += 1
    acc = 100 * c / len(train_dir)
    return acc

#Ejecución
promedios = raw()
print('Accuracy raw:', comparacion('raw'))

promedios = dft()
print('Accuracy dft:', comparacion('dft'))

promedios = mfcc()
print('Accuracy mfcc:', comparacion('mfcc'))