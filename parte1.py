import numpy as np
import soundfile as sf
import librosa as lb
import os
import seaborn as sns

train_dir = os.listdir('data/train')
test_dir = os.listdir('data/test')
señales = {'0': [], '1': [], '2': [], '3': [], '4': [], '5': [], '6': [], '7': [], '8': [], '9': []}
archivos = {'0': [], '1': [], '2': [], '3': [], '4': [], '5': [], '6': [], '7': [], '8': [], '9': []}

# Cargo los directorios
for path in train_dir:
    archivos[path.split('_')[0]].append('data/train/' + path)

# Cargo las señales
for n in range(10):
    for value in archivos[str(n)]:
        señales[str(n)].append(sf.read(value)[0])

# Función para normalizar la longitud de las señales
def normalize_length(señales):
    min_length = min(len(s) for s in señales)
    return [s[:min_length] for s in señales]

# Promedio de las señales
def raw():
    promedio_raw = {'0': [], '1': [], '2': [], '3': [], '4': [], '5': [], '6': [], '7': [], '8': [], '9': []}

    for n in range(10):
        señales_normalizadas = normalize_length(señales[str(n)])
        señales_stackeadas = np.stack(señales_normalizadas, axis=0)
        averaged_signal = np.mean(señales_stackeadas, axis=0)
        promedio_raw[str(n)] = averaged_signal
    return promedio_raw

# Transformada de Fourier
def dft():
    promedio_dft = {'0': [], '1': [], '2': [], '3': [], '4': [], '5': [], '6': [], '7': [], '8': [], '9': []}

    for n in range(10):
        señales_normalizadas = normalize_length(señales[str(n)])
        señales_stackeadas = np.stack(señales_normalizadas, axis=0)
        averaged_signal = np.mean(np.abs(np.fft.rfft(señales_stackeadas, axis=1)), axis=0)
        promedio_dft[str(n)] = averaged_signal
    return promedio_dft

# Mel Frequency Cepstral Coefficients
import matplotlib.pyplot as plt

def mfcc(coef=20):
    promedio_mfcc = {'0': [], '1': [], '2': [], '3': [], '4': [], '5': [], '6': [], '7': [], '8': [], '9': []}

    for n in range(10):
        señales_normalizadas = normalize_length(señales[str(n)])
        mfcc_features = [lb.feature.mfcc(y=s, n_mfcc=coef, n_fft=2048, hop_length=512) for s in señales_normalizadas]
        mfcc_stackeadas = np.stack(mfcc_features, axis=0)
        averaged_signal = np.mean(mfcc_stackeadas, axis=0)
        promedio_mfcc[str(n)] = averaged_signal
    return promedio_mfcc


#Comparación
def comparacion(feat, coef=20):
    c = 0
    conf_matrix = np.zeros((10, 10))
    for path in train_dir:
        a, _ = sf.read('data/train/' + path)
        if feat == 'dft':
            a = np.abs(np.fft.rfft(a))
        elif feat == 'mfcc':
            a = lb.feature.mfcc(y=a, sr=22050, n_mfcc=coef, n_fft=2048, hop_length=512)
        difs = []
        for n in range(10):
            difs.append(np.mean(np.abs((a - promedios[str(n)]) ** 2)))
        true_num = path.split('_')[0]
        pred_num = np.argmin(difs)
        conf_matrix[int(true_num)][pred_num] += 1
        if int(true_num) == pred_num:
            c += 1
    acc = 100 * c / len(train_dir)
    '''sns.heatmap(conf_matrix, annot=True, xticklabels=True, yticklabels=True, fmt='g', cmap='Blues')
    plt.xlabel('Predicted labels')
    plt.ylabel('True labels')
    plt.show()''' # matriz de confusión
    return acc

# Comparación (test)
def comparacion_t(feat, coef=20):
    c = 0
    conf_matrix = np.zeros((10, 10))
    for path in test_dir:
        a, _ = sf.read('data/test/' + path)
        if feat == 'dft':
            a = np.abs(np.fft.rfft(a))
        elif feat == 'mfcc':
            a = lb.feature.mfcc(y=a, sr=22050, n_mfcc=coef, n_fft=2048, hop_length=512)
        difs = []
        for n in range(10):
            difs.append(np.mean(np.abs((a - promedios[str(n)]) ** 2)))
        true_num = path.split('_')[0]
        pred_num = np.argmin(difs)
        conf_matrix[int(true_num)][pred_num] += 1
        if int(true_num) == pred_num:
            c += 1
    acc = 100 * c / len(test_dir)
    '''sns.heatmap(conf_matrix, annot=True, xticklabels=True, yticklabels=True, fmt='g', cmap='Blues')
    plt.xlabel('Predicted labels')
    plt.ylabel('True labels')
    plt.show()''' # matriz de confusión
    return acc

# Ejecución
promedios = raw()
print('Accuracy raw:', comparacion('raw'))
print('Accuracy raw (test):', comparacion_t('raw'))

promedios = dft()
print('Accuracy dft:', comparacion('dft'))
print('Accuracy dft (test):', comparacion_t('dft'))

promedios = mfcc()
print('Accuracy mfcc:', comparacion('mfcc'))
print('Accuracy mfcc (test):', comparacion_t('mfcc'))

#quiero ir variando la cantidad de coeficientes en el mfcc
'''coefs = list(range(1,41)) # coeficientes que voy a probar
presicion = []
t_presicion = []

for coef in coefs:
    promedios = mfcc(coef)
    presicion.append(comparacion('mfcc', coef))
    t_presicion.append(comparacion_t('mfcc', coef))
    print('done with', coef) #, useful as it takes a while to run

print(presicion)
print(t_presicion) # useful to save the results
plt.plot(coefs, presicion, label='Train')
plt.plot(coefs, t_presicion, label='Test')
plt.xlabel('Coeficientes')
plt.ylabel('Presición')
plt.legend()
plt.grid()
plt.show()'''