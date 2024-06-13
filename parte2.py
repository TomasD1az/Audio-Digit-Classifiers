import numpy as np

predictions_path = 'predictions/AudioMNIST_mfcc/test_predictions.npy'

data = np.load(predictions_path, allow_pickle=True)

print(f'Claves disponibles en el archivo de predicciones: {data.keys()}')

inputs = data['i']
outputs = data['o']
predictions = data['p']

inputs = np.concatenate(inputs, axis=0)
outputs = np.concatenate(outputs, axis=0)
predictions = np.concatenate(predictions, axis=0)

print(f'Forma de inputs: {inputs.shape}')
print(f'Forma de outputs: {outputs.shape}')
print(f'Forma de predictions: {predictions.shape}')

if outputs.ndim == 1:
    predicted_classes = np.argmax(predictions, axis=1)
    true_classes = outputs
    correct_predictions = np.sum(predicted_classes == outputs)
else:
    predicted_classes = np.argmax(predictions, axis=1)
    true_classes = np.argmax(outputs, axis=1)
    correct_predictions = np.sum(predicted_classes == true_classes)

total_predictions = len(outputs)
accuracy = (correct_predictions / total_predictions) * 100

print(f'Accuracy: {accuracy:.2f}%')

conf_matrix = confusion_matrix(true_classes, predicted_classes)

disp = ConfusionMatrixDisplay(confusion_matrix=conf_matrix)
disp.plot(cmap=plt.cm.Blues)
plt.title('Matriz de Confusión')
plt.show()