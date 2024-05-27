import sys

sys.path.insert(1, './configs')
sys.path.insert(1, './src')

import importlib
import argparse
import model
import dataloader
import torch
import os
from trainer import Trainer

argParser = argparse.ArgumentParser()
argParser.add_argument("config_name", help="Nombre del archivo de configuración que se quiere cargar")
args = argParser.parse_args()
config_name = list(vars(args).values())[0]
c = importlib.import_module(config_name)

## Model params
input_shape = c.network['input_shape']
if c.dataset['feature'] == 'mfcc':
    mod = model.NeuralNet_mfcc(input_shape)
else:
    mod = model.NeuralNet(input_shape)
mod.to(c.device)

## Dataset generation
train_dataloader, val_dataloader, test_dataloader = dataloader.gen_dataloaders(c.dataset['data_path'], c.dataset['feature'], c.training['batch_size'], 
                                                                    c.training['validation_split'])


## Train routine
if not os.path.exists(c.save_path + f'/{c.model_name}'):
    print('Creando directorio para los pesos del modelo: ', c.save_path + f'/{c.model_name}')
    os.mkdir(c.save_path + f'/{c.model_name}')
if not os.path.exists(c.logs_path + f'/{c.model_name}'):
    print('Creando directorio para los logs del entrenamiento: ', c.logs_path + f'/{c.model_name}')
    os.mkdir(c.logs_path + f'/{c.model_name}')
print('El modelo se va a entrenar en: ', c.device)
t = Trainer(mod, c.training['num_epochs'], c.training['lr'], train_dataloader, val_dataloader,
            c.device, c.save_path, c.logs_path, c.training['save_period'], c.model_name)

# Start training
t.train_loop()
