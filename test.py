
import sys

sys.path.insert(1, './configs')
sys.path.insert(1, './src')

import importlib
import argparse
import model
import dataloader
import torch
import os
import glob
from eval import Evaluator

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
    mod = model.NeuralNet2(input_shape)                                                
mod.to(c.device)     

weigths_list = glob.glob(os.path.join(c.save_path+f'/{c.model_name}', "*"))
mod.load_state_dict(torch.load(weigths_list[-1]))

## Dataset generation
train_dataloader, val_dataloader, test_dataloader = dataloader.gen_dataloaders(c.dataset['data_path'], c.dataset['feature'], c.training['batch_size'], 
                                                                    c.training['validation_split'], c.dataset['mfcc_params'])

# Iterators definition
train_iterator = iter(train_dataloader)
val_iterator = iter(val_dataloader)
test_iterator = iter(test_dataloader)

# Evaluation routine
if not os.path.exists(c.predictions_path + f'/{c.model_name}'):
    print('Creando directorio para las predicciones del modelo: ', c.predictions_path + f'/{c.model_name}')
    os.mkdir(c.predictions_path + f'/{c.model_name}')
e = Evaluator(mod, train_iterator, val_iterator, test_iterator, c.device, c.predictions_path, c.model_name)

# Make predictions
e.make_predictions()
