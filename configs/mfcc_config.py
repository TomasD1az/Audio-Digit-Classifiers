import torch

dataset = {
        'name' : 'AudioMNIST',
        'data_path' : './data',
        'feature' : 'mfcc',
        'mfcc_params' : None,
        } 

training = {
        'num_epochs' : 50,
        'lr' : 0.00001,
        'save_period': 25,
        'batch_size': 10,
        'validation_split': 0.7,
        }

network = {
        'input_shape' : 20,
        }

model_name = 'AudioMNIST_mfcc'
save_path = './models'
logs_path = './logs'
predictions_path = './predictions'

# Number of GPUs available. Use 0 for CPU mode.
ngpu = 1
device = torch.device("cuda:0" if (torch.cuda.is_available() and ngpu>0) else "cpu")
