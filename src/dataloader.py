from torch.utils.data import DataLoader, random_split
from dataset import AudioMNISTDataset

def gen_dataloaders(data_path, feature, batch_size, validation_split, mfcc_params = None):
    dataset = AudioMNISTDataset(data_path = data_path, feature = feature)
    full_dataset_length = len(dataset)
    train_data_length = round(full_dataset_length * validation_split)
    val_data_length = round(full_dataset_length * (1-validation_split))
    train_dataset, val_dataset = random_split(dataset, [train_data_length, val_data_length])
    test_dataset = AudioMNISTDataset(data_path = data_path, feature = feature, test = True)
    train_dataloader = DataLoader(train_dataset, batch_size = batch_size, shuffle = True)
    val_dataloader = DataLoader(val_dataset, batch_size = batch_size, shuffle = True)
    test_dataloader = DataLoader(test_dataset, batch_size = 1, shuffle = False)
    return train_dataloader, val_dataloader, test_dataloader
