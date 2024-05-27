from tqdm import tqdm
import torch
import os
from torch.utils.tensorboard.writer import SummaryWriter
import torch.nn as nn

class Trainer():
    def __init__(self, model: torch.nn.Module, num_epochs: int, lr: float, train_dataloader: torch.utils.data, val_dataloader: torch.utils.data, device: str,
                 save_root_path: str, logs_path: str, save_period: int, model_name: str):
        self.model = model
        self.loss = torch.nn.CrossEntropyLoss()
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr = lr, betas = (0.5, 0.999))
        self.num_epochs = num_epochs
        self.train_dataloader = train_dataloader
        self.val_dataloader = val_dataloader
        self.device = device
        self.save_period = save_period
        self.save_root_path = save_root_path
        self.model_name = model_name

        self.train_loss = []
        self.val_loss = []
        self.lr = []
        self.writer = SummaryWriter(log_dir=os.path.join(logs_path, self.model_name))
    
    def make_train_step(self): 
        def train_step(in_data, out_data):
            # Model in train mode
            self.model.train()
            # Makes predictions
            #in_data = in_data.reshape((30,1,8000))
            pred_data = self.model(in_data)
            #pred_data = pred_data.reshape((32,9))
            #out_data = out_data.reshape((32,1))
            # Computes loss
            loss = self.loss(pred_data, out_data)
            # Computes gradients
            loss.backward()
            # Updates parameters and zeroes gradients
            self.optimizer.step()
            self.optimizer.zero_grad()
            # Returns the loss
            return loss.item()
    
        # Returns the function that will be called inside the train loop
        return train_step 
    
    def train_loop(self):
        train_step = self.make_train_step()
        last_loss = 100
        trigger_times = 0
        patience = 5
        min_val_loss = 100
        for epoch in range(1,self.num_epochs+1):
            # Training
            for data in tqdm(self.train_dataloader):
                in_data, out_data = data[0].to(self.device), data[1].to(self.device)
                loss = train_step(in_data, out_data)
                self.train_loss.append(loss)
            
            # Validation
            with torch.no_grad():
                for data in tqdm(self.val_dataloader): 
                    in_val_data, out_val_data = data[0].to(self.device), data[1].to(self.device)
                    # Model in eval mode
                    self.model.eval()
                    # Make predictions
                    #in_val_data = in_val_data.reshape((30,1,8000))
                    pred_val_data = self.model(in_val_data)
                    # Compute loss
                    val_loss = self.loss(pred_val_data, out_val_data).item()
                    # Check early stopping
                    if val_loss < min_val_loss:
                        min_val_loss = val_loss
                        best_state = self.model.state_dict()
                    self.val_loss.append(val_loss)

            print(f'Epoch {epoch} of {self.num_epochs}')
            print("\n METRICS"+ 10 * ' ' + 'Training Set' + 10 * ' ' + 'Validation Set')
            print(f"{len('METRICS' + 11* ' ')* ' '}{loss:.2e}{14 * ' '}{val_loss:.2e}")
            self.writer.add_scalar('Loss/train', loss, epoch)
            self.writer.add_scalar('Loss/val', val_loss, epoch)
            if val_loss > last_loss:
                trigger_times += 1
            else:
            	trigger_times = 0
            if trigger_times >= patience:
                print('Early Stopping!')
                torch.save(best_state, self.save_root_path+f'/{self.model_name}/best_model.pt')
                break
            print(f'Trigger times: {trigger_times}')
            print(f'Last loss: {last_loss}')
            print(f'Current loss: {val_loss}')
            last_loss = val_loss
            if epoch % self.save_period == 0:
                torch.save(self.model.state_dict(), self.save_root_path+f'/{self.model_name}/{epoch}.pt')
       
