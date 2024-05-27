import numpy as np

class Evaluator():
    def __init__(self, model, train_iterator, val_iterator, test_iterator, device, save_path, model_name):
        self.model = model
        self.device = device
        self.train_iterator = train_iterator
        self.val_iterator = val_iterator
        self.test_iterator = test_iterator
        self.save_path = save_path
        self.model_name = model_name

    def make_predictions(self):
        pred = []
        inp = []
        out = []
        with open(self.save_path+f'/{self.model_name}/test_predictions.npy', 'wb') as f:
            for i in range(len(self.test_iterator)):
                batch = next(self.test_iterator)
                input = batch[0].to(self.device)
                output = batch[1].to(self.device)
                prediction = self.model(input).cpu().detach().numpy()
                print(prediction,output)
                pred.append(self.model(input).cpu().detach().numpy())
                inp.append(input.cpu().detach().numpy())
                out.append(output.cpu().detach().numpy())
            np.savez(f, i=inp, o=out, p=pred)
 
