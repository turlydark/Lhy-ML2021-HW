# create testing dataset
import config
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from classifier import Classifier

class TIMITDataset(Dataset):
    def __init__(self, X, y=None):
        self.data = torch.from_numpy(X).float()
        if y is not None:
            y = y.astype(np.int)
            self.label = torch.LongTensor(y)
        else:
            self.label = None

    def __getitem__(self, idx):
        if self.label is not None:
            return self.data[idx], self.label[idx]
        else:
            return self.data[idx]

    def __len__(self):
        return len(self.data)
# check device
def get_device():
    return 'cuda' if torch.cuda.is_available() else 'cpu'

if __name__ == "__main__":
    test = np.load(config.root_path + 'test_11.npy')

    test_set = TIMITDataset(test, None)
    test_loader = DataLoader(test_set, batch_size=config.batch_size, shuffle=False)

    # create model and load weights from checkpoint
    device = get_device()
    model = Classifier().to(device)
    model.load_state_dict(torch.load(config.model_path))

    # Make prediction.
    predict = []
    model.eval()  # set the model to evaluation mode
    with torch.no_grad():
        for i, data in enumerate(test_loader):
            inputs = data
            inputs = inputs.to(device)
            outputs = model(inputs)
            _, test_pred = torch.max(outputs, 1)  # get the index of the class with the highest probability

            for y in test_pred.cpu().numpy():
                predict.append(y)

    with open('prediction.csv', 'w') as f:
        f.write('Id,Class\n')
        for i, y in enumerate(predict):
            f.write('{},{}\n'.format(i, y))