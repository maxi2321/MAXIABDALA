import torch
import numpy as np

class CNN_LSTM(torch.nn.Module):
    def __init__(self, w, h, categories):
        super().__init__()
        self.w = w
        self.h = h
        self.w_half = self.w//4
        self.h_half = self.h//4
        self.categories = categories
        self.dr = 0.4
        self.cnn = torch.nn.Sequential(
            torch.nn.Conv2d(in_channels=1, out_channels=8, kernel_size=(3,3), stride=(1,1), padding=(1,1)),
            torch.nn.MaxPool2d(kernel_size=(2,2)),
            torch.nn.ReLU(),
            torch.nn.Dropout2d(self.dr),
            torch.nn.Conv2d(in_channels=8, out_channels=16, kernel_size=(3,3), stride=(1,1), padding=(1,1)),
            torch.nn.MaxPool2d(kernel_size=(2,2)),
            torch.nn.Dropout2d(self.dr),
            torch.nn.ReLU(),
            torch.nn.Flatten(),
            torch.nn.Linear(in_features = 16 * self.w_half * self.h_half, out_features = 64)
        )
        self.lstm = torch.nn.LSTM(input_size=64, hidden_size=64, num_layers=3, batch_first=True, dropout=self.dr)
        self.linear = torch.nn.Linear(in_features=64, out_features=self.categories)
        self.relu = torch.nn.ReLU()

    def forward(self,x):
        x = self.cnn(x)
        x, _ = self.lstm(x)
        x = self.linear(x)
        return self.relu(x[-1])
    

if __name__ == '__main__':
    model = CNN_LSTM(w = 10, h = 21, categories= 5)
    x = torch.rand(size=(10,1, 10, 21))
    y = model(x)
    print(y.shape)
    print(y)