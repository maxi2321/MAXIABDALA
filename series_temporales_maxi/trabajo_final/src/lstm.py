import torch

class LSTM(torch.nn.Module):
    def __init__(self, ws, categories):
        super().__init__()
        self.ws = ws
        self.categories = categories
        self.lstm = torch.nn.LSTM(input_size  = self.ws//2+1,
                                  hidden_size = 128,
                                  num_layers  = 3,
                                  batch_first = True,
                                  bidirectional= False)
        self.fcon = torch.nn.Linear( in_features  = 128,
                                     out_features = self.categories)

    def forward(self, x):
        x, _ = self.lstm(x)
        x = self.fcon(x)
        # x = torch.sigmoid(x)
        return x[:,-1,:].squeeze(1) # regresamos el ultimo valor (many to one)