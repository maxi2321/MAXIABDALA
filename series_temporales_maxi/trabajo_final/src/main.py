from read_train import read_folder, create_serie
from cnn_lstm import CNN_LSTM
import torch
import numpy as np
import matplotlib.pyplot as plt
import os
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

SR = 44100
IMG_WIDTH = 10
SPECT_WS = 300
# we read the data
data_folder = os.path.join('.', 'audios')
data_raw = read_folder(data_folder, samplerate=SR)

# We create the classes into an encoder
labels = list(data_raw.keys())
enc = LabelEncoder()
enc.fit(labels)

# we create x for train
y = []
x = []
for k, values in data_raw.items():
    for v in values:
        y.append(k)
        spects = np.array(create_serie(v, width=IMG_WIDTH, ws=SPECT_WS, sr = SR))
        x.append(spects)

y = enc.transform(np.array(y)).reshape(-1,1)

x_tr, x_val, y_tr, y_val =  train_test_split(x,y, test_size=0.25)
print(f'training set length: {len(x_tr)} - validation set length: {len(x_val)}')
# we create the model and optimizer
n_freqs = SPECT_WS//2+1
model = CNN_LSTM(w=IMG_WIDTH, h=n_freqs, categories=5)
optim = torch.optim.Adam(model.parameters(), lr = 1e-4)
criterio = torch.nn.CrossEntropyLoss()

# we train the model
epochs = 300
history = []
history_test = []
history_acc = []
best_test_acc = 0
filename = './best_model_params.pth'
for e in range(epochs):
    aux_loss = 0
    for i in range(len(x_tr)):
        x = torch.FloatTensor(x_tr[i]).unsqueeze(1)
        print(x.shape)
        y_pred = model(x).view(1,-1)
        loss = criterio(y_pred, torch.LongTensor(y_tr[i]))
        optim.zero_grad()  # pongo en 0 los gradiente
        loss.backward()     # calculo las derivadas respecto de los parametros
        optim.step()        #hago un descenso en la direccion opuesta al gradiente
        aux_loss += loss.item()
    aux_loss/=len(x_tr)
    history.append(aux_loss)

    with torch.no_grad():
        model.eval()
        val_loss = 0
        preds = []
        for i in range(len(x_val)):
            x = torch.FloatTensor(x_val[i]).unsqueeze(1)
            y_pred = model(x).view(1,-1)
            val_loss += criterio(y_pred, torch.LongTensor(y_val[i])).item()
            preds.append([y_pred.numpy().argmax()])
        val_loss/=len(x_val)
        history_acc.append(accuracy_score(y_val, preds))
        history_test.append(val_loss)
        if history_acc[-1] > best_test_acc:
            torch.save(model.state_dict(), filename)
            best_test_acc = history_acc[-1]
        if e%10 == 0:
            print(f'epoch: {e} - validation accuracy score: {history_acc[-1] }')
            print(f'epoch: {e} - training loss: {aux_loss} - test loss: {val_loss}')


fig, axs = plt.subplots(1,2)

axs[0].semilogy(history, label = 'training loss')
axs[0].semilogy(history_test, label = 'validation loss')
axs[0].legend()
axs[1].plot(history_acc, label = 'validation accuracy score')
axs[1].legend()
plt.show()

