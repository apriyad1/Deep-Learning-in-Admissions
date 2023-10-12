import torch
import torch.nn as nn
import torch.nn.functional as F
import pandas as pd
import numpy as np
import sys

%matplotlib inline
from sklearn.metrics import accuracy_score
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import f1_score
from sklearn.metrics import matthews_corrcoef
from sklearn.metrics import roc_auc_score

from torch.utils.data import DataLoader, TensorDataset, random_split, WeightedRandomSampler
import torch.optim as optim

import os
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, classification_report


data = pd.read_csv('original_edit-2.csv')
inputs = data.drop(['final_read_score'], axis=1).to_numpy()
labels = data['final_read_score'].astype('int64').to_numpy()
inputs = torch.from_numpy(inputs).float()
labels = torch.from_numpy(labels).float()

dataset = TensorDataset(inputs, labels)
train_size = int(0.8*len(dataset))
test_size = len(dataset) - train_size
train_dataset, test_dataset = random_split(dataset, [train_size-1, test_size+1])
train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False)


class Net(nn.Module):
    def __init__(self, i, h_size=500, h_next_size=200, h_next_next_size=100, n_classes=2, how_many_layers=2):
        super(Net, self).__init__()
        self.fc1 = nn.Linear(i.shape[1], h_size)
        self.layers = how_many_layers

        if self.layers == 2:
            self.fc2 = nn.Linear(h_size, n_classes)

        if self.layers == 3:
            self.fc3 = nn.Linear(h_size, h_next_size)
            self.fc4 = nn.Linear(h_next_size, n_classes)

        if self.layers == 4:
            self.fc3 = nn.Linear(h_size, h_next_size)
            self.fc4 = nn.Linear(h_next_size, h_next_next_size)
            self.fc5 = nn.Linear(h_next_next_size, n_classes)

    def forward(self, X):
        act = F.relu
        act1 = F.sigmoid
        act2 = torch.tanh

        if self.layers == 2:
            X = self.fc2(act(self.fc1(X)))
            X = F.softmax(X, dim=1)

        if self.layers == 3:
            X = self.fc4(act2(self.fc3(act(self.fc1(X)))))
            X = F.softmax(X, dim=1)

        if self.layers == 4:
            X = self.fc5(act1(self.fc4(act2(self.fc3(act(self.fc1(X)))))))
            X = self.fc5(act2(self.fc4(act2(self.fc3(act(self.fc1(X)))))))

        return X


net = Net(inputs, h_size=256, h_next_size=32, how_many_layers=3)

n_epochs = 600
learning_rate = 0.001
decay_rate = learning_rate / n_epochs
optimizer = optim.Adam(net.parameters(), lr=learning_rate, weight_decay=decay_rate)

lambda_reg = 0.01
lambda_entropy = 0
def loss_fn(model, outputs, targets):
    cross_entropy = nn.functional.cross_entropy(outputs, targets)
    l2_regularization = 0
    entropy_regularization = 0
    for param in model.parameters():
        l2_regularization += torch.norm(param, p=2) ** 2
        entropy_regularization += torch.mean(torch.sum(-outputs * torch.log(outputs), dim=1))
    loss = cross_entropy + lambda_reg * l2_regularization
    return loss


def test_instance(model):
    y_t = []
    y_s = []
    loss = 0
    acc = 0
    with torch.no_grad():
        for inputs, labels in test_loader:
            outputs = model(inputs)
            loss += loss_fn(model, outputs, labels.long())
            y_t.extend(labels.numpy().astype('int'))
            y_s.extend(torch.sigmoid(outputs).max(axis=1).indices.numpy())

    acc = accuracy_score(y_t, y_s)
    return loss, acc

iteration = 0
counter = 0
for epoch in range(n_epochs):
    running_loss = 0.0
    total = 0  # No. of total predictions
    correct = 0  # No. of correct predictions

    for i, (inputs, labels) in enumerate(train_loader):
        optimizer.zero_grad()
        outputs = net(inputs)
        loss = loss_fn(net, outputs, labels.long())
        loss.backward()
        optimizer.step()
        running_loss += loss.item() * inputs.size(0)
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()
    epoch_loss = running_loss / len(train_loader.dataset)  # Loss in every epoch
    epoch_acc = correct / total  # Accuracy for every epoch

    if epoch % 10 == 0 or epoch == n_epochs - 1:
        print(f'Epoch: {epoch + 1}/1600 | pLoss: {running_loss / len(inputs)} | Accuracy: {epoch_acc} | Loss: {epoch_loss}')

    if epoch % 50 == 0:
        test_loss, test_acc = test_instance(net)
        print(f'Epoch: {epoch + 1} | The test data Accuracy = {test_acc} | Test Loss = {test_loss}')
        if (counter < test_acc):
            save_net = net
            counter = test_acc

y_true = []
y_scores = []
test_loss = 0
with torch.no_grad():
    for inputs, labels in test_loader:
        outputs = save_net(inputs)
        test_loss += loss_fn(net, outputs, labels.long())
        y_true.extend(labels.numpy().astype('int'))
        y_scores.extend(torch.sigmoid(outputs).max(axis = 1).indices.numpy())


accuracy = accuracy_score(y_true, y_scores)
precision = precision_score(y_true,y_scores)
recall = recall_score(y_true, y_scores)
f1_val = f1_score(y_true, y_scores)
auc_roc = roc_auc_score(y_true, y_scores)

print('Accuracy: {:.4f}'.format(accuracy))
print('Precision: {:.4f}'.format(precision))
print('Recall: {:.4f}'.format(recall))
print('F1 Score: {:.4f}'.format(f1_val))
print('AUROC Score: {:.4f}'.format(auc_roc))
