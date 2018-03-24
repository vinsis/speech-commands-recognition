import torch
import torch.nn as nn
from torch.autograd import Variable
from torch.utils.data import DataLoader
from torch.optim import lr_scheduler
import os

import loader
from loader import train_data, validate_data, test_data

BATCH_SIZE = 64
EPOCHS = 5
LR = 0.01
N = len(loader.classes)

WEIGHT_DIR = './trained_weights'

use_cuda = torch.cuda.is_available()
criterion = nn.CrossEntropyLoss()

train_loader = DataLoader(train_data, batch_size = BATCH_SIZE, shuffle = True)
validate_loader = DataLoader(validate_data, batch_size = BATCH_SIZE, shuffle = True)
test_loader = DataLoader(test_data, batch_size = BATCH_SIZE, shuffle = True)

class ModelCNN(nn.Module):
    def __init__(self):
        super(ModelCNN, self).__init__()
        self.name = 'fcnn'
        self.main = nn.Sequential(
            nn.Conv1d(1, 5, 90, stride = 6),
            nn.ReLU(True),
            nn.BatchNorm1d(5),
            nn.Conv1d(5, 15, 31, stride = 6),
            nn.ReLU(True),
            nn.BatchNorm1d(15),
            nn.Conv1d(15, 25, 11, stride = 3),
            nn.ReLU(True),
            nn.BatchNorm1d(25),
            nn.Conv1d(25, 50, 11, stride = 2),
            nn.ReLU(True),
            nn.BatchNorm1d(50),
            nn.Conv1d(50, 75, 7, stride = 3),
            nn.ReLU(True),
            nn.BatchNorm1d(75),
            nn.AvgPool1d(26)
        )
        self.fc = nn.Linear(75, N)

    def forward(self, x):
        x = self.main(x)
        x = x.view(x.size(0),-1)
        x = self.fc(x)
        return x

model_cnn = ModelCNN()
if use_cuda: model_cnn.cuda()
optimizer = torch.optim.Adam(model_cnn.parameters(), lr = LR)
scheduler = lr_scheduler.ExponentialLR(optimizer, gamma = 0.6)

def train(model, epoch):
    model.train()
    print('Learning rate is:', optimizer.param_groups[0]['lr'])
    for i, (data, target) in enumerate(train_loader):
        target = torch.squeeze(target)
        if use_cuda:
            data, target = data.cuda(), target.cuda()
        data, target = Variable(data), Variable(target)
        optimizer.zero_grad()
        output = model(data)
        loss = criterion(output, target)
        loss.backward()
        optimizer.step()
        if i % 100 == 0:
            print('Epoch [{}], iteration [{}], loss [{}]'.format(epoch, i+1, loss.data[0]))
    test(model, epoch, validate_loader)
    print('Saving trained model after epoch {}'.format(epoch))
    filename = '{}_{}.pkl'.format(model.name, epoch)
    torch.save(model.state_dict(), os.path.join(WEIGHT_DIR, filename))

def test(model, epoch, dataloader):
    model.eval()
    correct = 0
    for i, (data, target) in enumerate(dataloader):
        if use_cuda:
            data, target = data.cuda(), target.cuda()
        data, target = Variable(data, volatile = True), Variable(target, volatile = True)
        output = model(data)
        pred = output.data.max(1, keepdim = True)[1]
        correct += pred.eq(target.data.view_as(pred)).cpu().sum()
    print('Evaluation: Epoch [{}] Accuracy: [{} / {}]'.format(epoch, correct, len(dataloader.dataset)))

def start_training(epochs = EPOCHS, preload_weights = None):
    if preload_weights is not None:
        model_cnn.load_state_dict(torch.load(preload_weights))
    for i in range(epochs):
        scheduler.step()
        train(model_cnn, i)
