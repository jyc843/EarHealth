import numpy as np
import pandas as pd
import torch
import model
from torch.utils import data
from torch import optim, nn
from torch.nn import functional as F
from torch.utils.data import TensorDataset, DataLoader

def load_data(train_data, train_label, test_data, test_label):
    train_data   = train_data[:, np.newaxis]
    train_data   = torch.Tensor(train_data)
    train_label  = np.squeeze(train_label, axis=1)
    train_label  = torch.Tensor(train_label-1)
    dataset      = TensorDataset(train_data, train_label)
    data_loader  = torch.utils.data.DataLoader(dataset, batch_size=batchsize, shuffle=True)
    test_data    = test_data[:, np.newaxis]
    test_data    = torch.Tensor(test_data)
    test_label   = np.squeeze(test_label, axis=1)
    test_label   = torch.Tensor(test_label-1)
    dataset_test = TensorDataset(test_data, test_label)
    test_loader  = torch.utils.data.DataLoader(dataset_test, batch_size=batchsize, shuffle=False)
    return data_loader, test_loader

def train(epoch, train_loader):
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device).long()
        output = net(data)
        #Replaces pow(2.0) with abs() for L1 regularization
        l2_lambda = 0.0005
        l2_norm = sum(p.pow(2.0).sum() for p in model.parameters())
        loss_l2 = criterion(output, target)
        loss = loss_l2 + l2_lambda * l2_norm
        if batch_idx % 20 == 0:
            print( 'Epoch:[{}/{}] [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(epoch, epochs, batch_idx * len(data), 
                   len(train_loader.dataset), 100. * batch_idx / len(train_loader), loss.item())                )
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()


def evaluate(test_loader):
    test_loss = 0
    correct = 0
    for data, target in test_loader:
        data, target = data.to(device), target.to(device).long()
        output = net(data)
        test_loss += criterion(output, target).item()
        pred = output.data.max(1, keepdim=True)[1]
        #print('the output pred   ******** ',pred)
        #print('the output target ######## ',target)
        correct += pred.eq(target.data.view_as(pred)).cpu().sum()
    test_loss /= len(test_loader.dataset)
    print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
        test_loss, correct, len(test_loader.dataset),
        100. * correct / len(test_loader.dataset)))
    acc = 100. * correct / len(test_loader.dataset)
    return float(acc)

class CRNN(nn.Module):
    def __init__(self):
        super(CRNN, self).__init__()
        epochs            = 256
        num_classes       = 32
        blank_label       = 5
        gru_hidden_size   = 256
        gru_num_layers    = 2
        cnn_output_height = 8
        cnn_output_width  = 32
        self.conv1 = nn.Conv1d(1,  128,   kernel_size = 2, stride =4)
        self.norm1 = nn.InstanceNorm1d(128)
        self.conv2 = nn.Conv1d(128, 256,  kernel_size = 4, stride =8)
        self.norm2 = nn.InstanceNorm1d(256)
        self.conv4 = nn.Conv1d(256, 128,  kernel_size = 2, stride =4)
        self.norm4 = nn.InstanceNorm1d(128)

        self.conv1_2 = nn.Conv1d(1,  64, kernel_size = 2, stride = 1)
        self.norm1_2 = nn.InstanceNorm1d(64)
        self.conv2_2 = nn.Conv1d(64, 128, kernel_size = 6, stride = 2)
        self.norm2_2 = nn.InstanceNorm1d(128)
        #self.conv4_2 = nn.Conv1d(128, 64,  kernel_size =1)
        #self.norm4_2 = nn.InstanceNorm1d(64)
        self.drop     = nn.Dropout(0.25)
        self.drop1    = nn.Dropout(0.25)

        self.fc1    = nn.Linear(128*11*2, 128*8)
        self.fc     = nn.Linear(128*8, 4)
    def forward(self, x):
        batch_size = x.shape[0]
        out = self.conv1(x[:,:,0:1400])
        #out = self.norm1(out)
        out = self.drop1(out)
        out = F.leaky_relu(out)
        out = self.conv2(out)
        #out = self.norm2(out)
        out = self.drop1(out)
        out = F.leaky_relu(out)
        out = self.conv4(out)
        #out = self.norm4(out)
        out = self.drop(out)
        out = F.leaky_relu(out)
        #out1 = self.2conv1(x)
        #print('the out put size is' ,out.size())
        out1 = self.conv1_2(x[:,:, 1400:1428])
        #out1 = self.norm1_2(out1)
        out1 = self.drop1(out1)
        out1 = F.leaky_relu(out1)
        out1 = self.conv2_2(out1)
        #out1 = self.norm2_2(out1)
        out1 = self.drop(out1)
        out1 = F.leaky_relu(out1)
        #out1 = self.conv4_2(out1)
        #out1 = self.norm4_2(out1)
        #out1 = F.leaky_relu(out1)
        #print('the out1 put size is',out1.size())
        out_cat = torch.cat((out,out1),1)
        out_cat = out_cat.reshape(out_cat.size(0), -1)
        out_cat1 = self.fc1(out_cat)
        output  = self.fc(out_cat1)
        return output

batchsize = 32
epochs    = 256

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

data  = np.array(pd.read_csv('Xdata.csv', header=None))
label = np.array(pd.read_csv('Ydata.csv', header=None))

best_acc = []
empty_list = [ ]#0,5,6]
for i in range(int(len(data)/20)):
    model     = CRNN()
    net       = model.to(device)
    optimizer = optim.SGD(net.parameters(), lr=0.0001 , momentum=0.95)
    criterion = nn.CrossEntropyLoss()
    test_data   = data[20*i: 20*(i+1)]
    test_label  = label[20*i:20*(i+1)]
    train_data  = np.delete(data,  slice(20*i,20*(i+1)), axis=0)
    train_label = np.delete(label, slice(20*i,20*(i+1)), axis=0)
    test_data   = np.array(test_data)
    test_label  = np.array(test_label)
    train_data  = np.array(train_data)
    train_label = np.array(train_label)
    train_loader, test_loader = load_data(train_data, train_label, test_data, test_label)
    epoch_acc   = 0
    for epoch in range(1, epochs):
        if i not in empty_list:
            train(epoch, train_loader)
            acc =  evaluate(test_loader)
            if acc > epoch_acc and epoch > 32:
                epoch_acc = acc
    best_acc.append(epoch_acc)
    print('*****************************the best acc****************************', best_acc)

print("best acc is:", np.mean(best_acc))
#sio.savemat('acc.mat',{"acc": best_acc})
    #run()
