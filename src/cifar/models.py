import torch
import torch.nn as nn
import torch.nn.functional as F

# Hyperparameters

train_batch_size = 100
test_batch_size = 100
n_epochs = 5
learning_rate = 4e-2
seed = 100
input_dim = 28*28
out_dim = 10
num_hidden_layers = 4
momentum = 0.9
kernel_size = 3


# Fully connected model
class FC(nn.Module):
    
    def __init__(self, in_dim, out_dim, num_hidden_layers, layer_size):
        super().__init__()

        self.num_layers = num_hidden_layers * 2 + 3 # *2 accounts for ReLU layers, +3 is input layer, input relu layer, output layer

        self.in_dim = in_dim
        self.out_dim = out_dim        

        self.layer_size = layer_size

        self.layer_list = nn.ModuleList()

        self.layer_list.append(nn.Linear(self.in_dim, self.layer_size))
        self.num_hidden_layers = num_hidden_layers

        for i in range(1,self.num_hidden_layers):
            self.layer_list.append(nn.Linear(self.layer_size, self.layer_size))
            

        self.layer_list.append(nn.Linear(self.layer_size, self.out_dim))
        
    def forward(self, x):

        x = x.view(-1, self.in_dim)

        for i in range(self.num_hidden_layers):
            x = F.relu(self.layer_list[i](x))

        return self.layer_list[self.num_hidden_layers](x)


# Convolutional model
class CNN(nn.Module):
    
    def __init__(self, in_dim, out_dim, num_hidden_layers):
        super().__init__()

        self.num_layers = num_hidden_layers * 2 + 3 # *2 accounts for ReLU layers, +3 is input layer, input relu layer, output layer

        self.in_dim = in_dim
        self.out_dim = out_dim        

        self.layer_list = nn.ModuleList()

        
        self.layer_list.append(nn.Conv2d(1,8,kernel_size, padding=1))
        self.num_hidden_layers = num_hidden_layers

        last_channel = 8
        for i in range(1,self.num_hidden_layers):
            self.layer_list.append(nn.Conv2d(last_channel, last_channel+8, kernel_size, padding=1))
            last_channel = last_channel+8
            
        self.layer_list.append(nn.BatchNorm2d(last_channel))
        self.layer_list.append(nn.MaxPool2d(2))
        self.layer_list.append(nn.Dropout(p=0.2))

        if num_hidden_layers == 1 or num_hidden_layers == 0:
          self.layer_list.append(nn.Linear(1568, self.out_dim))
        elif num_hidden_layers == 2:
          self.layer_list.append(nn.Linear(784, self.out_dim))
        elif num_hidden_layers == 3:
          self.layer_list.append(nn.Linear(216, self.out_dim))
        elif num_hidden_layers == 4:
          self.layer_list.append(nn.Linear(32, self.out_dim))
        
    def forward(self, x):

        x = self.layer_list[-3](F.relu(self.layer_list[0](x)))

        for i in range(1,self.num_hidden_layers):
          x = self.layer_list[-3](F.relu(self.layer_list[i](x)))

        x = self.layer_list[-4](x)

        x = torch.flatten(x,1)
        x = F.relu(self.layer_list[-2](x))
        x = F.relu(self.layer_list[-1](x))

        return x