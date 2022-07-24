import torch
import torch.nn as nn
import torch.nn.functional as F


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


def load_pretrained_models(model_names: list=None):
    """
    Loads pretrained CIFAR models from chenyaofo/pytorch-cifar-models
    """
    if model_names is None:
        model_names = [x for x in torch.hub.list('chenyaofo/pytorch-cifar-models') if '100' not in x]
    models = {}
    for model_name in model_names:
        models[model_name] = torch.hub.load('chenyaofo/pytorch-cifar-models', model_name, pretrained=True)
    return models