import os
import copy
import numpy as np
import torch
import torch.nn.functional as F
import json


class NoDimRedBackbone(torch.nn.Module):
    """
    CNN which does not reduce horizontal dimensions of the input.
    Useful for pixel column labeling.
    """
    def __init__(self, output_dim=32):
        """
        Constructor

        :param output_dim: number of neurons in the output layer
        :return: instance of the class
        """ 
        super().__init__()
        self.act = torch.nn.LeakyReLU()
        self.max_pool2 = torch.nn.MaxPool2d(kernel_size=(3,1))
        self.conv1 = torch.nn.Conv2d(1, 8, (7, 5), stride=(1,1), padding='same')
        self.conv2 = torch.nn.Conv2d(8, 32, (7, 5), padding='same')
        self.conv3 = torch.nn.Conv2d(32, output_dim, (3, 3), padding='same')
        self.padding_params = 1/8
        self.output_dim = output_dim

    def forward(self, x):
        """
        Extracts features from an input text line

        :param x: text line (batch)
        :return: descriptors (batch)
        """ 
        x = self.act(self.conv1(x))
        x = self.act(self.conv2(x))
        x = self.max_pool2(x)
        x = self.act(self.conv3(x))
        x = self.max_pool2(x)
        return x.mean(axis=2)


class ColClassifier(torch.nn.Module):
    """
    This class is used for classifying the pixel columns of a text line.
    It is getting outdated and might be removed at a later stage.
    """
    def __init__(self, backbone, feature_dim, nb_classes=12):
        super().__init__()
        self.backbone = backbone
        self.embed = torch.nn.Linear(self.backbone.output_dim, feature_dim)
        self.rnn  = torch.nn.LSTM(feature_dim, feature_dim, 2, bidirectional=True)
        self.head = torch.nn.Linear(2*feature_dim, nb_classes)
    
    def forward(self, x):
        x = self.embed(self.backbone(x).transpose(1,2).transpose(0,1))
        x, _ = self.rnn(x)
        x = self.head(x)
        return x.transpose(0,1)
    
    def save(self, folder):
        os.makedirs(folder, exist_ok=True)
        torch.save(self.backbone.state_dict(), os.path.join(folder, 'backbone.pth'))
        torch.save(self.embed.state_dict(), os.path.join(folder, 'embed.pth'))
        torch.save(self.rnn.state_dict(), os.path.join(folder, 'rnn.pth'))
        torch.save(self.head.state_dict(), os.path.join(folder, 'head.pth'))
    
    def load(self, folder, device='cuda:0'):
        self.backbone.load_state_dict(torch.load(os.path.join(folder, 'backbone.pth'), map_location=device))
        self.embed.load_state_dict(torch.load(os.path.join(folder, 'embed.pth'), map_location=device))
        self.rnn.load_state_dict(torch.load(os.path.join(folder, 'rnn.pth'), map_location=device))
        self.head.load_state_dict(torch.load(os.path.join(folder, 'head.pth'), map_location=device))
