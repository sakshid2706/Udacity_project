import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import torch as t
from torch import nn
from torchvision import datasets,transforms,models
from collections import OrderedDict as OD
from torch import optim
from PIL import Image
import json



def load_data(root = "./flowers"):
    with open('cat_to_name.json', 'r') as f:
        cat_to_name = json.load(f)
    
    data_dir = root
    train_dir = data_dir + '/train'
    valid_dir = data_dir + '/valid'
    test_dir = data_dir + '/test''


data_transforms = {
    'train':transforms.Compose([transforms.RandomRotation(45),
                              transforms.RandomResizedCrop(224),
                              transforms.RandomVerticalFlip(),
                              transforms.ToTensor(),
                              transforms.Normalize((0.485, 0.456, 0.406),(0.229, 0.224, 0.225))]),
    'valid':transforms.Compose([transforms.Resize(255),
                              transforms.RandomResizedCrop(224),
                              transforms.ToTensor(),
                              transforms.Normalize((0.485, 0.456, 0.406),(0.229, 0.224, 0.225))]),
    'test':transforms.Compose([transforms.Resize(255),
                              transforms.RandomResizedCrop(224),
                              transforms.ToTensor(),
                              transforms.Normalize((0.485, 0.456, 0.406),(0.229, 0.224, 0.225))])
}

# TODO: Load the datasets with ImageFolder
image_datasets = {
     'train': datasets.ImageFolder(train_dir,transform=data_transforms['train']),
     'valid': datasets.ImageFolder(valid_dir,transform=data_transforms['valid']),
     'test': datasets.ImageFolder(valid_dir,transform=data_transforms['test'])
}

# TODO: Using the image datasets and the trainforms, define the dataloaders
dataloaders = {
    'train':t.utils.data.DataLoader(image_datasets['train'],batch_size=64,shuffle=True),
    'test':t.utils.data.DataLoader(image_datasets['test'],batch_size=64,shuffle=True),
    'valid':t.utils.data.DataLoader(image_datasets['valid'],batch_size=64,shuffle=True)
}

return dataloaders['train'], dataloaders['test'], dataloaders['valid'], image_datasets['train']