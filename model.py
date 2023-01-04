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
import utility_fun

def Setup_Network(structure='vgg16',hidden_units=1120, lr=0.002,device='gpu'):
    if structure == 'vgg16':
        model = models.vgg16(pretrained=True)
    elif structure == 'densenet121':
        model = models.densenet121(pretrained=True)
    #Using GPU if it is available
    device=t.device("cuda" if t.cuda.is_available() else "cpu") 


    for param in model.parameters():
        param.requires_grad=False
    
    classifier =nn.Sequential(OD([
        ('input',nn.Linear(25088,1000)),
        ('relu',nn.ReLU()),
        ('hidden_layer1',nn.Linear(1000,120)),
        ('output',nn.LogSoftmax(dim=1))
        ]))
    model.classifier=classifier
    criterion=nn.NLLLoss()
    optimizer=optim.Adam(model.classifier.parameters(),lr=0.002) 
    
    if torch.cuda.is_available() and device == 'gpu':
        device = torch.device("cuda:0")
    else:
        device = torch.device("cpu")
    model.to(device)
    
    return model, criterion#, optimizer

def save_check(train_data, structure='vgg16', lr=0.02, epochs= 5, path='checkpoint.pth', model )

    model.class_to_idx = image_datasets['train'].class_to_idx
    checkpoint={'structure': 'vgg16',
            'lr': 0.002,
             'epochs':5,
            'optimizer': optimizer.state_dict(),
            'state_dict': model.state_dict(),
            'classifier': model.classifier,
             'class_to_idx':model.class_to_idx
            }

    t.save(checkpoint, 'checkpoint.pth')
    
def load_checkpoint(path):
    
    checkpoint = t.load(path)
    model,_,_=Network()
    model.load_state_dict(checkpoint['state_dict'])
    return model

def process_image(image):    
    img_pil=Image.open(image)
    img_transforms=transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize([0.485,0.456,0.406],
                            [0.229,0.224,0.225])
    ])
    
    image=img_transforms(img_pil)    
    return image
    
    
def predict(image_path, model, topk=5):
    device=t.device("cuda" if t.cuda.is_available() else "cpu")
    models.to(device)
    models.eval()
    image=process_image(image_path).numpy()
    image=t.from_numpy(np.array([image])).float()
    
    with t.no_grad():
        image=image.to(device)
        log_ps=models.forward(image)
        ps=t.exp(log_ps)
        top_p,top_class=ps.topk(topk,dim=1)
        
    return top_p,top_class




