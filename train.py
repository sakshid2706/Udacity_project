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
import model
import argparse

parser=argparse.ArgumentParser(description='For train.py')

parser.add_argument('data_dir', action="store", default="./flowers/")
parser.add_argument('--save_dir', action="store", default="./checkpoint.pth")
parser.add_argument('--arch', action="store", default="vgg16",help='Architecture required')
parser.add_argument('--learning_rate', action="store", type=float,default=0.002)
parser.add_argument('--hidden_units', action="store", dest="hidden_units", type=int, default=1120)
parser.add_arrgument(--epochs, action="store", default=5, type=int)
parser.add_argument('--gpu', action="store", default="gpu")

args = parser.parse_args()
from_where = args.data_dir
path = args.save_dir
lr = args.learning_rate
struct = args.arch
hidden_units = args.hidden_units
model_to = args.gpu
epochs = args.epochs

if t.cuda.is_available() and power == 'gpu':
    device = t.device("cuda:0")
else:
    device = t.device("cpu")
    
def main():
    trainloader, validloader, testloader, train_data = utility_fun.load_data(from_where)
    model, criterion = fmodel.setup_network(struct,hidden_units,lr,model_to)
    optimizer = optim.Adam(model.classifier.parameters(), lr= 0.002)
    
    #training on train dataset #testing on validation set
    epochs=5
    steps=0
    running_loss=0
    print_every=5

    for epoch in range(epochs):
        for inputs, labels in dataloaders['train']:
            steps+=1
            inputs,labels=inputs.to(device),labels.to(device)
            optimizer.zero_grad()
            outputs=model.forward(inputs)
            loss=criterion(outputs,labels)
            loss.backward()
            optimizer.step()
        
            running_loss+=loss.item()
        
            if steps% print_every==0:
                valid_loss=0
                accuracy=0
                model.eval()
                with t.no_grad():
                    for inputs,labels in dataloaders['valid']:
                        
                            inputs,labels=inputs.to(device),labels.to(device)
                    log_ps=model.forward(inputs)
                    batch_loss=criterion(log_ps,labels)
                    valid_loss+=batch_loss.item()
                    
                    ps=t.exp(log_ps)
                    top_p,top_class=ps.topk(1,dim=1)
                    equals = top_class == labels.view(*top_class.shape)
                    accuracy += t.mean(equals.type(t.FloatTensor)).item()
            
            print("Epoch: ",epoch+1,
                 "Loss: %.3f" %(running_loss/print_every),
                 "Validation Loss: %.3f" %(valid_loss/print_every),
                 "Accuracy: %.3f" %(accuracy/len(dataloaders['valid'])))
            running_loss=0
            model.train()
            
            
# TODO: Save the checkpoint 
model.class_to_idx = image_datasets['train'].class_to_idx
checkpoint={'input_size': 25088,
           'output_size': 102,
           'structure': 'vgg16',
           'lr': 0.002,
           'epochs':5,
           'optimizer': optimizer.state_dict(),
           'state_dict': model.state_dict(),
           'classifier': model.classifier,
            'class_to_idx':model.class_to_idx
           }

t.save(checkpoint, 'checkpoint.pth')

if __name__ == "__main__":
    main()