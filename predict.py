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

parser=argparse.ArgumentParser(description='For predict.py')

parser.add_argument('input', default='./flowers/test/12/image_04014.jpg', nargs='?', action="store", type = str)
parser.add_argument('--dir', action="store",dest="data_dir", default="./flowers/")
parser.add_argument('checkpoint', default='./checkpoint.pth', nargs='?', action="store", type = str)
parser.add_argument('--top_k', default=5, dest="top_k", action="store", type=int)
parser.add_argument('--category_names', dest="category_names", action="store", default='cat_to_name.json')
parser.add_argument('--gpu', default="gpu", action="store", dest="gpu")

args = parser.parse_args()
path_image = args.input
number_of_outputs = args.top_k
device = args.gpu
json_name = args.category_names
path = args.checkpoint

def main():
    models=model.load_checkpoint(path)
    with open('cat_to_name.json', 'r') as f:
        cat_to_name = json.load(f)
        
    probabilities,classes = model.predict(path_image, models, number_of_outputs, device)
    probability = np.array(probabilities[0])
    classes=np.array(classes[0])
    
    i = 0
    while i < number_of_outputs:
        print("{} with a probability of {}".format(classes[i], probability[i]))
        i += 1
    print("Finished Predicting!")
 

if __name__== "__main__":
    main()
    

