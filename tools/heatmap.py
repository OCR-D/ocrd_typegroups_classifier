from __future__ import print_function
import sys
import argparse
import torch
import torch.utils.data
from torch import nn, optim
from torch.nn import functional as F
from torchvision import datasets, transforms
from torchvision.utils import save_image
from torchvision.datasets import ImageFolder
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, utils, datasets
from PIL import Image
import sys
from math import exp, ceil, floor
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
import matplotlib

sys.path.append("../ocrd_typegroups_classifier")

from ocrd_typegroups_classifier.typegroups_classifier import TypegroupsClassifier

parser = argparse.ArgumentParser()
parser.add_argument("network", help="path to the network (.tgc) file", type=str)
parser.add_argument("input", help="input page image", type=str)
parser.add_argument("output", help="output file base", type=str)
args = parser.parse_args()

# Feel free to change it
stride = 53
crop_size = 53

# If you use a GPU, switch to 32 or 64
batch_size = 1

# Loading the image
im = Image.open(args.input).convert('RGB')
tensorize = transforms.ToTensor()

# Loading the classifier
tgc = TypegroupsClassifier.load(args.network)
classes = tgc.classMap.id2cl
print('Considered classes:', classes)

# Rescaling stuff
original_size = im.size
rescaling_ratio = 1000 / original_size[0]
im = im.resize((round(original_size[0] * rescaling_ratio), round(original_size[1] * rescaling_ratio)))
insize = round(crop_size / rescaling_ratio)
ostride = round(stride / rescaling_ratio)

valid_pos = []
for x in range(0, im.size[0]-crop_size, stride):
    for y in range(0, im.size[1]-crop_size, stride):
        valid_pos.append((x, y))

with torch.no_grad():
    batch = []
    position = []
    result = np.zeros((len(classes), original_size[1], original_size[0]))
    count = 0
    for x, y in tqdm(valid_pos):
        crop = tensorize(im.crop((x, y, x+crop_size, y+crop_size)))
        #crop = crop[0:3,:,:]
        batch.append(crop)
        position.append((x,y))
        if len(batch) >= batch_size:
            count += len(batch)
            tensors = torch.stack(batch).to(tgc.dev)
            out, _, _ = tgc.network(tensors)
            for bn in range(out.size(0)):
                p = position[bn]
                p = (round(p[0] / rescaling_ratio), round(p[1] / rescaling_ratio))
                
                if p[0]==0:
                    fromx=0
                else:
                    fromx=ceil(p[0]+insize/2-ostride/2)
                if p[0]+stride+224 >= original_size[0]:
                    tox = min(p[0]+insize, original_size[0])
                else:
                    tox = floor(p[0]+insize/2+ostride/2)
                
                if p[1]==0:
                    fromy=0
                else:
                    fromy=ceil(p[1]+insize/2-ostride/2)
                if p[1]+stride+insize >= original_size[1]:
                    toy = min(p[1]+insize, original_size[1])
                else:
                    toy = floor(p[1]+insize/2+ostride/2)
                
                for i in classes:
                    result[i, fromy:toy, fromx:tox] = out[bn, i].item()
            
            batch = []
            position = []
batch = []
position = []
tgc = None

print('Storing results...')
im = Image.open(args.input).convert('RGB')
a = np.max(result[:, :, :])
i = np.min(result[:, :, :])
hsv = matplotlib.colors.rgb_to_hsv(np.array(im))
for n in classes:
    hsv[:, :, 0] = (result[n,:,:] - i) / (a-i) / 3.0
    hsv[:, :, 1] = np.sqrt(6 * np.abs(1/6.0 - hsv[:, :, 0]))
    npa = matplotlib.colors.hsv_to_rgb(hsv).astype('uint8')
    img = Image.fromarray(npa)
    img.save('%s-%s.jpg' % (sys.argv[3], classes[n]))
