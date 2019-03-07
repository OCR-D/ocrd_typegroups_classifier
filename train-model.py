import os
import torch
from torch import nn
from torch import optim
from torchvision.datasets import ImageFolder
from torchvision import transforms
from torch.optim.lr_scheduler import LambdaLR

from ocrd_typegroups_classifier.network.vraec import vraec18
from ocrd_typegroups_classifier.network.vraec import vraec50
from ocrd_typegroups_classifier.typegroups_classifier import TypegroupsClassifier
from ocrd_typegroups_classifier.data.qloss import QLoss

# Loading and preparing the network
vraec = vraec50(layer_size=128, output_channels=16)
for l in range(2, 6):
    vraec.set_variational(l, False)
use_variational_layer = True

# Some settings for the training
learning_rate = 0.001
weight_decay = 0.0001
lr_decay = lambda epoch: 0.97 ** epoch
reconstruction_loss = nn.MSELoss()
classification_loss = nn.CrossEntropyLoss()
optimizer = optim.SGD(vraec.select_parameters(), lr=learning_rate, weight_decay=weight_decay, momentum=0.9)
scheduler = LambdaLR(optimizer, lr_lambda=[lr_decay])

# Creation of the typegroup classifier
tgc = TypegroupsClassifier(
    {
        'Antiqua':0,
        'Bastarda':1,
        'Fraktur':2,
        'Griechisch':3,
        'Hebräisch':4,
        'Kursiv':5,
        'Rotunda':6,
        'Textura':7,
        'Adornment':8,
        'Book covers and other irrelevant data':9,
        'Empty Pages':10,
        'Woodcuts - Engravings':11
    },
    vraec
)

# Data transformation & loading
# Note that due to the rotation, having several sequential shearing
# transforms sequentially is not the same as having only one with
# a larger range.
trans = transforms.Compose([
    transforms.RandomAffine(4, shear=10),
    transforms.RandomAffine(4, shear=10),
    transforms.RandomAffine(4, shear=10),
    transforms.RandomCrop(224),
    transforms.ColorJitter(brightness=0.5, contrast=0.5, saturation=0.1, hue=0.05),
    QLoss(min_q=5, max_q=80),
    transforms.ToTensor()
])
#training = ImageFolder('/cluster/seuret/patches/all', transform=trans)
# TODO : replace by correct path
training = ImageFolder('../extracted/samples', transform=trans)
training.target_transform = tgc.classMap.get_target_transform(training.class_to_idx)


data_loader = torch.utils.data.DataLoader(training,
                                          batch_size=64,
                                          shuffle=True,
                                          num_workers=4)

# Iterating over the data
print('Starting the training - grab a coffee and a good book!')
for epoch in range(200):
    # Modify learning rate
    scheduler.step()
    
    # Iterate over the data
    lossSum = 0
    good = 0
    known = 0
    for sample, label in data_loader:
        # Move data to device
        sample = sample.to(tgc.dev)
        label  = label.to(tgc.dev)
        
        # Training the (variational) autoencoder
        #tgc.network.set_variational(5, use_variational_layer)
        #rl = tgc.network.finetune(sample, optimizer=optimizer, loss_function=reconstruction_loss)
        tgc.network.set_variational(5, False)
        
        # Training the classifier on samples with known labels
        sample, label = tgc.filter(sample, label)
        if len(label)==0: # no known labels
            continue
        out, _, _ = tgc.network(sample)
        closs = classification_loss(out, label)
        optimizer.zero_grad()
        closs.backward()
        optimizer.step()
        lossSum += closs.item()
        
        # Computing accuracy
        _, p = torch.max(out, 1)
        good += torch.sum(p==label).item()
        known += len(label)
        print(lossSum)
    print('Epoch %d, loss %.1f, %d/%d=%.1f%%' % (epoch, lossSum, good, known, (100.0*good)/known))
    tgc.save(os.path.join('ocrd_typegroups_classifier', 'models', 'classifier.tgc'))
















































