import os
import torch
from torch import nn
from tqdm import tqdm
from torch import optim
from torchvision.datasets import ImageFolder
from torchvision import transforms
from torch.optim.lr_scheduler import LambdaLR

from ocrd_typegroups_classifier.network.vraec import vraec18
from ocrd_typegroups_classifier.network.vraec import vraec50
from ocrd_typegroups_classifier.typegroups_classifier import TypegroupsClassifier
from ocrd_typegroups_classifier.data.qloss import QLoss

# Loading and preparing the network
vraec = vraec50(layer_size=(96, 96, 96, 96), output_channels=11)
for l in range(2, 6):
    vraec.set_variational(l, False)
use_variational_layer = True

# Some settings for the training
learning_rate = 0.01
weight_decay = 0.0001
lr_decay = lambda epoch: 0.97 ** epoch
reconstruction_loss = nn.MSELoss()
classification_loss = nn.CrossEntropyLoss()
#optimizer = optim.SGD(vraec.select_parameters(), lr=learning_rate, weight_decay=weight_decay, momentum=0.9)
optimizer = optim.Adam(vraec.select_parameters(), lr=learning_rate, weight_decay=weight_decay)
scheduler = LambdaLR(optimizer, lr_lambda=[lr_decay])

# Creation of the typegroup classifier
tgc = TypegroupsClassifier(
    {
        'andere_schrift': 0,
        'antiqua': 1,
        'bastarda': 2,
        'fraktur': 3,
        'griechisch_kursiv': 4,
        'hebräisch': 5,
        'kursiv': 6,
        'nicht_schrift': 7,
        'rotunda': 8,
        'schwabacher': 9,
        'textura': 10
    },
    vraec
)

if os.path.exists(os.path.join('ocrd_typegroups_classifier', 'models', 'classifier.tgc')):
    tgc = TypegroupsClassifier.load(os.path.join('ocrd_typegroups_classifier', 'models', 'classifier.tgc'))

# Data transformation & loading
# Note that due to the rotation, having several sequential shearing
# transforms sequentially is not the same as having only one with
# a larger range.
trans = transforms.Compose([
    transforms.RandomAffine(12, shear=5),
    transforms.RandomResizedCrop(224, scale=(0.25, 1.0), ratio=(0.5, 1.5), interpolation=2),
    transforms.RandomAffine(12, shear=5),
    #transforms.RandomCrop(224),
    transforms.ColorJitter(brightness=0.5, contrast=0.5, saturation=0.1, hue=0.05),
    QLoss(min_q=5, max_q=80),
    transforms.ToTensor()
])
#training = ImageFolder('/cluster/seuret/patches/all', transform=trans)
# TODO : replace by correct path
training = ImageFolder('/disks/data1/seuret/ocrd/labelbox/training_data', transform=trans)
training.target_transform = tgc.classMap.get_target_transform(training.class_to_idx)


data_loader = torch.utils.data.DataLoader(training,
                                          batch_size=36,
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
    for sample, label in tqdm(data_loader):
        # Move data to device
        sample = sample.to(tgc.dev)
        label  = label.to(tgc.dev)
        
        # Training the (variational) autoencoder
        #tgc.network.set_variational(5, True)
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
    print('Epoch %d, loss %.1f, %d/%d=%.1f%%' % (epoch, lossSum, good, known, (100.0*good)/known))
    tgc.save(os.path.join('ocrd_typegroups_classifier', 'models', 'classifier.tgc'))
















































