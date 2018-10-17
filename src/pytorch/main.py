import sys
import argparse
from myconfig import *
from datasets import CarOrientationDataset
from models import ResNet18, ResNet50

import torch
import torch.nn as nn
from torch.autograd import Variable
from torch.utils.data import DataLoader
from torch.optim import lr_scheduler, Adam
import torchvision.transforms as T

def test(data_loader, model, criterion):
    model.eval()

    matched,total = 0,0
    total_loss = 0.0
    for imgs, _, degree_bins in data_loader:
        imgs, degree_bins = Variable(imgs.cuda()), Variable(degree_bins.cuda())
        out = model(imgs)
        loss = criterion(out, degree_bins)
        total_loss+=loss.data[0]
        _, predicted = torch.max(out.data, 1)
        matched+=predicted.eq(degree_bins.data).sum()
        total+=degree_bins.size(0)
    acc = 100.*matched/total
    print('total loss', total_loss/len(data_loader))
    return acc


### arguments
parser = argparse.ArgumentParser()
parser.add_argument('--lr', type=float, default=0.005, help='Initial learning rate.')
parser.add_argument('--size', type=int, default=50, help='Size of reshaped image')
parser.add_argument('--weight_decay', type=float, default=0, help='Weight decay (L2 loss on parameters).')
parser.add_argument('--epochs', type=int, default=20, help='max num of epochs')
parser.add_argument('--batch_size', type=int, default=16, help='batch size')
parser.add_argument('--workers', type=int, default=4, help='num of workers for data loading')
args = parser.parse_args()

### transofrm
transform_train = T.Compose([
        T.Resize((args.size, args.size)),
        T.ToTensor(),
        T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

transform_test = T.Compose([
        T.Resize((args.size, args.size)),
        T.ToTensor(),
        T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])


### Data preparation
lst = [l for l in open(cfg_annot_fn, 'r')]
lst.pop(0)

train_dataset = CarOrientationDataset(*cfg_extract_data_split(lst, cfg_train_views), transform=transform_train)
train_loader = DataLoader(train_dataset, batch_size=args.batch_size, num_workers=args.workers, drop_last=True,shuffle=True)
valid_dataset = CarOrientationDataset(*cfg_extract_data_split(lst, cfg_valid_views), transform=transform_test)
valid_loader = DataLoader(valid_dataset, batch_size=args.batch_size, num_workers=args.workers, drop_last=False, shuffle=False)

### model, optimizer, and loss
model = ResNet18(8).cuda()
optimizer = Adam(model.parameters(),
        lr=args.lr,
        weight_decay=args.weight_decay)
criterion = nn.CrossEntropyLoss()

#acc = test(valid_loader,model)
#print('initial acc:',acc)
best_acc = 0
for ep in range(args.epochs):
    total_loss = 0
    model.train()
    for imgs, _, degree_bins in train_loader:
        imgs, degree_bins = Variable(imgs.cuda()), Variable(degree_bins.cuda())
        out = model(imgs)
        loss = criterion(out, degree_bins)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        total_loss+=loss.data[0]
    if (ep+1)%1==0:
        print('epoch: {}\t loss: {:.6f}'.format(ep,total_loss/len(train_loader)))
        acc = test(valid_loader,model,criterion)
        print('Validation acc: {:.2f}'.format(acc))
        if best_acc>acc:
            torch.save(model.state_dict(),'model_state.pt')
