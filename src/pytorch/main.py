import argparse
import torch
import torch.nn as nn
import torchvision.transforms as T
from datasets import CarOrientationDataset
from models import ResNet18
from myconfig import *
from myconfig import *
from torch.autograd import Variable
from torch.optim import Adam
from torch.optim import lr_scheduler
from torch.utils.data import DataLoader


def test(data_loader, model, criterion):
    model.eval()

    matched,total = 0,0
    total_loss = 0.0
    for imgs, _, degree_bins in data_loader:
        imgs, degree_bins = Variable(imgs.cuda()), Variable(degree_bins.cuda())
        out = model(imgs)
        loss = criterion(out, degree_bins)
        total_loss+=float(loss.item())
        _, predicted = torch.max(out.data, 1)
        matched += predicted.eq(degree_bins.data).sum()
        total += degree_bins.size(0)
    acc = 100.*matched/total
    print('Total loss %.06f'% (total_loss/len(data_loader)))
    return acc


### arguments
parser = argparse.ArgumentParser()
parser.add_argument('--lr', type=float, default=0.005, help='Initial learning rate.')
parser.add_argument('--size', type=int, default=50, help='Size of reshaped image')
parser.add_argument('--weight_decay', type=float, default=5e-4, help='Weight decay (L2 loss on parameters).')
parser.add_argument('--epochs', type=int, default=100, help='Max num of epochs')
parser.add_argument('--batch_size', type=int, default=16, help='Batch size')
parser.add_argument('--workers', type=int, default=4, help='Num of workers for data loading')
parser.add_argument('--bins', type=int, default=16, help='Num bins')
parser.add_argument('--output', type=str, default='./model_state.pt', help='Save path for the trained model')
args = parser.parse_args()

### transofrm
transform_train = T.Compose([
        # T.Resize((args.size, args.size)),
        T.ToTensor(),
        T.Normalize(mean=[0.46872237, 0.4657492, 0.47433634], std=[0.10787867, 0.11330419, 0.11906589]),
])

transform_test = T.Compose([
        # T.Resize((args.size, args.size)),
        T.ToTensor(),
        T.Normalize(mean=[0.46872237, 0.4657492, 0.47433634], std=[0.10787867, 0.11330419, 0.11906589]),
])


### Data preparation
print('Preparing data')
lst = [l for l in open(cfg_annot_fn, 'r')]
lst.pop(0)

train_dataset = CarOrientationDataset(*cfg_extract_data_split(lst, cfg_train_views), transform=transform_train)
train_loader = DataLoader(train_dataset, batch_size=args.batch_size, num_workers=args.workers, drop_last=True, shuffle=True)
valid_dataset = CarOrientationDataset(*cfg_extract_data_split(lst, cfg_valid_views), transform=transform_test)
valid_loader = DataLoader(valid_dataset, batch_size=args.batch_size, num_workers=args.workers, drop_last=False, shuffle=False)

### model, optimizer, and loss
print('Configuring model')
model = ResNet18(16).cuda()
optimizer = Adam(model.parameters(),
        lr=args.lr,
        weight_decay=args.weight_decay)
exp_lr_scheduler = lr_scheduler.StepLR(optimizer, step_size=7, gamma=0.1)

criterion = nn.CrossEntropyLoss()

print('Performing initial validation')
acc = test(valid_loader,model, criterion)
print('Initial acc: {:.2f}'.format(acc))

best_test_acc = 0.0
print('=== Training the model ===')
for ep in range(args.epochs):
    total_loss = 0
    exp_lr_scheduler.step()
    model.train()
    for imgs, _, degree_bins in train_loader:
        imgs, degree_bins = Variable(imgs.cuda()), Variable(degree_bins.cuda())
        out = model(imgs)
        loss = criterion(out, degree_bins)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        total_loss+=float(loss.item())

    if (ep+1)%1==0:
        print('Epoch: {}\t Loss: {:.6f}'.format(ep,total_loss/len(train_loader)))
        acc = test(valid_loader,model, criterion)
        print('Validation acc: {:.2f}'.format(acc))
        if acc > best_test_acc:
            print('Best test acc %.03f.Saving the model to %s' % (acc, args.output))
            torch.save(model.state_dict(), args.output)
            best_test_acc = acc

print('=== Finished training the model ===')
print('Model saved to', args.output)
