import argparse
import time
import torch
import torch.nn as nn
import torchvision.transforms as T
import math
from datasets import DegOrientationDataset
from models import ResNet18
from myconfig import *
from torch.autograd import Variable
from torch.optim import Adam
from torch.optim import lr_scheduler
from torch.utils.data import DataLoader
import datetime

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print('Torch device:', device)


def test(data_loader, model):
    with torch.no_grad():
        matched = 0
        total = 0
        total_loss = 0.0
        tested_samples = 0
        num_samples = len(data_loader)
        for imgs, degrees, bins in data_loader:
            imgs, orientations, directions = imgs.to(device), degrees.to(device), bins.to(device)
            p = model(imgs)
            loss = criterion_direction(p, directions)
            total_loss += float(loss.item())

            _, predicted = torch.max(p.data, 1)
            matched += predicted.eq(directions.data).sum()

            total += directions.size(0)
            tested_samples += 1
            if tested_samples % 1000 == 0:
                print(f'[{tested_samples}/{num_samples}]    Loss: {loss}     Total Loss: {total_loss}')

        acc = float(matched) / total
        return acc


### arguments
parser = argparse.ArgumentParser()
parser.add_argument('--lr', type=float, default=0.16, help='Initial learning rate.')
parser.add_argument('--size', type=int, default=224, help='Size of reshaped image')
parser.add_argument('--weight_decay', type=float, default=1e-8, help='Weight decay (L2 loss on parameters).')
parser.add_argument('--epochs', type=int, default=100, help='Max num of epochs')
parser.add_argument('--batch_size', type=int, default=32, help='Batch size')
parser.add_argument('--num_workers', type=int, default=16, help='Num of workers for data loading')
parser.add_argument('--bins', type=int, default=16, help='Num bins')
parser.add_argument('--early_stop', type=int, default=10, help='Num. epochs with no improvement')
parser.add_argument('--output', type=str, default='./model_state.pth', help='Save path for the trained model')
parser.add_argument('--valid_freq', type=int, default=5, help="Epoch frequency under which the model is validated")
args = parser.parse_args()

### transofrm
transform_train = T.Compose([
        T.Resize((args.size, args.size)),
        T.ToTensor(),
        T.Normalize(mean=[0.46872237, 0.4657492, 0.47433634], std=[0.10787867, 0.11330419, 0.11906589]),
])

transform_test = T.Compose([
        T.Resize((args.size, args.size)),
        T.ToTensor(),
        T.Normalize(mean=[0.46872237, 0.4657492, 0.47433634], std=[0.10787867, 0.11330419, 0.11906589]),
])


### Data preparation
print('Preparing data')
lst = [l.strip() for l in open(cfg_anno_path, 'r')]
lst.pop(0) # remove header from csv file

train_dataset = DegOrientationDataset(*cfg_extract_data_split(lst, cfg_valid_views), transform=transform_train)
train_loader = DataLoader(train_dataset, batch_size=args.batch_size, num_workers=args.num_workers, drop_last=True, shuffle=True)
valid_dataset = DegOrientationDataset(*cfg_extract_data_split(lst, cfg_valid_views), transform=transform_test)
valid_loader = DataLoader(valid_dataset, batch_size=args.batch_size, num_workers=args.num_workers, drop_last=False, shuffle=False)

### model, optimizer, and loss
print('Configuring model')
model = ResNet18(num_classes=4, use_pretrained=True).cuda()
optimizer = Adam(model.parameters(),
        lr=args.lr,
        weight_decay=args.weight_decay)
exp_lr_scheduler = lr_scheduler.StepLR(optimizer, step_size=7, gamma=0.1)

criterion_orientation = nn.CrossEntropyLoss()
criterion_direction = nn.CrossEntropyLoss()

print('=== Running Initial Validation ===')
test(valid_loader, model)

print('=== Training the model ===')
now = datetime.datetime.now()
print(now.strftime("%Y-%m-%d %H:%M"))
best_test_acc = 0.0
for ep in range(args.epochs):
    start_time = time.time()
    new_best = False
    epoch_total_loss = 0
    exp_lr_scheduler.step()
    model.train()
    num_batches = len(train_loader)
    epoch_total_loss = 0.
    for index, loader_item in enumerate(train_loader):
        imgs, degrees, directions = loader_item
        imgs, degrees, directions = imgs.to(device), degrees.to(device), directions.to(device)
        optimizer.zero_grad()
        output = model(imgs)
        loss = criterion_direction(output, directions)
        # loss = angular_loss(out, degrees)
        loss.backward()
        optimizer.step()

        epoch_total_loss += float(loss.item())

    print('{}Epoch {} Loss: {:.6f}\t Total Loss: {:.6f}\tTime:{}'.format(
        '*' if new_best else '',
        ep,
        epoch_total_loss / num_batches,
        epoch_total_loss,
        time.strftime('%H:%M:%S', time.gmtime(time.time() - start_time))))

    if (ep+1) % 5 == 0:
        print('=== Validating the model ===')
        acc = test(valid_loader, model)
        print('Validation Accuracy: {:.4f}'.format(acc))
        if acc > best_test_acc:
            new_best = True
            print('Best test acc %.03f.Saving the model to %s' % (acc, args.output))
            torch.save(model.state_dict(), args.output)
            best_test_acc = acc


print('=== Finished training the model ===')
print('Model saved to', args.output)
