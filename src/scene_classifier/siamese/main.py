import argparse
import torchvision.datasets as dset
import torchvision.transforms as transforms
from datasets import SiameseNetworkDataset
from easydict import EasyDict as edict
from loss import ContrastiveLoss
from models import SiameseNetwork
from torch.autograd import Variable
from torch.optim import Adam
from torch.utils.data import DataLoader
import torch
import torchvision
import matplotlib.pyplot as plt

# Aux funcs
def imshow(img,text=None,should_save=False):
    npimg = img.numpy()
    plt.axis("off")
    if text:
        plt.text(75, 8, text, style='italic',fontweight='bold',
            bbox={'facecolor':'white', 'alpha':0.8, 'pad':10})
    plt.imshow(np.transpose(npimg, (1, 2, 0)))
    plt.show()

def show_plot(iteration,loss):
    plt.plot(iteration,loss)
    plt.show()

# Configuration
Config = edict({})
Config.training_dir = ""
Config.testing_dir = ""
Config.train_batch_size = 64
Config.train_number_epochs = 100
Config.num_workers = 16
Config.contrastive_margin = 1.0
Config.learning_rate = 0.0005

# Data
train_dataset = SiameseNetworkDataset(imageFolderDataset=dset.ImageFolder(root=Config.training_dir),
                                        transform=transforms.Compose([transforms.Resize((100,100)),
                                                                      transforms.ToTensor()
                                                                      ])
                                       ,should_invert=False)
train_dataloader = DataLoader(train_dataset,
                        shuffle=True,
                        num_workers=Config.num_workers,
                        batch_size=Config.train_batch_size)
test_dataset = SiameseNetworkDataset(imageFolderDataset=dset.ImageFolder(root=Config.testing_dir),
                                        transform=transforms.Compose([transforms.Resize((100,100)),
                                                                      transforms.ToTensor()
                                                                      ])
                                       ,should_invert=False)
test_dataloader = DataLoader(test_dataset,
                        shuffle=True,
                        num_workers=Config.num_workers,
                        batch_size=Config.train_batch_size)

# Visualizing the data
vis_dataloader = DataLoader(siamese_dataset,
                        shuffle=True,
                        num_workers=Config.num_workers,
                        batch_size=Config.batch_size)
dataiter = iter(vis_dataloader)
example_batch = next(dataiter)
concatenated = torch.cat((example_batch[0],example_batch[1]),0)
imshow(torchvision.utils.make_grid(concatenated))
print(example_batch[2].numpy())

# Model declaration
net = SiameseNetwork().cuda()
criterion = ContrastiveLoss(margin=Config.contrastive_margin)
optimizer = Adam(net.parameters(),lr = Config.learning_rate)

# Training
counter = []
loss_history = []
iteration_number = 0

for epoch in range(0,Config.train_number_epochs):
    for i, data in enumerate(train_dataloader,0):
        img0, img1, label = data
        img0, img1, label = Variable(img0).cuda(), Variable(img1).cuda() , Variable(label).cuda()
        output1, output2 = net(img0, img1)
        optimizer.zero_grad()
        loss_contrastive = criterion(output1, output2, label)
        loss_contrastive.backward()
        optimizer.step()
        if i %10 == 0 :
            print("Epoch number {}\n Current loss {}\n".format(epoch, loss_contrastive.data[0]))
            iteration_number +=10
            counter.append(iteration_number)
            loss_history.append(loss_contrastive.data[0])

show_plot(counter,loss_history)