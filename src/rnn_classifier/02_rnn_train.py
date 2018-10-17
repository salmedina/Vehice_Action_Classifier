import sys
from myconfig import *
import pickle
import numpy as np

import torch
import torch.nn as nn
from torch.autograd import Variable
from model import Bid_RNN

model = Bid_RNN(2, 4, class_num=4).cuda()
model.train()
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

featlst,lablst = [], []
for vid in train_vid_list:
    feats, labs = pickle.load(open('data/'+vid+'.pkl','rb'),encoding='latin1')
    featlst+=feats
    lablst+=labs
print('tracklet num:', len(featlst))

for epoch in range(100):
    total_loss = 0
    for i in np.random.permutation(len(featlst)).tolist():
        feat, labs = featlst[i], lablst[i]
        feat = Variable(torch.from_numpy(feat).cuda())
        labs = Variable(torch.from_numpy(labs).long().cuda())
        feat = feat.unsqueeze(0)
        out = model(feat)
        loss = criterion(out, labs)
        total_loss+=loss.data[0]

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    print('epoch: {}\t loss:{}'.format(epoch, total_loss/len(featlst)))

torch.save(model.state_dict(), 'rnn_state.pt')
