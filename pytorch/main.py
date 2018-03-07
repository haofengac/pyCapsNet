import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms, datasets
import time, os
from torch.autograd import Variable
from modules import *

def p(s):
    print(s)
    print('pausing')
    time.sleep(10000)

class MNIST:
    def __init__(self, bs=1):
        dataset_transform = transforms.Compose([
                       transforms.ToTensor(),
                       transforms.Normalize((0.1307,), (0.3081,))
                   ])

        train_dataset = datasets.MNIST('data', train=True, download=True, transform=dataset_transform)
        eval_dataset = datasets.MNIST('data', train=False, download=True, transform=dataset_transform)
        
        self.train_dataloader  = torch.utils.data.DataLoader(train_dataset, batch_size=bs, shuffle=True)
        self.eval_dataloader = torch.utils.data.DataLoader(eval_dataset, batch_size=bs, shuffle=True)

bs = 128
lr = 1e-3
use_cuda = True
optimizer = 'adam'
num_epochs = 100
disp_interval = 100
save_epoch = 1
val_epoch = 1
project_id = 'capsnet'
num_classes = 10
reconst_factor = 0.0005
save_dir = 'saved_models'
num_epochs = 100

if not os.path.exists(save_dir):
    os.makedirs(save_dir)

mnist = MNIST(bs=bs)
 # Variables
inputs = torch.FloatTensor(1)
labels = torch.FloatTensor(1)
eye = Variable(torch.eye(num_classes))
inputs = Variable(inputs)
labels = Variable(labels)

# Model
model = CapsNet(use_cuda=use_cuda)

# cuda
if use_cuda:
    inputs = inputs.cuda()
    labels = labels.cuda()
    model = model.cuda()
    eye = eye.cuda()

params = []

for key, value in dict(model.named_parameters()).items():
    if value.requires_grad:
        params += [{'params':[value],'lr':lr}]

# optimizer
if optimizer == "adam":
    optimizer = torch.optim.Adam(model.parameters())
elif optimizer == "sgd":
    optimizer = torch.optim.SGD(params)

criterion = CapsLoss()

print('Training started!')

for epoch in range(num_epochs):
    start = time.time()
    
    # train
    model.train()
    correct = 0
    train_loss = 0
    for batch_idx, (imgs, targets) in enumerate(mnist.train_dataloader):
        if imgs.size(0) != bs:
            continue
        targets = torch.eye(num_classes).index_select(dim=0, index=targets)
        inputs.data.resize_(imgs.size()).copy_(imgs)
        labels.data.resize_(targets.size()).copy_(targets)
        
        optimizer.zero_grad()
        outputs, reconst = model(inputs)
        
        scores = torch.sqrt((outputs ** 2).sum(2))
        loss = criterion(scores, labels, reconst, inputs)
        train_loss = loss.data.cpu().numpy()[0]

        # backward
        loss.backward()
        optimizer.step()

        scores, classes = F.softmax(scores).max(dim=1)
        predicted = eye.index_select(dim=0, index=classes.squeeze(1))
        
        predicted_idx = np.argmax(predicted.data.cpu().numpy(),1)
        label_idx = np.argmax(targets.numpy(), 1)
        correct = np.sum(predicted_idx == label_idx)
        
        # info
        if batch_idx % disp_interval == 0:
            end = time.time()
            print("[epoch %2d][iter %4d] loss: %.4f, acc: %.4f%% (%d/%d)" \
                            % (epoch, batch_idx, train_loss/(batch_idx+1), 100.*correct/bs, correct, bs))

    save_name = os.path.join(save_dir, '{}_{}.pth'.format(project_id, epoch))
    if save_epoch > 0 and batch_idx % save_epoch == 0:
        torch.save({
          'epoch': epoch,
        }, save_name)

    # val
    if epoch % val_epoch == 0:
        print('Validating...')
        correct = 0
        total = 0
        model.eval()
        for batch_idx, (imgs, targets) in enumerate(mnist.eval_dataloader):
            if imgs.size(0) != bs:
                continue
            targets = torch.eye(num_classes).index_select(dim=0, index=targets)
            inputs.data.resize_(imgs.size()).copy_(imgs)
            labels.data.resize_(targets.size()).copy_(targets)
            
            outputs, reconst = model(inputs)
            scores = torch.sqrt((outputs ** 2).sum(2))
            scores, classes = F.softmax(scores).max(dim=1)
            predicted = eye.index_select(dim=0, index=classes.squeeze(1))
        
            predicted_idx = np.argmax(predicted.data.cpu().numpy(),1)
            label_idx = np.argmax(targets.numpy(), 1)
            correct += np.sum(predicted_idx == label_idx)
            total += targets.size(0)
        print("[epoch %2d] val acc: %.4f%% (%d/%d)" \
                                % (epoch, 100.*correct/total, correct, total))
