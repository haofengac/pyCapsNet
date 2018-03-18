import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms, datasets
import time, os, argparse
from torch.autograd import Variable
from modules import *


class MNIST:
    def __init__(self, bs=1):
        dataset_transform = transforms.Compose([
                       transforms.ToTensor(),
                       transforms.Normalize((0.1307,), (0.3081,))
                   ])

        train_dataset = datasets.MNIST('data', train=True, download=True, transform=dataset_transform)
        eval_dataset = datasets.MNIST('data', train=False, download=True, transform=dataset_transform)
        
        self.num_classes = 10
        self.train_dataloader  = torch.utils.data.DataLoader(train_dataset, batch_size=bs, shuffle=True)
        self.eval_dataloader = torch.utils.data.DataLoader(eval_dataset, batch_size=bs, shuffle=True)

def parse_args():
    """
    Parse input arguments
    """
    parser = argparse.ArgumentParser(description='Cupy Capsnet')
    parser.add_argument('--bs', dest='bs',
                      help='batch size',
                      default='100', type=int)
    parser.add_argument('--lr', dest='lr',
                      help='learning rate',
                      default=1e-2, type=float)
    parser.add_argument('--opt', dest='optimizer',
                      help='optimizer',
                      default='adam', type=str)
    parser.add_argument('--disp', dest='disp_interval',
                      help='interval to display training loss',
                      default=1, type=int)
    parser.add_argument('--num_epochs', dest='num_epochs',
                      help='num epochs to train',
                      default=100, type=int)
    parser.add_argument('--val_epoch', dest='val_epoch',
                      help='num epochs to run validation',
                      default=1, type=int)
    parser.add_argument('--save_epoch', dest='save_epoch',
                      help='num epochs to save model',
                      default=1, type=int)
    parser.add_argument('--use_cuda', dest='use_cuda',
                      help='whether or not to use cuda',
                      default=True, type=bool)
    parser.add_argument('--save_dir', dest='save_dir',
                      help='directory to save trained models',
                      default=True, type=bool)
    
    args = parser.parse_args()
    
    return args

if __name__ == '__main__':
    args = parse_args()

    if not os.path.exists(args.save_dir):
        os.makedirs(args.save_dir)

    mnist = MNIST(bs=args.bs)
     # Variables
    inputs = torch.FloatTensor(1)
    labels = torch.FloatTensor(1)
    eye = Variable(torch.eye(mnist.num_classes))
    inputs = Variable(inputs)
    labels = Variable(labels)

    # Model
    model = CapsNet(use_cuda=args.use_cuda)

    # cuda
    if args.use_cuda:
        inputs = inputs.cuda()
        labels = labels.cuda()
        model = model.cuda()
        eye = eye.cuda()

    params = []

    for key, value in dict(model.named_parameters()).items():
        if value.requires_grad:
            params += [{'params':[value],'lr':args.lr}]

    # optimizer
    if args.optimizer == "adam":
        optimizer = torch.optim.Adam(model.parameters())
    elif args.optimizer == "sgd":
        optimizer = torch.optim.SGD(params)

    criterion = CapsLoss()

    print('Training started!')

    for epoch in range(args.num_epochs):
        start = time.time()

        # train
        model.train()
        correct = 0
        train_loss = 0
        for batch_idx, (imgs, targets) in enumerate(mnist.train_dataloader):
            if imgs.size(0) != args.bs:
                continue
                
            targets = eye.cpu().data.index_select(dim=0, index=targets)
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
            if batch_idx % args.disp_interval == 0:
                end = time.time()
                print("[epoch %2d][iter %4d] loss: %.4f, acc: %.4f%% (%d/%d)" \
                                % (epoch, batch_idx, train_loss/(batch_idx+1), 100.*correct/args.bs, correct, args.bs))

        save_name = os.path.join(args.save_dir, '{}_{}.pth'.format(project_id, epoch))
        if args.save_epoch > 0 and batch_idx % args.save_epoch == 0:
            torch.save({
              'epoch': epoch,
            }, save_name)

        # val
        if epoch % args.val_epoch == 0:
            print('Validating...')
            correct = 0
            total = 0
            model.eval()
            for batch_idx, (imgs, targets) in enumerate(mnist.eval_dataloader):
                if imgs.size(0) != args.bs:
                    continue
                targets = eye.cpu().data.index_select(dim=0, index=targets)
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
