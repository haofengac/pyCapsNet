from modules import *
import time, os
import cupy as cp
from mnist import MNIST
from modules import CapsNet, CapsLoss
from optim import AdamOptimizer
            
bs = 100
lr = 1e-2
use_cuda = True
opt = 'adam'
num_epochs = 100
disp_interval = 10
num_classes = 10
    
if __name__ == '__main__':

    mnist = MNIST(bs=bs, shuffle=True)
    eye = cp.eye(num_classes)
    model = CapsNet()

    criterion = CapsLoss()
    if opt == 'adam':
        optimizer = AdamOptimizer(lr=lr)
        
    print('Training started!')

    for epoch in range(num_epochs):
        start = time.time()

        # train
        correct = 0
        for batch_idx, (imgs, targets) in enumerate(mnist.train_dataset):
            optimizer.step()
            if imgs.shape[0] != bs:
                continue

            targets = cp.eye(num_classes)[targets]
            scores, reconst = model(imgs)
            loss, grad = criterion(scores, targets, reconst, imgs)
            model.backward(grad, optimizer)

            classes = cp.argmax(scores, axis=1)
            predicted = eye[cp.squeeze(classes), :]

            predicted_idx = cp.argmax(predicted, 1)
            label_idx = cp.argmax(targets, 1)
            correct = cp.sum(predicted_idx == label_idx)

            # info
            if batch_idx % disp_interval == 0:
                end = time.time()
                print("[epoch %2d][iter %4d] loss: %.4f, acc: %.4f%% (%d/%d)" \
                                % (epoch, batch_idx, loss, 100.*correct/bs, correct, bs))

        # val
        if epoch % val_epoch == 0:
            print('Validating...')
            correct = 0
            total = 0

            for batch_idx, (imgs, targets) in enumerate(mnist.eval_dataset):
                if imgs.shape[0] != bs:
                    continue

                targets = cp.eye(num_classes)[targets]
                scores, reconst = model(imgs)
                loss, grad = criterion(scores, targets, reconst, imgs)
                model.backward(grad, optimizer)

                classes = cp.argmax(scores, axis=1)
                predicted = eye[cp.squeeze(classes, axis=1), :]

                predicted_idx = cp.argmax(predicted, 1)
                label_idx = cp.argmax(targets, 1)
                correct += cp.sum(predicted_idx == label_idx)
                total += targets.shape[0]

            print("[epoch %2d] val acc: %.4f%% (%d/%d)" \
                                    % (epoch, 100.*correct/total, correct, total))
