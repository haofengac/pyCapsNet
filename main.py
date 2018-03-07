from modules import *
import numpy as np
import time, os
from torchvision import transforms, datasets

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
    
bs = 4
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

if not os.path.exists(save_dir):
    os.makedirs(save_dir)

print('Training started!')

for epoch in range(num_epochs):
    start = time.time()
    
    # train
    correct = 0
    train_loss = 0
    for batch_idx, (imgs, targets) in enumerate(mnist.train_dataloader):
        optimizer.step()
        if imgs.size(0) != bs:
            continue
        
        imgs = imgs.numpy()
        targets = targets.numpy()
        targets = np.eye(num_classes)[0, targets]
        
        scores, reconst = model(inputs)
        
        loss, grad = criterion(scores, labels, reconst, inputs)
        model.backward(grad, optimizer)

        scores, classes = scores.max(dim=1)
        predicted = eye[np.squeeze(classes, axis=1), :]
        
        predicted_idx = np.argmax(predicted, 1)
        label_idx = np.argmax(targets, 1)
        correct = np.sum(predicted_idx == label_idx)
        
        # info
        if batch_idx % disp_interval == 0:
            end = time.time()
            print("[epoch %2d][iter %4d] loss: %.4f, acc: %.4f%% (%d/%d)" \
                            % (epoch, batch_idx, train_loss/(batch_idx+1), 100.*correct/bs, correct, bs))

    # val
    if epoch % val_epoch == 0:
        print('Validating...')
        correct = 0
        total = 0

        for batch_idx, (imgs, targets) in enumerate(mnist.eval_dataloader):
            if imgs.size(0) != bs:
                continue
            
            imgs = imgs.numpy()
            targets = targets.numpy()
            targets = np.eye(num_classes)[0, targets]
            
            outputs, reconst = model(inputs)
            scores = np.sqrt(np.sum(outputs ** 2, axis=2))
            scores, classes = scores.max(dim=1)
            predicted = eye[np.squeeze(classes, axis=1), :]
        
            predicted_idx = np.argmax(predicted, 1)
            label_idx = np.argmax(targets, 1)
            correct = np.sum(predicted_idx == label_idx)
            total += targets.size(0)
        print("[epoch %2d] val acc: %.4f%% (%d/%d)" \
                                % (epoch, 100.*correct/total, correct, total))
        