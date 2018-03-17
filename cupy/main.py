from modules import *
import time, os, argparse
import cupy as cp
from mnist import MNIST
from modules import CapsNet, CapsLoss
from optim import AdamOptimizer

    
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
    parser.add_argument('--opt', dest='opt',
                      help='optimizer',
                      default='adam', type=str)
    parser.add_argument('--disp', dest='disp_interval',
                      help='interval to display training loss',
                      default='10', type=int)
    parser.add_argument('--num_epochs', dest='num_epochs',
                      help='interval to display training loss',
                      default='100', type=int)
    
    args = parser.parse_args()
    
    return args

if __name__ == '__main__':
    
    args = parse_args()
    
    mnist = MNIST(bs=args.bs, shuffle=True)
    eye = cp.eye(mnist.num_classes)
    model = CapsNet()

    criterion = CapsLoss()
    if args.opt == 'adam':
        optimizer = AdamOptimizer(lr=args.lr)
        
    print('Training started!')

    for epoch in range(args.num_epochs):
        start = time.time()

        # train
        correct = 0
        for batch_idx, (imgs, targets) in enumerate(mnist.train_dataset):
            optimizer.step()
            if imgs.shape[0] != args.bs:
                continue

            targets = eye[targets]
            scores, reconst = model(imgs)
            loss, grad = criterion(scores, targets, reconst, imgs)
            model.backward(grad, optimizer)

            classes = cp.argmax(scores, axis=1)
            predicted = eye[cp.squeeze(classes), :]

            predicted_idx = cp.argmax(predicted, 1)
            label_idx = cp.argmax(targets, 1)
            correct = cp.sum(predicted_idx == label_idx)

            # info
            if batch_idx % args.disp_interval == 0:
                end = time.time()
                print("[epoch %2d][iter %4d] loss: %.4f, acc: %.4f%% (%d/%d)" \
                                % (epoch, batch_idx, loss, 100.*correct/args.bs, correct, args.bs))

        # val
        if epoch % val_epoch == 0:
            print('Validating...')
            correct = 0
            total = 0

            for batch_idx, (imgs, targets) in enumerate(mnist.eval_dataset):
                if imgs.shape[0] != args.bs:
                    continue

                targets = eye[targets]
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
