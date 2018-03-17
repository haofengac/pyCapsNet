import time, os
import numpy as np
import cupy as cp
from urllib import request
import gzip
import pickle


class MNIST:
    def __init__(self, path='data', bs=1, shuffle=False):
        self.filename = [
        ["training_images","train-images-idx3-ubyte.gz"],
        ["test_images","t10k-images-idx3-ubyte.gz"],
        ["training_labels","train-labels-idx1-ubyte.gz"],
        ["test_labels","t10k-labels-idx1-ubyte.gz"]
        ]
        self.mean = 0.1307
        self.std = 0.3081
        self.num_classes = 10
        self.bs = bs
        self.path = path
        if not os.path.exists(self.path):
            os.mkdir(self.path)
        if not os.path.exists(self.path+'/mnist.pkl'):
            self.download_mnist()
        self.load(shuffle=shuffle)
        print('Loading complete.')
        
    def download_mnist(self):
        base_url = "http://yann.lecun.com/exdb/mnist/"
        for name in self.filename:
            print("Downloading "+name[1]+"...")
            request.urlretrieve(base_url+name[1], self.path+'/'+name[1])
        print("Download complete.")
        self.save_mnist()

    def save_mnist(self):
        mnist = {}
        for name in self.filename[:2]:
            with gzip.open(self.path+'/'+name[1], 'rb') as f:
                mnist[name[0]] = ((np.frombuffer(f.read(), np.uint8, offset=16).reshape(-1,28*28))/255.-self.mean)/self.std
        for name in self.filename[-2:]:
            with gzip.open(self.path+'/'+name[1], 'rb') as f:
                mnist[name[0]] = np.frombuffer(f.read(), np.uint8, offset=8)
        with open(self.path+'/'+"mnist.pkl", 'wb') as f:
            pickle.dump(mnist,f)
        print("Save complete.")

    def chunks(self, l):
        for i in range(0, len(l), self.bs):
            yield l[i:i + self.bs]

    def load(self, shuffle=False):
        with open(self.path+"/mnist.pkl",'rb') as f:
            mnist = pickle.load(f)
        if shuffle:
            n = mnist['training_images'].shape[0]
            idxs = np.arange(n)
            np.random.shuffle(idxs)
            mnist['training_images'] = mnist['training_images'].reshape((-1,1,28,28))
            mnist['training_images'] = list(self.chunks(mnist['training_images'][idxs]))
            mnist['training_labels'] = list(self.chunks(mnist['training_labels'][idxs]))
            self.train_dataset = zip(cp.array(mnist['training_images']), cp.array(mnist['training_labels']))
            
            n = mnist['test_images'].shape[0]
            idxs = np.arange(n)
            np.random.shuffle(idxs)
            mnist['test_images'] = mnist['test_images'].reshape((-1,1,28,28))
            mnist['test_images'] = list(self.chunks(mnist['test_images'][idxs]))
            mnist['test_labels'] = list(self.chunks(mnist['test_labels'][idxs]))
            self.eval_dataset = zip(cp.array(mnist['test_images']), cp.array(mnist['test_labels']))

            