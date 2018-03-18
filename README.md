# pyCapsNet

[![License][license]][license-url]

Pytorch, NumPy and CuPy implementations of Capsule Networks (CapsNet), based on the paper [Sabour, Sara, Nicholas Frosst, and Geoffrey E. Hinton. "Dynamic routing between capsules." Advances in Neural Information Processing Systems. 2017.] 

## Requirements

* Python 3

PyTorch Implementation:
* PyTorch 
  * Tested with PyTorch 0.3.0.post4
* CUDA 8 (if using CUDA)

CuPy Implementation:
* CuPy 2.0.0
* CUDA 8

## Motivation: Why another CapsNet implementation?
There are many great implementations of Capsule Networks [with PyTorch], [TensorFlow] and [Keras], so why do we need another one? This project actually provides three implementations of CapsNet: PyTorch, NumPy and CuPy. For the PyTorch version, I implemented CapsNet for performance check and visualizations; for the NumPy and CuPy ones, I aim to get a deeper understanding of the structure and the gradient flow of CapsNet. The computation graph that I used for this implementation is provided later in this document.

The purpose of this project is not to shoot for better performance or optimizing the speed, but to offer a better understanding of CapsNet implementation-wise. Reading the paper thoroughly is a must, but it is easy to get confused when it comes to real implementation. I will provide my own understanding in CapsNet and implementation walkthrough in this document.

This [video] really helped a lot for me to understand CapsNet. Take a minute and check it out.

## To Run
For NumPy and CuPy implementations, change into the corresponding directories, and run
```
python3 main.py --bs=100, --lr=1e-2, --opt='adam', '--disp'=1, --num_epochs=100, --val_epoch=1
```
For the PyTorch implementation, run
```
python3 main.py --bs=100, --lr=1e-2, --opt='adam', '--disp'=1, --num_epochs=100, --val_epoch=1, use_cuda=True
```
To visualize the reconstructed data, run the jupyter notebook in PyTorch/Visualization.ipnb.

## Capsule Networks: Key Points
A capsule is a neuron that outputs activity vectors to represent the instantiation parameters of certain entities. The magnitude of the activation vector corresponds to the probability that such entity exists, and the orientation represents the instantiation parameters. The [paper] proposes a multi-layer capsule network for different image classification tasks, and achieved state-of-the-art performance on the MNIST dataset.

### Activity Vectors
Unlike the neurons in most neural networks, the capsules in this architecture outputs an activity vector for each input. The paper introduces a nonlinear "squashing" function for the activity vector ![img](http://latex.codecogs.com/svg.latex?%5Ctextbf%7B%5Ctextit%7Bs%7D%7D):

![img](http://latex.codecogs.com/svg.latex?%5Ctext%7Bsquash%7D%28%5Ctextit%7B%5Ctextbf%7Bs%7D%7D%29%3D%5Cfrac%7B%7C%7C%5Ctextit%7B%5Ctextbf%7Bs%7D%7D%5E2%7C%7C%7D%7B1%2B%7C%7C%5Ctextit%7B%5Ctextbf%7Bs%7D%7D%5E2%7C%7C%7D%5Cfrac%7B%5Ctextit%7B%5Ctextbf%7Bs%7D%7D%7D%7B%7C%7C%5Ctextit%7B%5Ctextbf%7Bs%7D%7D%7C%7C%7D)

### Dynamic Routing Between Capsules
In this paper, the authors replaces the conventional max-pooling layer with dynamic routing, iteratively refining the coupling coefficient to route the prediction made by the last layer to the next one. Such routing is achieved by "routing-by-agreement", where a capsule prefers to route its outputs to the capsules in the next layer whose output has a greater dot product with its own output. A result can be, for example, that the "5" capsule would receive features that agrees with "5". In the paper, the authors investigated this iterative refinement of routing coefficients, and found out that the number of iterations for finding the coefficients can indeed help achieve a lower loss and better stability. However, this sequential and iterative structure can make CapsNet very slow. In this project, I followed the paper and set the number of routing iterations to 3.

### The Architecture
<div style="text-align:center"><img src='https://github.com/xanderchf/pyCapsNet/blob/master/capsnet.png' width=700></div>

The Capsule Network in the figure above consists of three parts: a convolutional layer (Conv1) and two capsule layers (PrimaryCaps and DigitCaps, which I will explain later). The DigitCaps layer yields a 16-dimensional vector for each of the 10 classes, and the L2 norm of these vectors becomes the class score. The decoder in the figure below consumes these vectors and tries to reconstruct the image. The final loss is the combination of the score loss ("margin loss" in the paper) and the reconstruction loss.

<div style="text-align:center"><img src="https://github.com/xanderchf/pyCapsNet/blob/master/decoder.png" width=500></div>

### Detailed Walkthrough

* Conv1:
  * Input size `(N, C=1, H=28, W=28)`
  * `in_channel=1, out_channel=256, kernel_size=(9,9), stride=1, padding=0`
  * This convolution yields output size `(N, 256, 20, 20)`

* PrimaryCaps:
  * Input size `(N, C=256, H=20, W=20)`
  * `ndim=8` Convolution kernels, each with `in_channel=256, out_channel=32, kernel_size=(9,9), stride=2, padding=0`
  * Each convolution yields output size `(N, 32, 6, 6)`; linearize for each batch and concatenate the output of each convolution and feed into DigitCaps.
  
* DigitCaps:
  * Input size `(N, ncaps_prev=32*6*6, ndim_prev=8)`
  * For convenience, each tensor involved in this layer is reshaped into 5 dimensions, corresponding to the dimensions of the weight, which is of size `(1, ncaps_prev=32*6*6, ncaps=10, ndim=16, ndim_prev=8)`. Note that the input and the weight has the same dimensions in `ncaps_prev` and `ncaps`, and `N` can be handled by broadcasting. Focusing on the last two dimensions, the weight is of size `(16, 8)` and the input is of size `(8, 1)`, therefore the output size gives `(16, 1)` in these two dimensions.
  
  * Outputs size `(N, 1, 10, 16, 1)`. Getting the L2 norm across the 3rd dimension to get scores for each class. This tensor is fed into the decoder to get reconstructions.
  
* Decoder:
  * Input size `(N, 10, 16)`, 
  * Three linear layers, sizes `(16*10, 512)`, `(512, 1024)` and `(1024, 28*28)`. The first two are followed with ReLU and the last layer is followed by a Sigmoid layer.
  * Outputs `(N, 28*28)`, i.e. the reconstruction.
  
## Computation Graph for DigitCaps
<div style="text-align:center"><img src='https://github.com/xanderchf/pyCapsNet/blob/master/compgraph_digitcaps.png' width=600></div>
This computation graph was originally used for a better understanding of the gradient flow. I redrew the graph for DigitCaps in SVG to provide a clear illustration for people interested in implementing this part. If you want to implement backpropagation of DigitCaps, mind the accumulated gradient from each routing iteration.

## Results
I achieved 99.41% validation accuracy at epoch 22 with the PyTorch implementation, which is close to the number reported on the paper. The CuPy implementation can quickly converge to 90%+, but overall trains much slower than the PyTorch version (still training to get the performance). The NumPy implementation is trained purely on CPU; Though I used multiprocessing in the network, it much slower than the GPU implementations. The reconstructed images are given below.

<div style="text-align:center"><img src=https://github.com/xanderchf/pyCapsNet/blob/master/reconst.jpeg width=800></div>

<!-- Markdown link & img dfn's -->
[license]: https://img.shields.io/github/license/mashape/apistatus.svg
[license-url]: https://github.com/xanderchf/pyCapsNet/blob/master/LICENSE
[Sabour, Sara, Nicholas Frosst, and Geoffrey E. Hinton. "Dynamic routing between capsules." Advances in Neural Information Processing Systems. 2017.]: https://arxiv.org/abs/1710.09829
[paper]: https://arxiv.org/abs/1710.09829
[with PyTorch]: https://github.com/gram-ai/capsule-networks
[TensorFlow]: https://github.com/ageron/handson-ml
[Keras]: https://github.com/XifengGuo/CapsNet-Keras
[video]: https://www.youtube.com/watch?v=2Kawrd5szHE
