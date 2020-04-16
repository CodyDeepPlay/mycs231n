# -*- coding: utf-8 -*-
"""
Created on Sun Apr 12 20:18:07 2020

@author: Mingming

Q6, Pytorch
"""

#%%
###############################################################################
#                 Part I: Preparation for PyTorch                             #
#                                                                             #
###############################################################################


import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.utils.data import sampler

import torchvision.datasets as dset
import torchvision.transforms as T
import torch.nn.functional as F  # useful stateless functions
import numpy as np

#%
NUM_TRAIN = 49000

# The torchvision.transforms package provides tools for preprocessing data
# and for performing data augmentation; here we set up a transform to
# preprocess the data by subtracting the mean RGB value and dividing by the
# standard deviation of each RGB value; we've hardcoded the mean and std.
transform = T.Compose([
                T.ToTensor(),
                T.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
            ])

# We set up a Dataset object for each split (train / val / test); Datasets load
# training examples one at a time, so we wrap each Dataset in a DataLoader which
# iterates through the Dataset and forms minibatches. We divide the CIFAR-10
# training set into train and val sets by passing a Sampler object to the
# DataLoader telling how it should sample from the underlying Dataset.
cifar10_train = dset.CIFAR10('./cs231n/datasets', train=True, download=True,
                             transform=transform)
loader_train = DataLoader(cifar10_train, batch_size=64, 
                          sampler=sampler.SubsetRandomSampler(range(NUM_TRAIN)))

cifar10_val = dset.CIFAR10('./cs231n/datasets', train=True, download=True,
                           transform=transform)
loader_val = DataLoader(cifar10_val, batch_size=64, 
                        sampler=sampler.SubsetRandomSampler(range(NUM_TRAIN, 50000)))

cifar10_test = dset.CIFAR10('./cs231n/datasets', train=False, download=True, 
                            transform=transform)
loader_test = DataLoader(cifar10_test, batch_size=64)

#%
USE_GPU = False

dtype = torch.float32 # we will be using float throughout this tutorial

if USE_GPU and torch.cuda.is_available():
    device = torch.device('cuda')
else:
    device = torch.device('cpu')

# Constant to control how frequently we print train loss
print_every = 100

print('using device:', device)





#%%
###############################################################################
#                                                                             #
#                      Part II: Barebones PyTorch                             #
#                                                                             #
###############################################################################


def flatten(x):
    N = x.shape[0] # read in N, C, H, W
    return x.view(N, -1)  # "flatten" the C * H * W values into a single vector per image

def test_flatten():
    x = torch.arange(12).view(2, 1, 3, 2)
    print('Before flattening: ', x)
    print('After flattening: ', flatten(x))

test_flatten()

#%%
##--------------- Barebones PyTorch: Two-Layer Network ----------------------##



def two_layer_fc(x, params):
    """
    A fully-connected neural networks; the architecture is:
    NN is fully connected -> ReLU -> fully connected layer.
    Note that this function only defines the forward pass; 
    PyTorch will take care of the backward pass for us.
    
    The input to the network will be a minibatch of data, of shape
    (N, d1, ..., dM) where d1 * ... * dM = D. The hidden layer will have H units,
    and the output layer will produce scores for C classes.
    
    Inputs:
    - x: A PyTorch Tensor of shape (N, d1, ..., dM) giving a minibatch of
      input data.
    - params: A list [w1, w2] of PyTorch Tensors giving weights for the network;
      w1 has shape (D, H) and w2 has shape (H, C).
    
    Returns:
    - scores: A PyTorch Tensor of shape (N, C) giving classification scores for
      the input data x.
    """
    # first we flatten the image
    x = flatten(x)  # shape: [batch_size, C x H x W]
    
    w1, w2, fc_b = params
    
    # Forward pass: compute predicted y using operations on Tensors. Since w1 and
    # w2 have requires_grad=True, operations involving these Tensors will cause
    # PyTorch to build a computational graph, allowing automatic computation of
    # gradients. Since we are no longer implementing the backward pass by hand we
    # don't need to keep references to intermediate values.
    # you can also use `.clamp(min=0)`, equivalent to F.relu()
    x = F.relu(x.mm(w1))   # x.mm(w1), the matrix product between 'x' and 'w1'
    x = x.mm(w2) + fc_b    # here is no 'softmax activation' here, because pytorch PyTorch's cross entropy loss performs a softmax, which will happen in the training loop
    return x
    

def two_layer_fc_test():
    hidden_layer_size = 42
    x = torch.zeros((64, 50), dtype=dtype)  # minibatch size 64, feature dimension 50
    w1 = torch.zeros((50, hidden_layer_size), dtype=dtype)
    w2 = torch.zeros((hidden_layer_size, 10), dtype=dtype)
    fc_b = torch.zeros( (10,) , dtype=dtype )
    scores = two_layer_fc(x, [w1, w2, fc_b])
    print(scores.size())  # you should see [64, 10]

two_layer_fc_test()

#%%
##--------------- Barebones PyTorch: Three-Layer ConvNet --------------------##


def three_layer_convnet(x, params):
    """
    Performs the forward pass of a three-layer convolutional network with the
    architecture defined above.

    Inputs:
    - x: A PyTorch Tensor of shape (N, 3, H, W) giving a minibatch of images
    - params: A list of PyTorch Tensors giving the weights and biases for the
      network; should contain the following:
      - conv_w1: PyTorch Tensor of shape (channel_1, 3, KH1, KW1) giving weights
        for the first convolutional layer
      - conv_b1: PyTorch Tensor of shape (channel_1,) giving biases for the first
        convolutional layer
      - conv_w2: PyTorch Tensor of shape (channel_2, channel_1, KH2, KW2) giving
        weights for the second convolutional layer
      - conv_b2: PyTorch Tensor of shape (channel_2,) giving biases for the second
        convolutional layer
      - fc_w: PyTorch Tensor giving weights for the fully-connected layer. Can you
        figure out what the shape should be?
      - fc_b: PyTorch Tensor giving biases for the fully-connected layer. Can you
        figure out what the shape should be?
    
    Returns:
    - scores: PyTorch Tensor of shape (N, C) giving classification scores for x
    """
    conv_w1, conv_b1, conv_w2, conv_b2, fc_w, fc_b = params
    scores = None
    ################################################################################
    # TODO: Implement the forward pass for the three-layer ConvNet.                #
    ################################################################################
    # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
    (N,         C,         H,   W)   = x.size()
    (channel_1, C,         KH1, KW1) = conv_w1.size()
    (channel_2, channel_1, KH2, KW2) = conv_w2.size()
    
    # input images size (N, H, W, 3),  conv_w1 size (KH1, KW1, 3, channel_1), here 3 is the image channels
    # new_H1 = 1 + (H + 2 * pad - KH1) / stride, here pad is 2. 
    # new_W1 = 1 + (W + 2 * pad - KW1) / stride, here pad is 2. 
    
    
    conv_layer1_structure = nn.Conv2d(in_channels  = C, 
                                      out_channels = channel_1, 
                                      kernel_size  = (KH1, KW1),
                                      stride       = 1,
                                      padding      = 2,
                                      bias         = True)
    
    
    # delete the built-in weights and bias from the Conv2d function, and assign my custom weights and bias
    # this step is critical, so later when calling loss.backward(), it can assign the gradients to my weights and bias parameters
    # otherwise, it will not update the gradient in params and then will not work with train_part2()
    del conv_layer1_structure.weight, conv_layer1_structure.bias  
    conv_layer1_structure.weight = conv_w1   # add the custom initialized parameter to this conv2d layer
    conv_layer1_structure.bias   = conv_b1
    
    conv_layer1 = conv_layer1_structure(x)
    
    # ------- so conv_layer1 size is (N, new_H1, new_W1, channel_1)
    relu_layer1 = nn.ReLU()(conv_layer1)

    # here input size (N, new_H1, new_W1, channel_1),  conv_w2 size (KH2, KW2, channel_1, channel_2), 
    # new_H2 = 1 + (new_H1 + 2 * pad - KH2) / stride, here pad is 1. 
    # new_W2 = 1 + (new_W1 + 2 * pad - KW2) / stride, here pad is 1. 
    conv_layer2_structure = nn.Conv2d(in_channels  = channel_1, 
                                      out_channels = channel_2, 
                                      kernel_size  = (KH2, KW2),
                                      stride       = 1,
                                      padding      = 1,
                                      bias         = True)
    
    # delete the built-in weights and bias from the Conv2d function, and assign my custom weights and bias
    del conv_layer2_structure.weight, conv_layer2_structure.bias  
    conv_layer2_structure.weight  = conv_w2 
    conv_layer2_structure.bias    = conv_b2
    conv_layer2 = conv_layer2_structure(relu_layer1)  # pass the previous layer into this conv2d layer with custom weights
    # ------- so conv_layer1 size is (N, new_H2, new_W2, channel_2)


    relu_layer2 = nn.ReLU()(conv_layer2)
    
    flat_layer  = nn.Flatten()(relu_layer2)
    # fc_w has a size of (D=new_H2*new_W2*channel_2, 10), here 10 means 10 classes
    # fc_b has a size of (10,),
    # Compute scores of shape (N, 10)
    scores = flat_layer.mm(fc_w) + fc_b
    

    params = [conv_layer1_structure.weight, conv_layer1_structure.bias , 
              conv_layer2_structure.weight , conv_layer2_structure.bias, 
              fc_w, fc_b]
    
    pass

    # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
    ################################################################################
    #                                 END OF YOUR CODE                             #
    ################################################################################
    return scores



def three_layer_convnet_test():
    x = torch.zeros((64, 3, 32, 32), dtype=dtype)  # minibatch size 64, image size [3, 32, 32]

    conv_w1 = torch.zeros((6, 3, 5, 5), dtype=dtype)  # [out_channel, in_channel, kernel_H, kernel_W]
    conv_b1 = torch.zeros((6,))  # out_channel
    conv_w2 = torch.zeros((9, 6, 3, 3), dtype=dtype)  # [out_channel, in_channel, kernel_H, kernel_W]
    conv_b2 = torch.zeros((9,))  # out_channel

    # you must calculate the shape of the tensor after two conv layers, before the fully-connected layer
    fc_w = torch.zeros((9 * 32 * 32, 10))
    fc_b = torch.zeros(10)

    scores = three_layer_convnet(x, [conv_w1, conv_b1, conv_w2, conv_b2, fc_w, fc_b])
    print(scores.size())  # you should see [64, 10]
three_layer_convnet_test()



#%%
##------------------ Barebones PyTorch: Initialization ----------------------##


def random_weight(shape):
    """
    Create random Tensors for weights; setting requires_grad=True means that we
    want to compute gradients for these Tensors during the backward pass.
    We use Kaiming normalization: sqrt(2 / fan_in)
    """
    if len(shape) == 2:  # FC weight
        fan_in = shape[0]
    else:
        fan_in = np.prod(shape[1:]) # conv weight [out_channel, in_channel, kH, kW]
    # randn is standard normal distribution generator. 
    w = torch.randn(shape, device=device, dtype=dtype) * np.sqrt(2. / fan_in)
    w.requires_grad = True
    return w

def zero_weight(shape):
    return torch.zeros(shape, device=device, dtype=dtype, requires_grad=True)

# create a weight of shape [3 x 5]
# you should see the type `torch.cuda.FloatTensor` if you use GPU. 
# Otherwise it should be `torch.FloatTensor`
random_weight((3, 5))



##------------------ Barebones PyTorch: Check Accuracy ----------------------##

def check_accuracy_part2(loader, model_fn, params):
    """
    Check the accuracy of a classification model.
    
    Inputs:
    - loader: A DataLoader for the data split we want to check
    - model_fn: A function that performs the forward pass of the model,
      with the signature scores = model_fn(x, params)
    - params: List of PyTorch Tensors giving parameters of the model
    
    Returns: Nothing, but prints the accuracy of the model
    """
    split = 'val' if loader.dataset.train else 'test'
    print('Checking accuracy on the %s set' % split)
    num_correct, num_samples = 0, 0
    # To prevent a graph from being built we scope our computation under a torch.no_grad() context manager.
    with torch.no_grad():
        for x, y in loader:
            x = x.to(device=device, dtype=dtype)  # move to device, e.g. GPU
            y = y.to(device=device, dtype=torch.int64)
            scores = model_fn(x, params)
            _, preds = scores.max(1)
            num_correct += (preds == y).sum()
            num_samples += preds.size(0)
        acc = float(num_correct) / num_samples
        print('Got %d / %d correct (%.2f%%)' % (num_correct, num_samples, 100 * acc))


##------------------ BareBones PyTorch: Training Loop ----------------------##
'''
We can now set up a basic training loop to train our network. We will train the 
model using stochastic gradient descent without momentum. We will use torch.functional.cross_entropy 
to compute the loss; you can read about it here.
The training loop takes as input the neural network function, a list of initialized parameters 
([w1, w2] in our example), and learning rate.
'''

def train_part2(model_fn, params, learning_rate):
    """
    Train a model on CIFAR-10.
    
    Inputs:
    - model_fn: A Python function that performs the forward pass of the model.
      It should have the signature scores = model_fn(x, params) where x is a
      PyTorch Tensor of image data, params is a list of PyTorch Tensors giving
      model weights, and scores is a PyTorch Tensor of shape (N, C) giving
      scores for the elements in x.
    - params: List of PyTorch Tensors giving weights for the model
    - learning_rate: Python scalar giving the learning rate to use for SGD
    
    Returns: Nothing
    """
    for t, (x, y) in enumerate(loader_train):
        # Move the data to the proper device (GPU or CPU)
        x = x.to(device=device, dtype=dtype)
        y = y.to(device=device, dtype=torch.long)

        # Forward pass: compute scores and loss
        scores = model_fn(x, params)
        loss = F.cross_entropy(scores, y)

        # Backward pass: PyTorch figures out which Tensors in the computational
        # graph has requires_grad=True and uses backpropagation to compute the
        # gradient of the loss with respect to these Tensors, and stores the
        # gradients in the .grad attribute of each Tensor.
        loss.backward()

        # Update parameters. We don't want to backpropagate through the
        # parameter updates, so we scope the updates under a torch.no_grad()
        # context manager to prevent a computational graph from being built.
        with torch.no_grad():  
            for w in params:
                w -= learning_rate * w.grad
                
                # this is already calculated after a mini batch of data
                # Manually zero the gradients after running the backward pass
                w.grad.zero_()

        if t % print_every == 0:
            print('Iteration %d, loss = %.4f' % (t, loss.item()))
            check_accuracy_part2(loader_val, model_fn, params)
            print()


##-------------- BareBones PyTorch: Train a Two-Layer Network ---------------##

hidden_layer_size = 4000
learning_rate = 1e-2

w1 = random_weight( (3 * 32 * 32, hidden_layer_size) )
w2 = random_weight( (hidden_layer_size, 10) )
fc_b = random_weight( (10,) )
params=[w1, w2, fc_b]
model_fn = two_layer_fc

train_part2(model_fn, params, learning_rate)



#%% 

##------------------ BareBones PyTorch: Train a ConvNet ---------------------##

learning_rate = 3e-3

channel_1 = 32
channel_2 = 16

conv_w1 = None
conv_b1 = None
conv_w2 = None
conv_b2 = None
fc_w = None
fc_b = None

################################################################################
# TODO: Initialize the parameters of a three-layer ConvNet.                    #
################################################################################
# *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

pass

conv_w1 = random_weight( (channel_1, 3, 5, 5)  )         # [out_channel, in_channel, kernel_H, kernel_W]
conv_b1 = random_weight( (channel_1,) )                  # out_channel
conv_w2 = random_weight( (channel_2, channel_1, 3, 3) )  # [out_channel, in_channel, kernel_H, kernel_W]
conv_b2 = random_weight( (channel_2,) )                  # out_channel

# you must calculate the shape of the tensor after two conv layers, before the fully-connected layer
fc_w = random_weight( (channel_2 * 32 * 32, 10) )
fc_b = random_weight( (10,) )

# *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
################################################################################
#                                 END OF YOUR CODE                             #
################################################################################

params = [conv_w1, conv_b1, conv_w2, conv_b2, fc_w, fc_b]
model_fn = three_layer_convnet

train_part2(model_fn, params, learning_rate)



#%%

###############################################################################
#                                                                             #
#                      Part III: PyTorch Module API                           #
#                                                                             #
###############################################################################

'''
Barebone PyTorch requires that we track all the parameter tensors by hand. 
This is fine for small networks with a few tensors, but it would be extremely inconvenient 
and error-prone to track tens or hundreds of tensors in larger networks.
PyTorch provides the nn.Module API for you to define arbitrary network architectures, 
while tracking every learnable parameters for you. In Part II, we implemented SGD ourselves. 
PyTorch also provides the torch.optim package that implements all the common optimizers, such as RMSProp, 
Adagrad, and Adam. It even supports approximate second-order methods like L-BFGS! You can refer to the doc 
for the exact specifications of each optimizer
'''

##------------------- Module API: Two-Layer Network  ---------------------##

class TwoLayerFC(nn.Module):
    def __init__(self, input_size, hidden_size, num_classes):
        super().__init__()
        # assign layer objects to class attributes
        self.fc1 = nn.Linear(input_size, hidden_size)
        # nn.init package contains convenient initialization methods
        # http://pytorch.org/docs/master/nn.html#torch-nn-init 
        nn.init.kaiming_normal_(self.fc1.weight)
        self.fc2 = nn.Linear(hidden_size, num_classes)
        nn.init.kaiming_normal_(self.fc2.weight)
    
    def forward(self, x):
        # forward always defines connectivity
        x = flatten(x)
        fc1_layer = self.fc1(x)
        relu1_layer = F.relu(fc1_layer)
        scores = self.fc2(relu1_layer)
        return scores

def test_TwoLayerFC():
    input_size = 50
    x = torch.zeros((64, input_size), dtype=dtype)  # minibatch size 64, feature dimension 50
    model = TwoLayerFC(input_size, 42, 10)
    scores = model(x)
    print(scores.size())  # you should see [64, 10]
test_TwoLayerFC()


##------------------- Module API: Three-Layer Network  ---------------------##


class ThreeLayerConvNet(nn.Module):
    def __init__(self, in_channel, channel_1, channel_2, num_classes):
        super().__init__()
        ########################################################################
        # TODO: Set up the layers you need for a three-layer ConvNet with the  #
        # architecture defined above.                                          #
        ########################################################################
        # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

        self.conv_layer1 = nn.Conv2d(in_channels  = in_channel, 
                                     out_channels = channel_1, 
                                     kernel_size  = (5, 5),
                                     stride       = 1,
                                     padding      = 2,
                                     bias         = True)
        #nn.init.kaiming_normal_(self.conv_layer1.weight) 
        # no need to initialize the bias, as bias=True will do that. 
    

        
        self.conv_layer2 = nn.Conv2d(in_channels  = channel_1, 
                                     out_channels = channel_2, 
                                     kernel_size  = (3, 3),
                                     stride       = 1,
                                     padding      = 1,
                                     bias         = True)
        #nn.init.kaiming_normal_(self.conv_layer2.weight)
        
        
        self.fc1 = nn.Linear(channel_2 * 32 * 32, num_classes)
        nn.init.kaiming_normal_(self.fc1.weight)
        
        pass
        
        # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
        ########################################################################
        #                          END OF YOUR CODE                            #       
        ########################################################################

    def forward(self, x):
        scores = None
        ########################################################################
        # TODO: Implement the forward function for a 3-layer ConvNet. you      #
        # should use the layers you defined in __init__ and specify the        #
        # connectivity of those layers in forward()                            #
        ########################################################################
        # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
        
        # forward always defines connectivity

        conv1 = self.conv_layer1(x)
        relu1 = F.relu(conv1)
        
        conv2 = self.conv_layer2(relu1)
        relu2 = F.relu(conv2)
        
        flat = flatten(relu2)
        
        scores = self.fc1(flat)
        
        pass
        
        # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
        ########################################################################
        #                             END OF YOUR CODE                         #
        ########################################################################
        return scores


def test_ThreeLayerConvNet():
    x = torch.zeros((64, 3, 32, 32), dtype=dtype)  # minibatch size 64, image size [3, 32, 32]
    model = ThreeLayerConvNet(in_channel=3, channel_1=12, channel_2=8, num_classes=10)
    scores = model(x)
    print(scores.size())  # you should see [64, 10]
test_ThreeLayerConvNet()


##------------------- Module API: Check Accuracy ---------------------##

def check_accuracy_part34(loader, model):
    if loader.dataset.train:
        print('Checking accuracy on validation set')
    else:
        print('Checking accuracy on test set')   
    num_correct = 0
    num_samples = 0
    model.eval()  # set model to evaluation mode
    with torch.no_grad():
        for x, y in loader:
            x = x.to(device=device, dtype=dtype)  # move to device, e.g. GPU
            y = y.to(device=device, dtype=torch.long)
            scores = model(x)
            _, preds = scores.max(1)
            num_correct += (preds == y).sum()
            num_samples += preds.size(0)
        acc = float(num_correct) / num_samples
        print('Got %d / %d correct (%.2f)' % (num_correct, num_samples, 100 * acc))


##------------------- Module API: Training loop---------------------##

def train_part34(model, optimizer, epochs=1):
    """
    Train a model on CIFAR-10 using the PyTorch Module API.
    
    Inputs:
    - model: A PyTorch Module giving the model to train.
    - optimizer: An Optimizer object we will use to train the model
    - epochs: (Optional) A Python integer giving the number of epochs to train for
    
    Returns: Nothing, but prints model accuracies during training.
    """
    model = model.to(device=device)  # move the model parameters to CPU/GPU
    for e in range(epochs):
        for t, (x, y) in enumerate(loader_train):
            model.train()  # put model to training mode
            x = x.to(device=device, dtype=dtype)  # move to device, e.g. GPU
            y = y.to(device=device, dtype=torch.long)

            scores = model(x)
            loss = F.cross_entropy(scores, y)

            # Zero out all of the gradients for the variables which the optimizer
            # will update.
            optimizer.zero_grad()

            # This is the backwards pass: compute the gradient of the loss with
            # respect to each  parameter of the model.
            loss.backward()

            # Actually update the parameters of the model using the gradients
            # computed by the backwards pass.
            optimizer.step()

            if t % print_every == 0:
                print('Iteration %d, loss = %.4f' % (t, loss.item()))
                check_accuracy_part34(loader_val, model)
                print()


##------------------ Module API: Train a Two-Layer Network------------------##

hidden_layer_size = 4000
learning_rate = 1e-2
model = TwoLayerFC(3 * 32 * 32, hidden_layer_size, 10)
optimizer = optim.SGD(model.parameters(), lr=learning_rate)

train_part34(model, optimizer)


##------------------ Module API: Train a Three-Layer Network------------------##

learning_rate = 3e-3
channel_1 = 32
channel_2 = 16

model = None
optimizer = None
################################################################################
# TODO: Instantiate your ThreeLayerConvNet model and a corresponding optimizer #
################################################################################
# *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

model = ThreeLayerConvNet(3, channel_1, channel_2, num_classes=10)
optimizer = optim.SGD(model.parameters(), lr=learning_rate)
pass
# *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
################################################################################
#                                 END OF YOUR CODE                             
################################################################################

train_part34(model, optimizer)

#%%


###############################################################################
#                                                                             #
#                      Part IV:  PyTorch Sequential API                       #
#                                                                             #
###############################################################################


##------------------ Sequential API: Two-Layer Network ------------------##

# We need to wrap `flatten` function in a module in order to stack it
# in nn.Sequential
class Flatten(nn.Module):
    def forward(self, x):
        return flatten(x)

hidden_layer_size = 4000
learning_rate = 1e-2

model = nn.Sequential(
    Flatten(),
    nn.Linear(3 * 32 * 32, hidden_layer_size),
    nn.ReLU(),
    nn.Linear(hidden_layer_size, 10),
)

# you can use Nesterov momentum in optim.SGD
optimizer = optim.SGD(model.parameters(), lr=learning_rate,
                     momentum=0.9, nesterov=True)

train_part34(model, optimizer)


##------------------ Sequential API: Three-Layer ConvNet ------------------##

channel_1 = 32
channel_2 = 16
learning_rate = 1e-2

model = None
optimizer = None

################################################################################
# TODO: Rewrite the 2-layer ConvNet with bias from Part III with the           #
# Sequential API.                                                              #
################################################################################
# *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

# you can use Nesterov momentum in optim.SGD
model = nn.Sequential(
    nn.Conv2d(in_channels  = in_channel, 
              out_channels = channel_1, 
              kernel_size  = (5, 5),
              stride       = 1,
              padding      = 2,
              bias         = True),
    nn.ReLU(),
    
    nn.Conv2d(in_channels  = channel_1, 
              out_channels = channel_2, 
              kernel_size  = (3, 3),
              stride       = 1,
              padding      = 1,   
              bias         = True),
    nn.ReLU(),
    Flatten(),
    nn.Linear(channel_2 * 32 * 32, 10),  # fully connected layer
)


optimizer = optim.SGD(model.parameters(), lr=learning_rate,
                     momentum=0.9, nesterov=True)

pass

# *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
################################################################################
#                                 END OF YOUR CODE                             
################################################################################

train_part34(model, optimizer)















