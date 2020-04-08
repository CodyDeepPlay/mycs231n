# -*- coding: utf-8 -*-
"""
Created on Mon Mar 30 20:44:41 2020

@author: Mingming

Q5 to implement tensor flow in deep learning


Chage the setting for using a different python interpreter
--> Tools
    --> Preferences
        --> Python interpreter
                --> select 'Use the following startup script'
                    Then, paste your python interpreter location

newly changed python interpreter location after createa new virtual environment for tensorflow
C:\\Users\Mingming\Anaconda3\envs\tf_20_env\python.exe 

original default python interpreter location from spyder
C:\\Users\Mingming\Anaconda3\lib/site-packages\spyder\scientific_startup.py

"""
#%%

import os
import tensorflow as tf
import numpy as np
import math
import timeit
import matplotlib.pyplot as plt
#%%
#%matplotlib inline

def load_cifar10(num_training=49000, num_validation=1000, num_test=10000):
    """
    Fetch the CIFAR-10 dataset from the web and perform preprocessing to prepare
    it for the two-layer neural net classifier. These are the same steps as
    we used for the SVM, but condensed to a single function.
    """
    # Load the raw CIFAR-10 dataset and use appropriate data types and shapes
    cifar10 = tf.keras.datasets.cifar10.load_data()    
    (X_train, y_train), (X_test, y_test) = cifar10
    X_train = np.asarray(X_train, dtype=np.float32)
    y_train = np.asarray(y_train, dtype=np.int32).flatten()
    X_test = np.asarray(X_test, dtype=np.float32)
    y_test = np.asarray(y_test, dtype=np.int32).flatten()

    # Subsample the data
    mask = range(num_training, num_training + num_validation)
    X_val = X_train[mask]
    y_val = y_train[mask]
    mask = range(num_training)
    X_train = X_train[mask]
    y_train = y_train[mask]
    mask = range(num_test)
    X_test = X_test[mask]
    y_test = y_test[mask]

    # Normalize the data: subtract the mean pixel and divide by std
    mean_pixel = X_train.mean(axis=(0, 1, 2), keepdims=True)
    std_pixel = X_train.std(axis=(0, 1, 2), keepdims=True)
    X_train = (X_train - mean_pixel) / std_pixel
    X_val = (X_val - mean_pixel) / std_pixel
    X_test = (X_test - mean_pixel) / std_pixel

    return X_train, y_train, X_val, y_val, X_test, y_test

# If there are errors with SSL downloading involving self-signed certificates,
# it may be that your Python version was recently installed on the current machine.
# See: https://github.com/tensorflow/tensorflow/issues/10779
# To fix, run the command: /Applications/Python\ 3.7/Install\ Certificates.command
#   ...replacing paths as necessary.

# Invoke the above function to get our data.
NHW = (0, 1, 2)
X_train, y_train, X_val, y_val, X_test, y_test = load_cifar10()
print('Train data shape: ', X_train.shape)
print('Train labels shape: ', y_train.shape, y_train.dtype)
print('Validation data shape: ', X_val.shape)
print('Validation labels shape: ', y_val.shape)
print('Test data shape: ', X_test.shape)
print('Test labels shape: ', y_test.shape)


#%%
class Dataset(object):
    def __init__(self, X, y, batch_size, shuffle=False):
        """
        Construct a Dataset object to iterate over data X and labels y
        
        Inputs:
        - X: Numpy array of data, of any shape
        - y: Numpy array of labels, of any shape but with y.shape[0] == X.shape[0]
        - batch_size: Integer giving number of elements per minibatch
        - shuffle: (optional) Boolean, whether to shuffle the data on each epoch
        """
        assert X.shape[0] == y.shape[0], 'Got different numbers of data and labels'
        self.X, self.y = X, y
        self.batch_size, self.shuffle = batch_size, shuffle

    def __iter__(self):
        N, B = self.X.shape[0], self.batch_size
        idxs = np.arange(N)
        if self.shuffle:
            np.random.shuffle(idxs)
        return iter((self.X[i:i+B], self.y[i:i+B]) for i in range(0, N, B))


train_dset = Dataset(X_train, y_train, batch_size=64, shuffle=True)
val_dset   = Dataset(X_val, y_val, batch_size=64, shuffle=False)
test_dset  = Dataset(X_test, y_test, batch_size=64)

#%%

# We can iterate through a dataset like this:
for t, (x, y) in enumerate(train_dset):
    print(t, x.shape, y.shape)
    if t > 5: break

#%%  Setup the use for CPU or GPU
# Set up some global variables
USE_GPU = False

if USE_GPU:
    device = '/device:GPU:0'
else:
    device = '/cpu:0'

# Constant to control how often we print when training models
print_every = 100

print('Using device: ', device)



#%%
###############################################################################
#                                                                             #
#                        Part II: Barebone TensorFlow                         #
#                                                                             #
###############################################################################

# create flatten function

def flatten(x):
    """    
    Input:
    - TensorFlow Tensor of shape (N, D1, ..., DM)
    
    Output:
    - TensorFlow Tensor of shape (N, D1 * ... * DM)
    """
    N = tf.shape(x)[0]
    return tf.reshape(x, (N, -1))

def test_flatten():
    # Construct concrete values of the input data x using numpy
    x_np = np.arange(24).reshape((2, 3, 4))
    print('x_np:\n', x_np, '\n')
    # Compute a concrete output value.
    x_flat_np = flatten(x_np)
    print('x_flat_np:\n', x_flat_np, '\n')

test_flatten()

#%%  define a two-layer network using tf, 


def two_layer_fc(x, params):
    """
    A fully-connected neural network; the architecture is:
    fully-connected layer -> ReLU -> fully connected layer.
    Note that we only need to define the forward pass here; TensorFlow will take
    care of computing the gradients for us.
    
    The input to the network will be a minibatch of data, of shape
    (N, d1, ..., dM) where d1 * ... * dM = D. The hidden layer will have H units,
    and the output layer will produce scores for C classes.

    Inputs:
    - x: A TensorFlow Tensor of shape (N, d1, ..., dM) giving a minibatch of
      input data.
    - params: A list [w1, w2] of TensorFlow Tensors giving weights for the
      network, where w1 has shape (D, H) and w2 has shape (H, C).
    
    Returns:
    - scores: A TensorFlow Tensor of shape (N, C) giving classification scores
      for the input data x.
    """
    w1, w2 = params                   # Unpack the parameters
    x = flatten(x)                    # Flatten the input; now x has shape (N, D)
    h = tf.nn.relu(tf.matmul(x, w1))  # Hidden layer: h has shape (N, H)
    scores = tf.matmul(h, w2)         # Compute scores of shape (N, C)
    return scores


def two_layer_fc_test():
    hidden_layer_size = 42

    # Scoping our TF operations under a tf.device context manager 
    # lets us tell TensorFlow where we want these Tensors to be
    # multiplied and/or operated on, e.g. on a CPU or a GPU.
    with tf.device(device):        
        x = tf.zeros((64, 32, 32, 3))
        w1 = tf.zeros((32 * 32 * 3, hidden_layer_size))
        w2 = tf.zeros((hidden_layer_size, 10))

        # Call our two_layer_fc function for the forward pass of the network.
        scores = two_layer_fc(x, [w1, w2])

    print(scores.shape)

two_layer_fc_test()


#%%

def three_layer_convnet(x, params):
    """
    A three-layer convolutional network with the architecture described above.
    
    Inputs:
    - x: A TensorFlow Tensor of shape (N, H, W, 3) giving a minibatch of images
    - params: A list of TensorFlow Tensors giving the weights and biases for the
      network; should contain the following:
      - conv_w1: TensorFlow Tensor of shape (KH1, KW1, 3, channel_1) giving
        weights for the first convolutional layer.
      - conv_b1: TensorFlow Tensor of shape (channel_1,) giving biases for the
        first convolutional layer.
      - conv_w2: TensorFlow Tensor of shape (KH2, KW2, channel_1, channel_2)
        giving weights for the second convolutional layer
      - conv_b2: TensorFlow Tensor of shape (channel_2,) giving biases for the
        second convolutional layer.
      - fc_w: TensorFlow Tensor giving weights for the fully-connected layer.
        Can you figure out what the shape should be? 
      - fc_b: TensorFlow Tensor giving biases for the fully-connected layer.
        Can you figure out what the shape should be? 
    """
    conv_w1, conv_b1, conv_w2, conv_b2, fc_w, fc_b = params
    scores = None
    ############################################################################
    # TODO: Implement the forward pass for the three-layer ConvNet.            #
    ############################################################################
    # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

    conv_layer1 = tf.nn.conv2d(x, 
                               filters = conv_w1, 
                               padding = [[0, 0], [2, 2], [2, 2], [0, 0]],  # zero padding of 2, used in format 'NHWC'
                               strides = 1,
                               data_format = 'NHWC')   
    # input images size (N, H, W, 3),  conv_w1 size (KH1, KW1, 3, channel_1), here 3 is the image channels
    # new_H1 = 1 + (H + 2 * pad - KH1) / stride, here pad is 2. 
    # new_W1 = 1 + (W + 2 * pad - KW1) / stride, here pad is 2. 
        
    conv_layer1 = conv_layer1 + conv_b1  # conv_b1 with shape (channel_1,)    
    # ------- so conv_layer1 size is (N, new_H1, new_W1, channel_1)
    
    
    
    relu_layer1 = tf.nn.relu(conv_layer1)    # shape (N, new_H1, new_W1, channel_1)
    
    
    conv_layer2 = tf.nn.conv2d(relu_layer1, 
                               filters = conv_w2, 
                               padding = [[0, 0], [1, 1], [1, 1], [0, 0]],  # zero padding of 1, used in format 'NHWC'
                               strides = 1,
                               data_format = 'NHWC')  
    # here input size (N, new_H1, new_W1, channel_1),  conv_w2 size (KH2, KW2, channel_1, channel_2), 
    # new_H2 = 1 + (new_H1 + 2 * pad - KH2) / stride, here pad is 1. 
    # new_W2 = 1 + (new_W1 + 2 * pad - KW2) / stride, here pad is 1. 
        
    conv_layer2 = conv_layer2 + conv_b2     # conv_b2 with shape(channel_2,)
    # ------- so conv_layer1 size is (N, new_H2, new_W2, channel_2)
    
    
    relu_layer2 = tf.nn.relu(conv_layer2)   # (N, new_H2, new_W2, channel_2)
    relu_layer2_flat = flatten(relu_layer2) # Flatten the input; now x has shape (N, D=new_H2*new_W2*channel_2)
    
    scores = tf.matmul(relu_layer2_flat, fc_w)+fc_b    # fc_w has a size of (D=new_H2*new_W2*channel_2, 10), here 10 means 10 classes
    # fc_b has a size of (10,),
    # Compute scores of shape (N, 10)
    
 
    pass

    # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
    ############################################################################
    #                              END OF YOUR CODE                            #
    ############################################################################
    return scores



'''
After defing the forward pass of the three-layer ConvNet above, run the following cell to test your implementation. Like the two-layer network, we run the graph on a batch of zeros just to make sure the function doesn't crash, and produces outputs of the correct shape.
When you run this function, scores_np should have shape (64, 10).
'''

def three_layer_convnet_test():
    
    with tf.device(device):
        x = tf.zeros((64, 32, 32, 3))
        conv_w1 = tf.zeros((5, 5, 3, 6))
        conv_b1 = tf.zeros((6,))
        conv_w2 = tf.zeros((3, 3, 6, 9))
        conv_b2 = tf.zeros((9,))
        fc_w = tf.zeros((32 * 32 * 9, 10))
        fc_b = tf.zeros((10,))
        params = [conv_w1, conv_b1, conv_w2, conv_b2, fc_w, fc_b]
        scores = three_layer_convnet(x, params)

    # Inputs to convolutional layers are 4-dimensional arrays with shape
    # [batch_size, height, width, channels]
    print('scores_np has shape: ', scores.shape)

three_layer_convnet_test()

#%% Barebones TensorFlow: Training Step

'''
We now define the training_step function performs a single training step. This will take three basic steps:
1. Compute the loss
2. Compute the gradient of the loss with respect to all network weights
3. Make a weight update step using (stochastic) gradient descent.

We need to use a few new TensorFlow functions to do all of this:
For computing the cross-entropy loss we'll use tf.nn.sparse_softmax_cross_entropy_with_logits:
    https://www.tensorflow.org/versions/r2.0/api_docs/python/tf/nn/sparse_softmax_cross_entropy_with_logits
For averaging the loss across a minibatch of data we'll use tf.reduce_mean: 
    https://www.tensorflow.org/versions/r2.0/api_docs/python/tf/reduce_mean
For computing gradients of the loss with respect to the weights we'll use tf.GradientTape (useful for Eager execution): 
    https://www.tensorflow.org/versions/r2.0/api_docs/python/tf/GradientTape
We'll mutate the weight values stored in a TensorFlow Tensor using tf.assign_sub ("sub" is for subtraction): 
    https://www.tensorflow.org/api_docs/python/tf/assign_sub
'''

def training_step(model_fn, x, y, params, learning_rate):
    with tf.GradientTape() as tape:
        scores = model_fn(x, params) # Forward pass of the model
        loss   = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=y, logits=scores)
        total_loss = tf.reduce_mean(loss)                # average the loss across a minibatch of data
        grad_params = tape.gradient(total_loss, params)  # compute the gradient

        # Make a vanilla gradient descent step on all of the model parameters
        # Manually update the weights using assign_sub()
        for w, grad_w in zip(params, grad_params):
            w.assign_sub(learning_rate * grad_w)  # mutate the weight values using substraction
                        
        return total_loss


def train_part2(model_fn, init_fn, learning_rate):
    """
    Train a model on CIFAR-10.
    
    Inputs:
    - model_fn: A Python function that performs the forward pass of the model
      using TensorFlow; it should have the following signature:
      scores = model_fn(x, params) where x is a TensorFlow Tensor giving a
          minibatch of image data, params is a list of TensorFlow Tensors holding
          the model weights, and scores is a TensorFlow Tensor of shape (N, C)
          giving scores for all elements of x.
    - init_fn: A Python function that initializes the parameters of the model.
          It should have the signature params = init_fn() where params is a list
          of TensorFlow Tensors holding the (randomly initialized) weights of the
          model.
    - learning_rate: Python float giving the learning rate to use for SGD.
    """
    
    
    params = init_fn()  # Initialize the model parameters            
        
    for t, (x_np, y_np) in enumerate(train_dset):
        # Run the graph on a batch of training data.
        loss = training_step(model_fn, x_np, y_np, params, learning_rate)
        
        # Periodically print the loss and check accuracy on the val set.
        if t % print_every == 0:
            print('Iteration %d, loss = %.4f' % (t, loss))
            check_accuracy(val_dset, x_np, model_fn, params)

def check_accuracy(dset, x, model_fn, params):
    """
    Check accuracy on a classification model, e.g. for validation.
    
    Inputs:
    - dset: A Dataset object against which to check accuracy
    - x: A TensorFlow placeholder Tensor where input images should be fed
    - model_fn: the Model we will be calling to make predictions on x
    - params: parameters for the model_fn to work with
      
    Returns: Nothing, but prints the accuracy of the model
    """
    num_correct, num_samples = 0, 0
    for x_batch, y_batch in dset:
        scores_np = model_fn(x_batch, params).numpy()
        y_pred = scores_np.argmax(axis=1)
        num_samples += x_batch.shape[0]
        num_correct += (y_pred == y_batch).sum()
    acc = float(num_correct) / num_samples
    print('Got %d / %d correct (%.2f%%)' % (num_correct, num_samples, 100 * acc))

#%%   Barebones TensorFlow: Initialization

'''
We'll use the following utility method to initialize the weight matrices for our models 
using Kaiming's normalization method.
'''

def create_matrix_with_kaiming_normal(shape):
    if len(shape) == 2:
        fan_in, fan_out = shape[0], shape[1]
    elif len(shape) == 4:
        fan_in, fan_out = np.prod(shape[:3]), shape[3]
    return tf.keras.backend.random_normal(shape) * np.sqrt(2.0 / fan_in)


#%% Barebones TensorFlow: Train a Two-Layer Network

'''
We are finally ready to use all of the pieces defined above to train a two-layer fully-connected network on CIFAR-10.
We just need to define a function to initialize the weights of the model, and call train_part2.
Defining the weights of the network introduces another important piece of TensorFlow API: tf.Variable. 
A TensorFlow Variable is a Tensor whose value is stored in the graph and persists across runs of the computational graph; 
however unlike constants defined with tf.zeros or tf.random_normal, the values of a Variable can be mutated as the graph runs; 
these mutations will persist across graph runs. Learnable parameters of the network are usually stored in Variables.
You don't need to tune any hyperparameters, but you should achieve validation accuracies above 40% after one epoch of training.
'''
def two_layer_fc_init():
    """
    Initialize the weights of a two-layer network, for use with the
    two_layer_network function defined above. 
    You can use the `create_matrix_with_kaiming_normal` helper!
    
    Inputs: None
    
    Returns: A list of:
    - w1: TensorFlow tf.Variable giving the weights for the first layer
    - w2: TensorFlow tf.Variable giving the weights for the second layer
    """
    hidden_layer_size = 4000
    w1 = tf.Variable(create_matrix_with_kaiming_normal((3 * 32 * 32, hidden_layer_size)))
    w2 = tf.Variable(create_matrix_with_kaiming_normal((hidden_layer_size, 10)))
    return [w1, w2]

learning_rate = 1e-2
train_part2(two_layer_fc, two_layer_fc_init, learning_rate)




#%% Barebones TensorFlow: Train a three-layer ConvNet
'''
Barebones TensorFlow: Train a three-layer ConvNet
We will now use TensorFlow to train a three-layer ConvNet on CIFAR-10.
You need to implement the three_layer_convnet_init function. Recall that the architecture of the network is:
1.Convolutional layer (with bias) with 32 5x5 filters, with zero-padding 2
2.ReLU
3.Convolutional layer (with bias) with 16 3x3 filters, with zero-padding 1
4.ReLU
5.Fully-connected layer (with bias) to compute scores for 10 classes
You don't need to do any hyperparameter tuning, but you should see validation accuracies 
above 43% after one epoch of training.
'''

def three_layer_convnet_init():
    """
    Initialize the weights of a Three-Layer ConvNet, for use with the
    three_layer_convnet function defined above.
    You can use the `create_matrix_with_kaiming_normal` helper!
    
    Inputs: None
    
    Returns a list containing:
    - conv_w1: TensorFlow tf.Variable giving weights for the first conv layer
    - conv_b1: TensorFlow tf.Variable giving biases for the first conv layer
    - conv_w2: TensorFlow tf.Variable giving weights for the second conv layer
    - conv_b2: TensorFlow tf.Variable giving biases for the second conv layer
    - fc_w: TensorFlow tf.Variable giving weights for the fully-connected layer
    - fc_b: TensorFlow tf.Variable giving biases for the fully-connected layer
    """
    params = None
    ############################################################################
    # TODO: Initialize the parameters of the three-layer network.              #
    ############################################################################
    # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
    image_channel = 3
    image_H, image_W = 32, 32
    
    KH1, KW1 = 5, 5
    channel_1 = 32
    conv_w1 = tf.Variable(create_matrix_with_kaiming_normal( (KH1, KW1, image_channel, channel_1) )) 
    conv_b1 = tf.Variable(create_matrix_with_kaiming_normal( (1,1,1, channel_1) ))      
    # conv_layer1 size is (N, new_H1, new_W1, channel_1)
    
    pad,stride = 2,1
    new_H1 = 1 + (image_H + 2 * pad - KH1) / stride
    new_W1 = 1 + (image_W + 2 * pad - KW1) / stride
    
    
    KH2, KW2  = 3, 3
    channel_2 = 16
    conv_w2 = tf.Variable(create_matrix_with_kaiming_normal((KH2, KW2, channel_1, channel_2))) 
    conv_b2 = tf.Variable(create_matrix_with_kaiming_normal( (1,1,1, channel_2 ) )) 
    # conv_layer2 size is (KH2, KW2, channel_1, channel_2)
    pad,stride = 1,1
    new_H2 =int( 1 + (new_H1 + 2 * pad - KH2) / stride )
    new_W2 =int( 1 + (new_W1 + 2 * pad - KW2) / stride )

    image_classes = 10
    fc_w = tf.Variable(create_matrix_with_kaiming_normal( (new_H2*new_W2*channel_2, image_classes) )) 
    fc_b = tf.Variable(create_matrix_with_kaiming_normal( (1, image_classes) )) 
    # Compute scores of shape (N, 10)
    params = [conv_w1, conv_b1, conv_w2, conv_b2, fc_w, fc_b]
    pass

    # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
    ############################################################################
    #                             END OF YOUR CODE                             #
    ############################################################################
    return params

learning_rate = 3e-3
train_part2(three_layer_convnet, three_layer_convnet_init, learning_rate)



#%%
###############################################################################
#                                                                             #
#                    Part III: Keras model subclass API                       #
#                                                                             #
###############################################################################


'''
Implementing a neural network using the low-level TensorFlow API is a good way to understand 
 TensorFlow works, but it's a little inconvenient - we had to manually keep track of all 
 Tensors holding learnable parameters. This was fine for a small network, but could quickly 
 become unweildy for a large complex model.
 
Fortunately TensorFlow 2.0 provides higher-level APIs such as tf.keras which make it easy to 
build models out of modular, object-oriented layers. Further, TensorFlow 2.0 uses eager execution 
that evaluates operations immediately, without explicitly constructing any computational graphs. 
This makes it easy to write and debug models, and reduces the boilerplate code.
In this part of the notebook we will define neural network models using the tf.keras.Model API. 
To implement your own model, you need to do the following:
    
1. Define a new class which subclasses tf.keras.Model. Give your class an intuitive name that describes it, 
    like TwoLayerFC or ThreeLayerConvNet.
2. In the initializer __init__() for your new class, define all the layers you need as class attributes. 
    The tf.keras.layers package provides many common neural-network layers, like tf.keras.layers.Dense for 
    fully-connected layers and tf.keras.layers.Conv2D for convolutional layers. Under the hood, these layers will 
    construct Variable Tensors for any learnable parameters. Warning: Don't forget to
    call super(YourModelName, self).__init__() as the first line in your initializer!
3. Implement the call() method for your class; this implements the forward pass of your model, 
    and defines the connectivity of your network. Layers defined in __init__() implement __call__() 
    so they can be used as function objects that transform input Tensors into output Tensors. 
    Don't define any new layers in call(); any layers you want to use in the forward pass should be defined 
    in __init__().

After you define your tf.keras.Model subclass, you can instantiate it and use it like the model functions 
from Part II.
'''


# Keras Model Subclassing API: Two-Layer Network

class TwoLayerFC(tf.keras.Model):  # subclassing the Model class
    # in the case of subclassing a Model class, we should define layers in __init__
    # and implement the method's forward pass in call()
    def __init__(self, hidden_size, num_classes): 
        super(TwoLayerFC, self).__init__()          # this will set the self.built=True
        
        initializer = tf.initializers.VarianceScaling(scale=2.0)  # initialize given size of weight matrix, with randomly generated numbers
        
        self.fc1 = tf.keras.layers.Dense(hidden_size, activation='relu',
                                   kernel_initializer=initializer)
        self.fc2 = tf.keras.layers.Dense(num_classes, activation='softmax',
                                   kernel_initializer=initializer)
        self.flatten = tf.keras.layers.Flatten()
    
    def call(self, x, training=False):
        x = self.flatten(x)
        x = self.fc1(x)
        x = self.fc2(x)
        return x


def test_TwoLayerFC():
    """ A small unit test to exercise the TwoLayerFC model above. """
    input_size, hidden_size, num_classes = 50, 42, 10
    x = tf.zeros((64, input_size))
    model = TwoLayerFC(hidden_size, num_classes)
    with tf.device(device):  # chose to use CPU or GPU, device name was defined earlier
        scores = model(x)
        print(scores.shape)
        
test_TwoLayerFC()

#%% Keras Model Subclassing API: Three-Layer ConvNet
'''
Now it's your turn to implement a three-layer ConvNet using the tf.keras.Model API. 
Your model should have the same architecture used in Part II:

    1.Convolutional layer with 5 x 5 kernels, with zero-padding of 2
    2. ReLU nonlinearity
    3. Convolutional layer with 3 x 3 kernels, with zero-padding of 1
    4. ReLU nonlinearity
    5. Fully-connected layer to give class scores
    6. Softmax nonlinearity

You should initialize the weights of your network using the same initialization method 
as was used in the two-layer network above.
Hint: Refer to the documentation for tf.keras.layers.Conv2D and tf.keras.layers.Dense:
'''


class ThreeLayerConvNet(tf.keras.Model):
    def __init__(self, channel_1, channel_2, num_classes):
        super(ThreeLayerConvNet, self).__init__()
        ########################################################################
        # TODO: Implement the __init__ method for a three-layer ConvNet. You   #
        # should instantiate layer objects to be used in the forward pass.     #
        ########################################################################
        # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

        initializer = tf.initializers.VarianceScaling(scale=2.0)  # initialize given size of weight matrix, with randomly generated numbers

        self.zeropad1 = tf.keras.layers.ZeroPadding2D(padding=(2,2),
                                                      data_format= 'channels_last')
        self.zeropad2 = tf.keras.layers.ZeroPadding2D(padding=(1,1),
                                                      data_format= 'channels_last')

        self.conv1 = tf.keras.layers.Conv2D(filters=channel_1, # number of conv filters
                                            kernel_size=(5,5), # the filter dimensions (height, width)
                                            strides=(1,1),
                                            padding='valid',
                                            kernel_initializer=initializer,
                                            data_format= 'channels_last', # channel_last, also the default settings
                                            activation='relu')
    
        self.conv2 = tf.keras.layers.Conv2D(filters=channel_2, # number of conv filters
                                            kernel_size=(3,3), # the filter dimensions (height, width)
                                            strides=(1,1),
                                            padding='valid',
                                            kernel_initializer=initializer,
                                            data_format= 'channels_last', # channel_last, also the default settings
                                            activation='relu')
        
        self.fc = tf.keras.layers.Dense(num_classes, 
                                        activation='softmax',
                                        kernel_initializer=initializer)
        
        self.flat = tf.keras.layers.Flatten(data_format= 'channels_last')
        
        pass

        # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
        ########################################################################
        #                           END OF YOUR CODE                           #
        ########################################################################
        
    def call(self, x, training=False):
        scores = None
        ########################################################################
        # TODO: Implement the forward pass for a three-layer ConvNet. You      #
        # should use the layer objects defined in the __init__ method.         #
        ########################################################################
        # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
        x=self.zeropad1(x)
        x=self.conv1(x)    # first conv layer with relu nonlinearity
        x=self.zeropad2(x)
        x=self.conv2(x)    # 2nd conv layer with relu nonlinearity
        x=self.flat(x)
        scores=self.fc(x)  # fully connected layer with softmax nonlinearity
        
        pass

        # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
        ########################################################################
        #                           END OF YOUR CODE                           #
        ########################################################################        
        return scores


def test_ThreeLayerConvNet():    
    channel_1, channel_2, num_classes = 12, 8, 10
    model = ThreeLayerConvNet(channel_1, channel_2, num_classes)
    with tf.device(device):
        x = tf.zeros((64, 3, 32, 32))
        scores = model(x)
        print(scores.shape)

test_ThreeLayerConvNet()


#%%
###############################################################################
#                                                                             #
#              Keras Model Subclassing API: Eager Training                    #
#                                                                             #
###############################################################################

'''
While keras models have a builtin training loop (using the model.fit), sometimes 
you need more customization. Here's an example, of a training loop implemented with eager execution.
In particular, notice tf.GradientTape. Automatic differentiation is used in the backend 
for implementing backpropagation in frameworks like TensorFlow. During eager execution, 
tf.GradientTape is used to trace operations for computing gradients later. A particular 
tf.GradientTape can only compute one gradient; subsequent calls to tape will throw a runtime error. 
TensorFlow 2.0 ships with easy-to-use built-in metrics under tf.keras.metrics module. 
Each metric is an object, and we can use update_state() to add observations and reset_state() 
to clear all observations. We can get the current result of a metric by calling result() on the metric object.
'''

def train_part34(model_init_fn, optimizer_init_fn, num_epochs=1, is_training=False):
    """
    Simple training loop for use with models defined using tf.keras. It trains
    a model for one epoch on the CIFAR-10 training set and periodically checks
    accuracy on the CIFAR-10 validation set.
    
    Inputs:
    - model_init_fn: A function that takes no parameters; when called it
      constructs the model we want to train: model = model_init_fn()
    - optimizer_init_fn: A function which takes no parameters; when called it
      constructs the Optimizer object we will use to optimize the model:
      optimizer = optimizer_init_fn()
    - num_epochs: The number of epochs to train for
    
    Returns: Nothing, but prints progress during training
    """    
    with tf.device(device):

        # Compute the loss like we did in Part II
        loss_fn = tf.keras.losses.SparseCategoricalCrossentropy()  # compute cross entropy loss between labels and predictions
        
        model     = model_init_fn()
        optimizer = optimizer_init_fn()
        
        train_loss     = tf.keras.metrics.Mean(name='train_loss') # compute the (weighted) mean of the given values
        train_accuracy = tf.keras.metrics.SparseCategoricalAccuracy(name='train_accuracy')  # calculate how often prediction matches integer labels
    
        val_loss = tf.keras.metrics.Mean(name='val_loss')
        val_accuracy = tf.keras.metrics.SparseCategoricalAccuracy(name='val_accuracy')
        
        t = 0
        my_print_epoch = 0
        for epoch in range(num_epochs):
            
                                      
            my_print_epoch = epoch + 1  # when printing for view, epoch starting from 1, not 0.
            # Reset the metrics - https://www.tensorflow.org/alpha/guide/migration_guide#new-style_metrics
            train_loss.reset_states()      # starting each epoch, clear all observations
            train_accuracy.reset_states()  
            
            for x_np, y_np in train_dset:       # train_dset is an pre-constructed data object
                with tf.GradientTape() as tape: # tf.GradientTape(), record operations for automatic differentiation
                                                # is used to trace operations for computing gradients later.
                    
                    # Use the model function to build the forward pass.
                    scores = model(x_np, training=is_training)  # the initialized model will take input data x_np, and computes is score
                    loss   = loss_fn(y_np, scores)    # given the score and prediction, calculate the loss
      
                    # GradientTape() class has a method called 'gradient'
                    # it computes the gradient using operations recorded in context of this tape.
                    gradients = tape.gradient(loss,   # the target to be differentiated
                                              model.trainable_variables) 
                    
                    optimizer.apply_gradients(zip(gradients, model.trainable_variables)) # update the gradients into the training variables
                    
                    # Update the metrics
                    train_loss.update_state(loss)   # add an observation
                    train_accuracy.update_state(y_np, scores)  
                    
                    if t % print_every == 0:
                        val_loss.reset_states()
                        val_accuracy.reset_states()
                        for test_x, test_y in val_dset:
                            # During validation at end of epoch, training set to False
                            prediction = model(test_x, training=False)
                            t_loss = loss_fn(test_y, prediction)

                            val_loss.update_state(t_loss)
                            val_accuracy.update_state(test_y, prediction)
                        
                        template = 'Iteration {}, Epoch {}, Loss: {}, Train Accuracy: {}%, Val Loss: {}, Val Accuracy: {}%'
                        
                        print (template.format(t, my_print_epoch, #epoch+1,
                                             '%.4f'%(train_loss.result()),               # training loss
                                             '%.2f'%(train_accuracy.result()*100),       # training acc
                                             '%.4f'%(val_loss.result()),                 # validation loss
                                             '%.2f'%(val_accuracy.result()*100)))        # validation acc
                    t += 1


#%%  Keras Model Subclassing API: Train a Two-Layer Network

hidden_size, num_classes = 4000, 10
learning_rate = 1e-2

def model_init_fn():
    return TwoLayerFC(hidden_size, num_classes)

def optimizer_init_fn():
    return tf.keras.optimizers.SGD(learning_rate=learning_rate)

train_part34(model_init_fn, optimizer_init_fn)


#%% 
###############################################################################
#                                                                             #
#          Keras Model Subclassing API: Train a Three-Layer ConvNet           #
#                                                                             #
###############################################################################


learning_rate = 3e-3
channel_1, channel_2, num_classes = 32, 16, 10

def model_init_fn():
    model = None
    ###########################################################################
    # TODO: Complete the implementation of model_fn.                          #
    ###########################################################################
    # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
    model = ThreeLayerConvNet(32, 16, 10)  # the model is defined above, (channel_1, channel_2, num_classes)
    pass

    # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
    ###########################################################################
    #                           END OF YOUR CODE                              #
    ###########################################################################
    return model

def optimizer_init_fn():
    optimizer = None
    ###########################################################################
    # TODO: Complete the implementation of model_fn.                          #
    ###########################################################################
    # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
    optimizer = tf.keras.optimizers.SGD(learning_rate=3e-3)
    pass

    # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
    ###########################################################################
    #                           END OF YOUR CODE                              #
    ###########################################################################
    return optimizer

train_part34(model_init_fn, optimizer_init_fn)

#%%


###############################################################################
#                                                                             #
#                      Part IV: Keras Sequential API                          #
#                   Keras Sequential API: Two-Layer Network                   #
#                                                                             #
###############################################################################


learning_rate = 1e-2

def model_init_fn():
    input_shape = (32, 32, 3)
    hidden_layer_size, num_classes = 4000, 10
    initializer = tf.initializers.VarianceScaling(scale=2.0)
    layers = [
        tf.keras.layers.Flatten(input_shape=input_shape),
        tf.keras.layers.Dense(hidden_layer_size, activation='relu',
                              kernel_initializer=initializer),
        tf.keras.layers.Dense(num_classes, activation='softmax', 
                              kernel_initializer=initializer),
    ]
    model = tf.keras.Sequential(layers)
    return model

def optimizer_init_fn():
    return tf.keras.optimizers.SGD(learning_rate=learning_rate) 

train_part34(model_init_fn, optimizer_init_fn)




#%% get rid of custom training loop, but use Keras built-in API instead

model = model_init_fn()  # use Keras sequeuncy API to build the model structure
# model.compile helps to configure the model for training
model.compile(optimizer=tf.keras.optimizers.SGD(learning_rate=learning_rate),  # input the optimizer function
              loss='sparse_categorical_crossentropy',                  # calculate the cross entropy loss
              metrics=[tf.keras.metrics.sparse_categorical_accuracy])  # calcualte the accuracy
model.fit(X_train, y_train, batch_size=64, epochs=1, validation_data=(X_val, y_val))
model.evaluate(X_test, y_test)



#%% 

###############################################################################
#                   Keras Sequential API: Three-Layer ConvNet                 #
#                                                                             #
###############################################################################

def model_init_fn():
    model = None
    ############################################################################
    # TODO: Construct a three-layer ConvNet using tf.keras.Sequential.         #
    ############################################################################
    # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
    input_shape = (32, 32, 3)
    num_classes = 10
    initializer = tf.initializers.VarianceScaling(scale=2.0)
    layers = [
        tf.keras.layers.ZeroPadding2D(input_shape=input_shape, 
                                      padding=(2,2),data_format= 'channels_last'),
        tf.keras.layers.Conv2D(filters=32, # number of conv filters
                                            kernel_size=(5,5), # the filter dimensions (height, width)
                                            strides=(1,1),
                                            padding='valid',
                                            kernel_initializer=initializer,
                                            data_format= 'channels_last', # channel_last, also the default settings
                                            activation='relu'),
        tf.keras.layers.ZeroPadding2D(padding=(1,1),data_format= 'channels_last'),
        tf.keras.layers.Conv2D(filters=16, # number of conv filters
                                            kernel_size=(3,3), # the filter dimensions (height, width)
                                            strides=(1,1),
                                            padding='valid',
                                            kernel_initializer=initializer,
                                            data_format= 'channels_last', # channel_last, also the default settings
                                            activation='relu'),
        
        tf.keras.layers.Flatten(data_format= 'channels_last'),
        tf.keras.layers.Dense(num_classes, activation='softmax',
                                        kernel_initializer=initializer) 
        ]
    
    model = tf.keras.Sequential(layers)
    pass

    # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
    ############################################################################
    #                            END OF YOUR CODE                              #
    ############################################################################
    return model


learning_rate = 5e-4
def optimizer_init_fn():
    optimizer = None
    ############################################################################
    # TODO: Complete the implementation of model_fn.                           #
    ############################################################################
    # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
    optimizer = tf.keras.optimizers.SGD(learning_rate=5e-4) 
    pass

    # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
    ############################################################################
    #                           END OF YOUR CODE                               #
    ############################################################################
    return optimizer

train_part34(model_init_fn, optimizer_init_fn)


#%%   also try to use keras API to get rid of custom training loops

model = model_init_fn()
model.compile(optimizer='sgd',
              loss='sparse_categorical_crossentropy',
              metrics=[tf.keras.metrics.sparse_categorical_accuracy])
model.fit(X_train, y_train, batch_size=64, epochs=1, validation_data=(X_val, y_val))
model.evaluate(X_test, y_test)

#%%

###############################################################################
#                        Part IV: Functional API                              #
#                                                                             #
###############################################################################

'''
In the previous section, we saw how we can use tf.keras.Sequential to stack layers 
to quickly build simple models. But this comes at the cost of losing flexibility.
Often we will have to write complex models that have non-sequential data flows: 
    a layer can have multiple inputs and/or outputs, such as stacking the output of 
    2 previous layers together to feed as input to a third! (Some examples are residual 
                                                             connections and dense blocks.)
In such cases, we can use Keras functional API to write models with complex topologies such as:
Multi-input models
Multi-output models
Models with shared layers (the same layer called several times)
Models with non-sequential data flows (e.g. residual connections)
Writing a model with Functional API requires us to create a tf.keras.Model instance 
and explicitly write input tensors and output tensors for this model. 
'''

def two_layer_fc_functional(input_shape, hidden_size, num_classes):  
    
    initializer = tf.initializers.VarianceScaling(scale=2.0)
    inputs      = tf.keras.Input(shape=input_shape)
    flattened_inputs = tf.keras.layers.Flatten()(inputs)
    
    fc1_output = tf.keras.layers.Dense(hidden_size, activation='relu',
                                 kernel_initializer=initializer)(flattened_inputs)
   
    scores = tf.keras.layers.Dense(num_classes, activation='softmax',
                             kernel_initializer=initializer)(fc1_output)

    # Instantiate the model given inputs and outputs.
    model = tf.keras.Model(inputs=inputs, outputs=scores)
    return model

def test_two_layer_fc_functional():
    """ A small unit test to exercise the TwoLayerFC model above. """
    input_size, hidden_size, num_classes = 50, 42, 10
    input_shape = (50,)
    
    x = tf.zeros((64, input_size))
    model = two_layer_fc_functional(input_shape, hidden_size, num_classes)
    
    with tf.device(device):
        scores = model(x)
        print(scores.shape)
        
test_two_layer_fc_functional()


input_shape = (32, 32, 3)
hidden_size, num_classes = 4000, 10
learning_rate = 1e-2

def model_init_fn():
    model = two_layer_fc_functional(input_shape, hidden_size, num_classes)
    return model

def optimizer_init_fn():
    optimizer = tf.keras.optimizers.SGD(learning_rate=learning_rate)
    return optimizer

train_part34(model_init_fn, optimizer_init_fn)



#%%
###############################################################################
#           Part V: CIFAR-10 open-ended challenge                             #
#                                                                             #
###############################################################################

class CustomConvNet(tf.keras.Model):
   
    
    def __init__(self, channel_1=32, channel_2=64, num_classes=10, batch_size=64):
        super(CustomConvNet, self).__init__()
        ############################################################################
        # TODO: Construct a model that performs well on CIFAR-10                   #
        ############################################################################
        # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
      
        initializer = tf.initializers.VarianceScaling(scale=2.0)  # initialize given size of weight matrix, with randomly generated numbers

        self.zeropad1 = tf.keras.layers.ZeroPadding2D(padding=(2,2),
                                                      data_format= 'channels_last')
        self.zeropad2 = tf.keras.layers.ZeroPadding2D(padding=(1,1),
                                                      data_format= 'channels_last')



        self.conv1 = tf.keras.layers.Conv2D(filters=channel_1, # number of conv filters
                                            kernel_size=(5,5), # the filter dimensions (height, width)
                                            strides=(1,1),
                                            padding='valid',
                                            kernel_initializer=initializer,
                                            data_format= 'channels_last', # channel_last, also the default settings
                                            activation='relu')
    
        self.conv2 = tf.keras.layers.Conv2D(filters=channel_2, # number of conv filters
                                            kernel_size=(3,3), # the filter dimensions (height, width)
                                            strides=(1,1),
                                            padding='valid',
                                            kernel_initializer=initializer,
                                            data_format= 'channels_last', # channel_last, also the default settings
                                            activation='relu')
        
        
        self.maxpool1 = tf.keras.layers.MaxPool2D(pool_size=(2, 2), strides=(2,2),
                                                  padding='valid',
                                                  data_format= 'channels_last',)
        
        self.dropout = tf.keras.layers.Dropout(0.3)
                                               #noise_shape=(batch_size, 32, 32, 1))  # (batch_size, height, width, 1)
        
        
        

        self.fc = tf.keras.layers.Dense(num_classes, 
                                        activation='softmax',
                                        kernel_initializer=initializer)
        
        self.flat = tf.keras.layers.Flatten(data_format= 'channels_last')



        pass

        # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
        ############################################################################
        #                            END OF YOUR CODE                              #
        ############################################################################
    
    def call(self, input_tensor, training=False):
        ############################################################################
        # TODO: Construct a model that performs well on CIFAR-10                   #
        ############################################################################
        # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
        #batch_size = input_tensor.shape[0]
        #height     = input_tensor.shape[1] 
        #width      = input_tensor.shape[2]
            
        x=self.zeropad1(input_tensor)
        x=self.conv1(x)    # first conv layer with relu nonlinearity
        x=self.dropout(x)  # noise_shape=(x.shape[0], x.shape[1], x.shape[2], 1))  #  (batch_size, height, width, channel), dropout mask will be the same across all channels 
        
        
        x=self.zeropad2(x)
        x=self.conv2(x)    # 2nd conv layer with relu nonlinearity
        x=self.maxpool1(x)
        
        x=self.flat(x)
        scores=self.fc(x)  # fully connected layer with softmax nonlinearity
        
        pass

        # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
        ############################################################################
        #                            END OF YOUR CODE                              #
        ############################################################################
        
        return scores


'''
N=57000 number of training examples, batch_size=64, 
57000/64 = 890.625

it will take about 890 iterations to finish one epoch of training
'''
print_every = 100    # print results every 100 iterations during training, 
num_epochs  = 10

#device = '/device:GPU:0'   # Change this to a CPU/GPU as you wish!
device = '/cpu:0'        # Change this to a CPU/GPU as you wish!

#model = CustomConvNet()
def model_init_fn():
    model = CustomConvNet(channel_1=32, channel_2=64, num_classes=10, batch_size=64)
    return model


def optimizer_init_fn():
    # build a learning decay schedule, use this during optimizer building.
    lr_schedule = tf.keras.optimizers.schedules.ExponentialDecay(  # use exponetial learning rate decay during training
        initial_learning_rate= 0.01,
        decay_steps=5000,  # decay the learning rate every given decay_steps
        decay_rate=0.96,   # every time for decay learning rate, decay with decay_rate of exponential decay base.
        staircase=False)
 
    #learning_rate = 1e-3
    optimizer = tf.keras.optimizers.Adam(learning_rate=1e-3, 
                                          epsilon = 1e-8, 
                                          beta_1 = .9, beta_2 = .999)#(learning_rate=lr_schedule, )    
    return optimizer


train_part34(model_init_fn, optimizer_init_fn, num_epochs=num_epochs, is_training=True)

#%%

# running through each sub testing data sets, and conduct the evaluation of the model
test_set = 0
overall_acc=0
for x_np, y_np in test_dset: 
    acc=model.evaluate(x_np, y_np, verbose=0)
    overall_acc+=acc[1]
    test_set +=1
        
    template = 'For test data set {}, loss is {}, test accuracy is {}%.'
    print(template.format(test_set+1, 
                          '%.4f'%(acc[0]*100),
                          '%.2f'%(acc[1]*100)) )    
# print out the overall testing accuracy for all the data sets    
overall_acc = overall_acc/test_set 
template = 'Overall test accuracy is {}%.'
print('\n')
print(template.format('%.2f'%(overall_acc*100)) )   
    
    
#%%   
# summarize history for accuracy
plt.figure()
plt.plot(model.history.history['accuracy'])
plt.plot(model.history.history['val_accuracy'])
plt.title('model accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.show()
# summarize history for loss
plt.plot(model.history.history['loss'])
plt.plot(model.history.history['val_loss'])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.show()    
    
    
    
    
    
    
    