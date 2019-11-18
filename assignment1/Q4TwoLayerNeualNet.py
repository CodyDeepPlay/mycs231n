# -*- coding: utf-8 -*-
"""
Created on Sun Oct 13 21:18:53 2019

@author: Mingming
cs231 Q4, two-layer neural network
"""

# A bit of setup

import numpy as np
import matplotlib.pyplot as plt

from cs231n.classifiers.neural_net import TwoLayerNet

#%matplotlib inline
plt.rcParams['figure.figsize'] = (10.0, 8.0) # set default size of plots
plt.rcParams['image.interpolation'] = 'nearest'
plt.rcParams['image.cmap'] = 'gray'

# for auto-reloading external modules
# see http://stackoverflow.com/questions/1907993/autoreload-of-modules-in-ipython
#%load_ext autoreload
#%autoreload 2

def rel_error(x, y):
    """ returns relative error """
    return np.max(np.abs(x - y) / (np.maximum(1e-8, np.abs(x) + np.abs(y))))


#%%
# Create a small net and some toy data to check your implementations.
# Note that we set the random seed for repeatable experiments.

input_size = 4   # number of features
hidden_size = 10 # number of neurons in hidden layer
num_classes = 3  # number of classes
num_inputs = 5   # number of observations in the input data
output_size=num_classes

def init_toy_model():
    np.random.seed(0)
    return TwoLayerNet(input_size, hidden_size, output_size, std=1e-1)

def init_toy_data():
    np.random.seed(1)
    X = 10 * np.random.randn(num_inputs, input_size)
    y = np.array([0, 1, 2, 2, 1])
    return X, y

net = init_toy_model()
X, y = init_toy_data()


#%%  neural_net.py TwoLayerNet.loss() t0 calculate the scores
scores = net.loss(X)
print('Your scores:')
print(scores)
print()
print('correct scores:')
correct_scores = np.asarray([
  [-0.81233741, -1.27654624, -0.70335995],
  [-0.17129677, -1.18803311, -0.47310444],
  [-0.51590475, -1.01354314, -0.8504215 ],
  [-0.15419291, -0.48629638, -0.52901952],
  [-0.00618733, -0.12435261, -0.15226949]])
print(correct_scores)
print()

# The difference should be very small. We get < 1e-7
print('Difference between your scores and correct scores:')
print(np.sum(np.abs(scores - correct_scores)))

#%% compute the loss
loss, _ = net.loss(X, y, reg=0.05)
correct_loss = 1.30378789133

# should be very small, we get < 1e-12
print('Difference between your loss and correct loss:')
print(np.sum(np.abs(loss - correct_loss)))


from cs231n.gradient_check import eval_numerical_gradient

#%% calculate the gradient

# Use numeric gradient checking to check your implementation of the backward pass.
# If your implementation is correct, the difference between the numeric and
# analytic gradients should be less than 1e-8 for each of W1, W2, b1, and b2.

loss, grads = net.loss(X, y, reg=0.05)

# these should all be less than 1e-8 or so
for param_name in grads:
    f = lambda W: net.loss(X, y, reg=0.05)[0]
    param_grad_num = eval_numerical_gradient(f, net.params[param_name], verbose=False)
    print('%s max relative error: %e' % (param_name, rel_error(param_grad_num, grads[param_name])))


#%% train the network

net = init_toy_model()
stats = net.train(X, y, X, y,
            learning_rate=1e-1, reg=5e-6,
            num_iters=100, verbose=False)

print('Final training loss: ', stats['loss_history'][-1])

# plot the loss history
plt.plot(stats['loss_history'])
plt.xlabel('iteration')
plt.ylabel('training loss')
plt.title('Training Loss history')
plt.show()

#%% load the data

from cs231n.data_utils import load_CIFAR10

def get_CIFAR10_data(num_training=49000, num_validation=1000, num_test=1000):
    """
    Load the CIFAR-10 dataset from disk and perform preprocessing to prepare
    it for the two-layer neural net classifier. These are the same steps as
    we used for the SVM, but condensed to a single function.  
    """
    # Load the raw CIFAR-10 data
    cifar10_dir = 'cs231n/datasets/cifar-10-batches-py'
    
    # Cleaning up variables to prevent loading data multiple times (which may cause memory issue)
    try:
       del X_train, y_train
       del X_test, y_test
       print('Clear previously loaded data.')
    except:
       pass

    X_train, y_train, X_test, y_test = load_CIFAR10(cifar10_dir)
        
    # Subsample the data
    mask = list(range(num_training, num_training + num_validation))
    X_val = X_train[mask]
    y_val = y_train[mask]
    mask = list(range(num_training))
    X_train = X_train[mask]
    y_train = y_train[mask]
    mask = list(range(num_test))
    X_test = X_test[mask]
    y_test = y_test[mask]

    # Normalize the data: subtract the mean image
    mean_image = np.mean(X_train, axis=0)
    X_train -= mean_image
    X_val -= mean_image
    X_test -= mean_image

    # Reshape data to rows
    X_train = X_train.reshape(num_training, -1)
    X_val = X_val.reshape(num_validation, -1)
    X_test = X_test.reshape(num_test, -1)

    return X_train, y_train, X_val, y_val, X_test, y_test


# Invoke the above function to get our data.
X_train, y_train, X_val, y_val, X_test, y_test = get_CIFAR10_data()
print('Train data shape: ', X_train.shape)
print('Train labels shape: ', y_train.shape)
print('Validation data shape: ', X_val.shape)
print('Validation labels shape: ', y_val.shape)
print('Test data shape: ', X_test.shape)
print('Test labels shape: ', y_test.shape)

#%%  Train a network, we will use SGD, and we will adjust the learning rate with an 
# exponential learning rate schedule as optimization proceeds.

input_size = 32 * 32 * 3
hidden_size = 50
num_classes = 10
net = TwoLayerNet(input_size, hidden_size, num_classes)

# Train the network
stats = net.train(X_train, y_train, X_val, y_val,
            num_iters=1000, batch_size=200,
            learning_rate=1e-4, learning_rate_decay=0.95,
            reg=0.25, verbose=True)

# Predict on the validation set
val_acc = (net.predict(X_val) == y_val).mean()
print('Validation accuracy: ', val_acc)


#%% Debug the training

# Plot the loss function and train / validation accuracies
plt.subplot(2, 1, 1)
plt.plot(stats['loss_history'])
plt.title('Loss history')
plt.xlabel('Iteration')
plt.ylabel('Loss')

plt.subplot(2, 1, 2)
plt.plot(stats['train_acc_history'], label='train')
plt.plot(stats['val_acc_history'], label='val')
plt.title('Classification accuracy history')
plt.xlabel('Epoch')
plt.ylabel('Classification accuracy')
plt.legend()
plt.show()


#%% TUNE the hyperparameters 
# hidden layer size
# learning rate
# number of training epoches
# regularization strength


best_net = None # store the best model into this 
best_val_acc = 0
#################################################################################
# TODO: Tune hyperparameters using the validation set. Store your best trained  #
# model in best_net.                                                            #
#                                                                               #
# To help debug your network, it may help to use visualizations similar to the  #
# ones we used above; these visualizations will have significant qualitative    #
# differences from the ones we saw above for the poorly tuned network.          #
#                                                                               #
# Tweaking hyperparameters by hand can be fun, but you might find it useful to  #
# write code to sweep through possible combinations of hyperparameters          #
# automatically like we did on the previous exercises.                          #
#################################################################################
# *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****


hidden_size = [10,210]
learning_rate=[1e-5, 1e-3]
regularization=[0.025, 0.75]

hidden_range  = np.arange(hidden_size[0], hidden_size[1], (hidden_size[1]-hidden_size[0])/2, dtype='int')
learning_range = np.arange(learning_rate[0], learning_rate[1], (learning_rate[1]-learning_rate[0])/2)
regularization_range     = np.arange(regularization[0], regularization[1], (regularization[1]-regularization[0])/2)


input_size = 32 * 32 * 3
num_classes = 10
count=0

total_para_sets = len(learning_range) * len(hidden_range)* len(regularization_range)



for hi_size in hidden_range:
    for lr in learning_range:
        for reg in regularization_range:
            
                count +=1  # track how many times we have calculated the results
                print ('Calculate %d out of %d'% (count, total_para_sets)) 
                
                net = TwoLayerNet(input_size, hi_size, num_classes)
                # Train the network
                stats = net.train(X_train, y_train, X_val, y_val,
                            num_iters=2000, batch_size=400, # increase the number of iteration, and batch_size will help to increase number of epoches
                                                            # in the train(), iterations_per_epoch = max(num_train / batch_size, 1), increase batch_size 
                                                                # will decrease iterations_per_epoch, and increase number of epoches
                                                            # iterations // iterations_per_epoch, increase the num_iter, will help increase number of epoches
                            learning_rate=lr, learning_rate_decay=0.95,
                            reg=reg, verbose=True)

                # Predict on the validation set
                val_acc = (net.predict(X_val) == y_val).mean()

                if val_acc > best_val_acc: 
                    best_val_acc = val_acc   # update tje best validation acc
                    best_net     = net       # save the best neural net model
                
                    
pass

# *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

test_acc = (best_net.predict(X_test) == y_test).mean()
print('Test accuracy: ', test_acc)

#%% save the model for later use
import pickle
pkl_filename = "best_NN_model.pkl"  # best neural net model
with open(pkl_filename, 'wb') as file:
    pickle.dump(best_net, file)


#%% reload the model, and use it for prediction
# Load from file
with open(pkl_filename, 'rb') as file:
    pickle_model = pickle.load(file)

#% Visualize and debug the training
test_acc = (pickle_model.predict(X_test) == y_test).mean()
print('Test accuracy: ', test_acc)


#%%

# Plot the loss function and train / validation accuracies
plt.subplot(2, 1, 1)
plt.plot(stats['loss_history'])
plt.title('Loss history')
plt.xlabel('Iteration')
plt.ylabel('Loss')

plt.subplot(2, 1, 2)
plt.plot(stats['train_acc_history'], label='train')
plt.plot(stats['val_acc_history'], label='val')
plt.title('Classification accuracy history')
plt.xlabel('Epoch')
plt.ylabel('Classification accuracy')
plt.legend()
plt.show()










