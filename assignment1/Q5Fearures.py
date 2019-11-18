# -*- coding: utf-8 -*-
"""
Created on Mon Oct 21 20:46:49 2019

@author: Mingming
cs231 high level features
"""

import random
import numpy as np
from cs231n.data_utils import load_CIFAR10
import matplotlib.pyplot as plt


#%matplotlib inline
plt.rcParams['figure.figsize'] = (10.0, 8.0) # set default size of plots
plt.rcParams['image.interpolation'] = 'nearest'
plt.rcParams['image.cmap'] = 'gray'


# for auto-reloading extenrnal modules
# see http://stackoverflow.com/questions/1907993/autoreload-of-modules-in-ipython
#%%

from cs231n.features import color_histogram_hsv, hog_feature

def get_CIFAR10_data(num_training=49000, num_validation=1000, num_test=1000):
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
    
    return X_train, y_train, X_val, y_val, X_test, y_test

X_train, y_train, X_val, y_val, X_test, y_test = get_CIFAR10_data()

#%% 
'''
For each image, compute the Histogram of Oriented Gradients (HOG), as well as color hisgrogram using the hue channel in HSV color space.
We form our final feature vector for each image by concatenating
the HOG and color histogram feature vectors.
'''

from cs231n.features import *

num_color_bins = 10 # Number of bins in the color histogram
feature_fns = [hog_feature, lambda img: color_histogram_hsv(img, nbin=num_color_bins)]
X_train_feats = extract_features(X_train, feature_fns, verbose=True)
X_val_feats   = extract_features(X_val, feature_fns)
X_test_feats  = extract_features(X_test, feature_fns)

# Preprocessing: Subtract the mean feature
mean_feat = np.mean(X_train_feats, axis=0, keepdims=True)
X_train_feats -= mean_feat
X_val_feats -= mean_feat
X_test_feats -= mean_feat

# Preprocessing: Divide by standard deviation. This ensures that each feature
# has roughly the same scale.
std_feat = np.std(X_train_feats, axis=0, keepdims=True)
X_train_feats /= std_feat
X_val_feats /= std_feat
X_test_feats /= std_feat

# Preprocessing: Add a bias dimension
X_train_feats = np.hstack([X_train_feats, np.ones((X_train_feats.shape[0], 1))])
X_val_feats = np.hstack([X_val_feats, np.ones((X_val_feats.shape[0], 1))])
X_test_feats = np.hstack([X_test_feats, np.ones((X_test_feats.shape[0], 1))])

#%% TRAIN A SVM CLASSIFIER

# Use the validation set to tune the learning rate and regularization strength

from cs231n.classifiers.linear_classifier import LinearSVM

learning_rates = [1e-9, 1e-8, 1e-7]
regularization_strengths = [5e4, 5e5, 5e6]

results = {}
best_val = -1
best_svm = None

################################################################################
# TODO:                                                                        #
# Use the validation set to set the learning rate and regularization strength. #
# This should be identical to the validation that you did for the SVM; save    #
# the best trained classifer in best_svm. You might also want to play          #
# with different numbers of bins in the color histogram. If you are careful    #
# you should be able to get accuracy of near 0.44 on the validation set.       #
################################################################################
# *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

#learning_range = np.arange(learning_rates[0], learning_rates[1], (learning_rates[1]-learning_rates[0])/20)
#reg_range      = np.arange(regularization_strengths[0], regularization_strengths[1], (regularization_strengths[1]-regularization_strengths[0])/20)

num_iters = 600   # start with a small number
for my_lr in learning_rates:
    for my_reg in regularization_strengths:

        svm = LinearSVM()
        loss_hist = svm.train(X_train_feats, y_train, 
                              learning_rate=my_lr, reg=my_reg,
                              num_iters=num_iters, verbose=True)

        y_train_pred = svm.predict(X_train_feats)
        y_val_pred = svm.predict(X_val_feats)
        train_acc  = np.mean(y_train == y_train_pred)
        val_acc    = np.mean(y_val == y_val_pred)

        results.update( {(my_lr, my_reg): (train_acc, val_acc)} )
         
        if  val_acc > best_val:
            best_val = val_acc
            best_svm = svm # The LinearSVM object that achieved the highest validation rate.

pass

# *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

# Print out results.
for lr, reg in sorted(results):
    train_accuracy, val_accuracy = results[(lr, reg)]
    print('lr %e reg %e train accuracy: %f val accuracy: %f' % (
                lr, reg, train_accuracy, val_accuracy))
    
print('best validation accuracy achieved during cross-validation: %f' % best_val) 

#%%

# Evaluate your trained SVM on the test set
y_test_pred = best_svm.predict(X_test_feats)
test_accuracy = np.mean(y_test == y_test_pred)
print(test_accuracy)

#%%
# An important way to gain intuition about how an algorithm works is to
# visualize the mistakes that it makes. In this visualization, we show examples
# of images that are misclassified by our current system. The first column
# shows images that our system labeled as "plane" but whose true label is
# something other than "plane".

examples_per_class = 8
classes = ['plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']
for cls, cls_name in enumerate(classes):
    idxs = np.where((y_test != cls) & (y_test_pred == cls))[0]
    idxs = np.random.choice(idxs, examples_per_class, replace=False)
    for i, idx in enumerate(idxs):
        plt.subplot(examples_per_class, len(classes), i * len(classes) + cls + 1)
        plt.imshow(X_test[idx].astype('uint8'))
        plt.axis('off')
        if i == 0:
            plt.title(cls_name)
            
plt.show()



#%% NEURAL NETWORK ON IMAGE FEATURES

# Preprocessing: Remove the bias dimension
# Make sure to run this cell only ONCE
print(X_train_feats.shape)  # size(49000, 155)
X_train_feats0 = X_train_feats[:, :-1]
X_val_feats0 = X_val_feats[:, :-1]
X_test_feats0 = X_test_feats[:, :-1]

print(X_train_feats0.shape) # size(49000, 154)

#%%
from cs231n.classifiers.neural_net import TwoLayerNet


input_size  = X_train_feats0.shape[1]  # number of input features in each image
hi_size     = 100  # hidden layer size
num_classes = 10

best_net     = None
best_val_acc = 0

################################################################################
# TODO: Train a two-layer neural network on image features. You may want to    #
# cross-validate various parameters as in previous sections. Store your best   #
# model in the best_net variable.                                              #
################################################################################
# *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****


#hidden_size = [400,600]
#learning_rate=[1e-5, 1e-3]
#regularization=[0.025, 0.95]


learning_rates = [1e-4,  1e-3]
regularization = [0.025,  0.5]

#hidden_range  = np.arange(hidden_size[0], hidden_size[1], (hidden_size[1]-hidden_size[0])/5, dtype='int')
learning_range = np.arange(learning_rates[0], learning_rates[1], (learning_rates[1]-learning_rates[0])/3)
regularization_range = np.arange(regularization[0], regularization[1], (regularization[1]-regularization[0])/3)


total_para_sets = len(learning_range) * len(regularization_range)


#from sklearn import preprocessing 
#X_train_feats0 = preprocessing.normalize(X_train_feats0, axis = 1)   # normalize the data 



#net = TwoLayerNet(input_size, hi_size, num_classes)
#%%
count  = 0  # for tracking iterations, visualization, not required for model
#for hi_size in hidden_range:
for lr in learning_range:
    for reg in regularization_range:
        
            #n = 0
           # lr = learning_range[n] 
            #reg = regularization_range[n] 
        
            count +=1  # track how many times we have calculated the results
            print ('Calculate %d out of %d'% (count, total_para_sets)) 
            
            net = TwoLayerNet(input_size, hi_size, num_classes)
            # Train the network
            stats = net.train(X_train_feats0, y_train, X_val_feats0, y_val,
                        num_iters=1000, batch_size=400, # increase the number of iteration, and batch_size will help to increase number of epoches
                                                        # in the train(), iterations_per_epoch = max(num_train / batch_size, 1), increase batch_size 
                                                            # will decrease iterations_per_epoch, and increase number of epoches
                                                        # iterations // iterations_per_epoch, increase the num_iter, will help increase number of epoches
                        learning_rate=lr, learning_rate_decay=0.95,
                        reg=reg, verbose=True)

            # Predict on the validation set
            #val_acc = (net.predict(X_val_feats0) == y_val).mean()
            val_acc = stats['val_acc_history'][-1] # get the latest validation accuracy during the current training
            print('Current val acc is %.2f' %(val_acc))
            if val_acc > best_val_acc: 
                best_val_acc = val_acc   # update the best validation acc
                best_net     = net       # save the best neural net model
            
pass

# *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

# plt.figure()
# plt.imshow(X_test_feats0, aspect='auto')


# Run your best neural net classifier on the test set. You should be able
# to get more than 55% accuracy.

test_acc = (best_net.predict(X_test_feats0) == y_test).mean()
print(test_acc)

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

#%% take the training out, and debug
################################################################################
##########                         BELOW ARE DEBUGGING       ###################




def loss_func(W1, b1, W2, b2, X, y=None, reg=0.0):
        """
        Compute the loss and gradients for a two layer fully connected neural
        network.

        Inputs:
        - X: Input data of shape (N, D). Each X[i] is a training sample.
        - y: Vector of training labels. y[i] is the label for X[i], and each y[i] is
          an integer in the range 0 <= y[i] < C. This parameter is optional; if it
          is not passed then we only return scores, and if it is passed then we
          instead return the loss and gradients.
        - reg: Regularization strength.

        Returns:
        If y is None, return a matrix scores of shape (N, C) where scores[i, c] is
        the score for class c on input X[i].

        If y is not None, instead return a tuple of:
        - loss: Loss (data loss and regularization loss) for this batch of training
          samples.
        - grads: Dictionary mapping parameter names to gradients of those parameters
          with respect to the loss function; has the same keys as self.params.
        """
        # Unpack variables from the params dictionary
       # W1, b1 = params['W1'], self.params['b1']
       # W2, b2 = self.params['W2'], self.params['b2']
        N, D = X.shape # N number of observations 
                       # D, dimenstion of the input data
                       # C, number of classes, also the output size of the NeuralNet

        '''
        W1 = std * np.random.randn(input_size, hidden_size) # return a matrix with size (input_size, hidden_size),
                                                                           # these initial values are drawn from a standard normal distribution
        b1= np.zeros(hidden_size) 
        W2 = std * np.random.randn(hidden_size, output_size) 
        b2 = np.zeros(output_size)
        '''
        # Compute the forward pass
        scores = np.zeros((N,len(b2)))
        #############################################################################
        # TODO: Perform the forward pass, computing the class scores for the input. #
        # Store the result in the scores variable, which should be an array of      #
        # shape (N, C).                                                             #
        #############################################################################
        # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
         
        #f  = lambda x: 1.0/(1.0 + np.exp(-x)) # define a sigmoid activation function
        #f  = lambda x: np.maximum(0, x) # define a Relu activation function
        '''
        for i in range(N):
            h1 = f(np.dot(W1.T,X[i]) + b1)    # calculate the first hidden layer
            scores[i] = np.dot(W2.T,h1) + b2  # calculate the output layer
       '''
        h1_score = np.dot(X, W1) + b1  # calculate the score first hidden layer
        h1 = np.maximum(0, h1_score)   # Relu to activate
        scores = np.dot(h1, W2) + b2   # calculate the output layer
    
    
        pass

        # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

        # If the targets are not given then jump out, we're done
        if y is None:
            return scores

        # Compute the loss
        loss = None
        #############################################################################
        # TODO: Finish the forward pass, and compute the loss. This should include  #
        # both the data loss and L2 regularization for W1 and W2. Store the result  #
        # in the variable loss, which should be a scalar. Use the Softmax           #
        # classifier loss.                                                          #
        #############################################################################
        # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

        exp_scores = np.exp(scores)
        probs = exp_scores / np.sum(exp_scores, axis=1, keepdims=True) # [N x K]

        # compute the loss: average cross-entropy loss and regularization
        correct_logprobs = -np.log(probs[range(N),y])
        data_loss = np.sum(correct_logprobs)/N
        reg_loss = 0.5*reg*np.sum(W1*W1) + 0.5*reg*np.sum(W2*W2)
        loss = data_loss + reg_loss
        
        pass

        # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

        # Backward pass: compute gradients
        grads = {}
        #############################################################################
        # TODO: Compute the backward pass, computing the derivatives of the weights #
        # and biases. Store the results in the grads dictionary. For example,       #
        # grads['W1'] should store the gradient on W1, and be a matrix of same size #
        #############################################################################
        # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
        
        # BACKPROPAGATION BASED ON THE CALCUALTE BEFORE, AS SHOW BELOW HERE:
        #h1_score = np.dot(X, W1) + b1  # calculate the score first hidden layer
        #h1 = np.maximum(0, h1_score)   # Relu to activate
        #scores = np.dot(h1, W2) + b2   # calculate the output layer
    
        # compute the gradient on scores
        dscores = probs
        dscores[range(N),y] -= 1 # based on the inference of the equation, derevative of the loss function
                                    # respect to scores is equal to the prob for correct class minus one, and  
                                    # prob for incorrect classes stay unchanged.
        dscores /= N # definition of the loss over all examples has a term of 1/N
        
        dW2 = np.dot(h1.T, dscores) # back propagate 2nd layer
        db2 = np.sum(dscores, axis=0, keepdims=True)
        
        dh1 = np.dot(dscores, W2.T) # back propagate hidden layer (also first layer)
        dh1[h1 <= 0] = 0            # backprop the ReLU non-linearity

        dW1 = np.dot(X.T, dh1)      # back propagate 1st layer
        db1 = np.sum(dh1, axis=0, keepdims=True)

        # add regularization gradient contribution
        dW2 += reg * W2
        dW1 += reg * W1
        
        grads.update({'W1': dW1, 'W2': dW2,'b1': db1, 'b2': db2})
        
        pass

        # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

        return loss, grads
    
    
    
def predict(W1, b1, W2, b2, X):
    """
    Use the trained weights of this two-layer network to predict labels for
    data points. For each data point we predict scores for each of the C
    classes, and assign each data point to the class with the highest score.

    Inputs:
    - X: A numpy array of shape (N, D) giving N D-dimensional data points to
      classify.

    Returns:
    - y_pred: A numpy array of shape (N,) giving predicted labels for each of
      the elements of X. For all i, y_pred[i] = c means that X[i] is predicted
      to have class c, where 0 <= c < C.
    """
    y_pred = None
    ###########################################################################
    # TODO: Implement this function; it should be VERY simple!                #
    ###########################################################################
    # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
    
    # retrieve the parameters
    #W1 = self.params['W1'] 
    #b1 = self.params['b1'] 
    #W2 = self.params['W2'] 
    #b2 = self.params['b2'] 
   
    h1_score = np.dot(X, W1) + b1  # calculate the score first hidden layer
    h1 = np.maximum(0, h1_score)   # Relu to activate
    scores = np.dot(h1, W2) + b2   # calculate the output layer

    y_pred = np.argmax(scores, axis=1)  # find the biggest scores in each training example


    pass

    # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

    return y_pred   



 
#%%

#net = TwoLayerNet(input_size, hi_size, num_classes)
std=1e-4
params = {}
W1= std * np.random.randn(input_size, hi_size) # return a matrix with size (input_size, hidden_size),                                                                  # these initial values are drawn from a standard normal distribution
b1 = np.zeros(hi_size) 
W2 = std * np.random.randn(hi_size, num_classes) 
b2 = np.zeros(num_classes)




X = X_train_feats0 
y = y_train 
X_val = X_val_feats0
y_val = y_val
 

learning_rate=1e-3
learning_rate_decay=0.95
reg=0.25
num_iters=1000
batch_size=200
verbose=True

"""
Train this neural network using stochastic gradient descent.

Inputs:
- X: A numpy array of shape (N, D) giving training data.
- y: A numpy array f shape (N,) giving training labels; y[i] = c means that
  X[i] has label c, where 0 <= c < C.
- X_val: A numpy array of shape (N_val, D) giving validation data.
- y_val: A numpy array of shape (N_val,) giving validation labels.
- learning_rate: Scalar giving learning rate for optimization.
- learning_rate_decay: Scalar giving factor used to decay the learning rate
  after each epoch.
- reg: Scalar giving regularization strength.
- num_iters: Number of steps to take when optimizing.
- batch_size: Number of training examples to use per step.
- verbose: boolean; if true print progress during optimization.
"""
num_train = X.shape[0]
iterations_per_epoch = max(num_train // batch_size, 1)

# Use SGD to optimize the parameters in self.model
loss_history = []
train_acc_history = []
val_acc_history = []

plt.figure(1)

for it in range(num_iters):
    X_batch = None
    y_batch = None

    #########################################################################
    # TODO: Create a random minibatch of training data and labels, storing  #
    # them in X_batch and y_batch respectively.                             #
    #########################################################################
    # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
    my_selection = np.random.choice(num_train, size=batch_size) # selection some samples from the training data
    X_batch = X[my_selection]
    y_batch = y[my_selection]

    pass

    # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

    # Compute loss and gradients using the current minibatch
    loss, grads = loss_func( W1, b1, W2, b2, X_batch, y=y_batch, reg=reg)
    loss_history.append(loss)

    #########################################################################
    # TODO: Use the gradients in the grads dictionary to update the         #
    # parameters of the network (stored in the dictionary self.params)      #
    # using stochastic gradient descent. You'll need to use the gradients   #
    # stored in the grads dictionary defined above.                         #
    #########################################################################
    # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
    # perform a parameter update
    W1 += -learning_rate * grads['W1'] 
    b1 += -learning_rate * np.squeeze( grads['b1'] )
    
    W2 += -learning_rate * grads['W2']
    b2 += -learning_rate * np.squeeze( grads['b2'] )
    #b2 += -learning_rate *  grads['b2'] 

    
    pass

    # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

    if verbose and it % 100 == 0:
        print('iteration %d / %d: loss %f' % (it, num_iters, loss))

    # Every epoch, check train and val accuracy and decay learning rate.
    if it % iterations_per_epoch == 0:
        # Check accuracy
        train_acc = ( predict(W1, b1, W2, b2, X_batch) == y_batch).mean()
        val_acc   = ( predict(W1, b1, W2, b2, X_val) == y_val).mean()
        train_acc_history.append(train_acc)
        val_acc_history.append(val_acc)

        # Decay learning rate
        learning_rate *= learning_rate_decay


#%%
print( set(predict(W1, b1, W2, b2, X_val)) ) 
plt.figure()
plt.imshow(X_val[y_val==2] ,  aspect='auto')

predict(W1, b1, W2, b2, X_val[0:6])






h1_score = np.dot(X_val, W1) + b1  # calculate the score first hidden layer
h1 = np.maximum(0, h1_score)   # Relu to activate
scores = np.dot(h1, W2) + b2   # calculate the output layer


plt.figure(10)
for n in range(len(scores)):
    plt.plot(scores[n,:])
plt.plot(b2, 'k')


plt.figure(11)
plt.imshow(X_val ,  aspect='auto')

