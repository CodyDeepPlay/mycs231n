# -*- coding: utf-8 -*-
"""
Created on Wed Aug 21 22:15:00 2019

@author: Mingming

This is the assignment that Mingming conduct for KNN
"""

import random
import numpy as np
#import os
#cwd = os.getcwd()
#import sys
#sys.path.insert(0, cwd+'\\cs231n')  # add the path for folder 'cs231n', where a lot of dependent function is located.
#from data_utils import load_CIFAR10    # subfolder located in the assignment folder


from cs231n.data_utils import load_CIFAR10

import matplotlib.pyplot as plt

# This is a bit of magic to make matplotlib figures appear inline in the notebook
# rather than in a new window.
#%matplotlib inline
plt.rcParams['figure.figsize'] = (10.0, 8.0) # set default size of plots
plt.rcParams['image.interpolation'] = 'nearest'
plt.rcParams['image.cmap'] = 'gray'

# Some more magic so that the notebook will reload external python modules;
# see http://stackoverflow.com/questions/1907993/autoreload-of-modules-in-ipython
#%load_ext autoreload
#%autoreload 2

#%%
# Load the raw CIFAR-10 data.
cifar10_dir = 'cs231n/datasets/cifar-10-batches-py'

# Cleaning up variables to prevent loading data multiple times (which may cause memory issue)
try:
   del X_train, y_train
   del X_test, y_test
   print('Clear previously loaded data.')
except:
   pass

X_train, y_train, X_test, y_test = load_CIFAR10(cifar10_dir)

# As a sanity check, we print out the size of the training and test data.
print('Training data shape: ', X_train.shape)
print('Training labels shape: ', y_train.shape)
print('Test data shape: ', X_test.shape)
print('Test labels shape: ', y_test.shape)
#%%
# Visualize some examples from the dataset.
# We show a few examples of training images from each class.
classes = ['plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']
num_classes = len(classes)
samples_per_class = 7
for y, cls in enumerate(classes):
    idxs = np.flatnonzero(y_train == y)
    idxs = np.random.choice(idxs, samples_per_class, replace=False)
    for i, idx in enumerate(idxs):
        plt_idx = i * num_classes + y + 1
        plt.subplot(samples_per_class, num_classes, plt_idx)
        plt.imshow(X_train[idx].astype('uint8'))
        plt.axis('off')
        if i == 0:
            plt.title(cls)
plt.show()
#%%
# Subsample the data for more efficient code execution in this exercise
num_training = 5000
mask = list(range(num_training))
X_train = X_train[mask]
y_train = y_train[mask]

num_test = 500
mask = list(range(num_test))
X_test = X_test[mask]
y_test = y_test[mask]

# Reshape the image data into rows
X_train = np.reshape(X_train, (X_train.shape[0], -1))
X_test = np.reshape(X_test, (X_test.shape[0], -1))
print(X_train.shape, X_test.shape)

#%%
#import sys

#new = cwd+'\\cs231n'
#sys.path.intert(0, new+'\\classifier') # where the classifier is stored

from cs231n.classifiers.k_nearest_neighbor import KNearestNeighbor


# Create a kNN classifier instance. 
# Remember that training a kNN classifier is a noop: 
# the Classifier simply remembers the data and does no further processing 
classifier = KNearestNeighbor()
classifier.train(X_train, y_train)


#% calcualte the distance
# Open cs231n/classifiers/k_nearest_neighbor.py and implement
# compute_distances_two_loops.

# Test your implementation:
dists = classifier.compute_distances_no_loops(X_test)
#dists = classifier.compute_distances_two_loops(X_test)

#dists = classifier.compute_distances_one_loop(X_test)
#%%
# We can visualize the distance matrix: each row is a single test example and
# its distances to training examples
plt.figure()
plt.subplot(2,1,1)
plt.imshow(dists, interpolation='none')
plt.ylabel('one loop')
plt.show()
plt.subplot(2,1,2)
plt.imshow(dists, interpolation='none')
plt.show()
plt.ylabel('no loop')
#%%
# Now implement the function predict_labels and run the code below:
# We use k = 1 (which is Nearest Neighbor).
y_test_pred = classifier.predict_labels(dists, k=5)

# Compute and print the fraction of correctly predicted examples
num_correct = np.sum(y_test_pred == y_test)
accuracy = float(num_correct) / num_test
print('Got %d / %d correct => accuracy: %f' % (num_correct, num_test, accuracy))

#%%


num_folds = 5
k_choices = [1, 3, 5, 8, 10, 12, 15, 20, 50, 100]

X_train_folds = []
y_train_folds = []
################################################################################
# TODO:                                                                        #
# Split up the training data into folds. After splitting, X_train_folds and    #
# y_train_folds should each be lists of length num_folds, where                #
# y_train_folds[i] is the label vector for the points in X_train_folds[i].     #
# Hint: Look up the numpy array_split function.                                #
################################################################################
# *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****


myfolds =  np.array_split(np.arange( len(X_train)), num_folds)
X_train_folds = X_train[myfolds,:]
y_train_folds = (y_train[:,np.newaxis])[myfolds,:]


# *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

# A dictionary holding the accuracies for different values of k that we find
# when running cross-validation. After running cross-validation,
# k_to_accuracies[k] should be a list of length num_folds giving the different
# accuracy values that we found when using that value of k.
    

    

################################################################################
# TODO:                                                                        #
# Perform k-fold cross validation to find the best value of k. For each        #
# possible value of k, run the k-nearest-neighbor algorithm num_folds times,   #
# where in each case you use all but one of the folds as training data and the #
# last fold as a validation set. Store the accuracies for all fold and all     #
# values of k in the k_to_accuracies dictionary.                               #
################################################################################
# *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
k_to_accuracies = {}

num_valid = len(myfolds[0])  # in each fold, how many examples are there

for k in k_choices:
    print('K is ' + str(k)+ '\n')
    k_accuracy = [] # intialize the space to store the accuracy for a certain fold cross validation
    for a_fold in range(num_folds):
        print('Current fold is ' + str(a_fold))
        # take 1 fold within myfolds as the validation data 
        X_valid_fold = X_train_folds[a_fold,:]
        y_valid_fold = np.squeeze(y_train_folds[a_fold])
        
        # take the rest of the 4 folds in myfolds as the training data
        X_train_fold = np.delete(X_train, myfolds[a_fold], axis = 0)
        y_train_fold = np.delete(y_train, myfolds[a_fold], axis = 0)
        
        
        classifier.train(X_train_fold, y_train_fold)
        dists = classifier.compute_distances_no_loops(X_valid_fold)
        y_valid_pred = classifier.predict_labels(dists, k=k)
        num_correct = np.sum(y_valid_pred == y_valid_fold)
        k_accuracy.append( float(num_correct) / num_valid)  # store the accuracy for all cross validation for a certaion K
    
    #best = max(k_accuracy)
    k_to_accuracies.update({k: k_accuracy})        
    
    pass

pass

# *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
#%%
# Print out the computed accuracies
all_ks = []
all_acccuracy = []
for k in sorted(k_to_accuracies):
    for accuracy in k_to_accuracies[k]:
        all_ks.append(k)
        all_acccuracy.append(accuracy)
        print('k = %d, accuracy = %f' % (k, accuracy))

import matplotlib.pyplot as plt
plt.figure()
plt.plot(all_ks, all_acccuracy, '--.')
best_k = all_ks[np.argmax(all_acccuracy)]
print('The best parameters for KNN is: k=' + str(best_k) )
plt.xlabel('The parameter value for K')
plt.ylabel('Validattion accuracy with the same data set')






