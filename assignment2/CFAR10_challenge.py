# -*- coding: utf-8 -*-
"""
Created on Tue Apr  7 15:36:15 2020

@author: Mingming

This is coming from the end of the Q5TensorFlow.py

Open challenge for CFAR-10 data set

"""



#%%

import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import os


#%

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


#%
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
val_dset   = Dataset(X_val,    y_val,    batch_size=64, shuffle=False)
test_dset  = Dataset(X_test,   y_test,   batch_size=64)
'''
train_dset = Dataset(X_train[0:1000], y_train[0:1000], batch_size=64, shuffle=True)
val_dset   = Dataset(X_val[0:128],    y_val[0:128],    batch_size=64, shuffle=False)
test_dset  = Dataset(X_test[0:500],   y_test[0:500],   batch_size=64)
'''

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

print('Using device: ', device)
#%%

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

        model     = model_init_fn(is_training=is_training)
        optimizer = optimizer_init_fn()
                
        train_loss_results   = []
        train_accuracy_results = []
        val_loss_results     = []
        val_accuracy_results = []
       
        # Compute the loss like we did in Part II
        loss_fn = tf.keras.losses.SparseCategoricalCrossentropy()  # compute cross entropy loss between labels and predictions

        
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

            # after each iteration, tracking the results for later use. 
            train_loss_results.append( float(train_loss.result()) )
            train_accuracy_results.append( float(train_accuracy.result() ))
            val_loss_results.append( float(val_loss.result()) )
            val_accuracy_results.append( float(val_accuracy.result()) )

    plot_results= {'train_loss_results': train_loss_results,
                   'train_accuracy_results': train_accuracy_results,
                   'val_loss_results': val_loss_results,
                   'val_accuracy_results': val_accuracy_results,} 
    
    
    return model, plot_results

                                
#%%

class CustomConvNet(tf.keras.Model):
   
    
    def __init__(self, channel_1=32, channel_2=64, channel_3=32, channel_4=16, 
                 num_classes=10, batch_size=64, is_training=False):
        super(CustomConvNet, self).__init__()
        ############################################################################
        # TODO: Construct a model that performs well on CIFAR-10                   #
        ############################################################################
        # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
      
        initializer = tf.initializers.VarianceScaling(scale=2.0)  # initialize given size of weight matrix, with randomly generated numbers

        
        self.bathnorm =  tf.keras.layers.BatchNormalization(axis=3,
                                                    momentum=0.99, epsilon=0.001,
                                                    beta_initializer='zeros', 
                                                    gamma_initializer='ones',
                                                    moving_mean_initializer='zeros', 
                                                    moving_variance_initializer='ones',
                                                    trainable=is_training)
        


        #self.zeropad1 = tf.keras.layers.ZeroPadding2D(padding=(2,2),data_format= 'channels_last')
        #self.zeropad2 = tf.keras.layers.ZeroPadding2D(padding=(1,1),data_format= 'channels_last')



        self.conv1 = tf.keras.layers.Conv2D(filters=channel_1, # number of conv filters
                                            kernel_size=(5,5), # the filter dimensions (height, width)
                                            strides=(1,1),
                                            #padding='valid',
                                            padding = [[0, 0], [2, 2], [2, 2], [0, 0]],
                                            kernel_initializer=initializer,
                                            data_format= 'channels_last', # channel_last, also the default settings
                                            activation='relu')
    
        self.conv2 = tf.keras.layers.Conv2D(filters=channel_2, # number of conv filters
                                            kernel_size=(3,3), # the filter dimensions (height, width)
                                            strides=(1,1),
                                            #padding='valid',
                                            padding = [[0, 0], [1, 1], [1, 1], [0, 0]],
                                            kernel_initializer=initializer,
                                            data_format= 'channels_last', # channel_last, also the default settings
                                            activation='relu')
       
        self.conv3 = tf.keras.layers.Conv2D(filters=channel_3, # number of conv filters
                                            kernel_size=(3,3), # the filter dimensions (height, width)
                                            strides=(1,1),
                                            padding='valid',
                                            kernel_initializer=initializer,
                                            data_format= 'channels_last', # channel_last, also the default settings
                                            activation='relu')
        
        
        self.conv4 = tf.keras.layers.Conv2D(filters=channel_4, # number of conv filters
                                            kernel_size=(2,2), # the filter dimensions (height, width)
                                            strides=(1,1),
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
        
        #x=self.zeropad1(input_tensor)
        x=self.conv1(input_tensor)    # first conv layer with relu nonlinearity
        x=self.bathnorm(x)
        x=self.dropout(x)  # noise_shape=(x.shape[0], x.shape[1], x.shape[2], 1))  #  (batch_size, height, width, channel), dropout mask will be the same across all channels 
        
        
        #x=self.zeropad2(x)
        x=self.conv2(x)    # 2nd conv layer with relu nonlinearity
        #x=self.bathnorm(x)
        x=self.maxpool1(x)
        x=self.dropout(x)
        
        x=self.conv3(x)    # 3nd conv layer with relu nonlinearity
        #x=self.bathnorm(x)
        x=self.maxpool1(x)
        x=self.dropout(x)
        
        x=self.conv4(x)    # 4nd conv layer with relu nonlinearity
        #x=self.bathnorm(x)
        x=self.maxpool1(x)
        
        x=self.flat(x)
        scores=self.fc(x)  # fully connected layer with softmax nonlinearity
        
        pass

        # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
        ############################################################################
        #                            END OF YOUR CODE                              #
        ############################################################################
        
        return scores
    
#%%

'''
N=57000 number of training examples, batch_size=64, 
57000/64 = 890.625

it will take about 890 iterations to finish one epoch of training
'''

# Constant to control how often we print when training models
print_every = 200   # print results every 100 iterations during training, 
num_epochs  = 10

#device = '/device:GPU:0'   # Change this to a CPU/GPU as you wish!
device = '/cpu:0'        # Change this to a CPU/GPU as you wish!

#model = CustomConvNet()
def model_init_fn(is_training=False):
    model = CustomConvNet(channel_1=32, channel_2=64, channel_3=32, channel_4=16,
                          num_classes=10, batch_size=64, is_training=False)
    return model


def optimizer_init_fn():
    # build a learning decay schedule, use this during optimizer building.
    lr_schedule = tf.keras.optimizers.schedules.ExponentialDecay(  # use exponetial learning rate decay during training
        initial_learning_rate= 0.01,
        decay_steps=5000,  # decay the learning rate every given decay_steps
        decay_rate=0.96,   # every time for decay learning rate, decay with decay_rate of exponential decay base.
        staircase=False)
 
    #learning_rate = 1e-3
    optimizer = tf.keras.optimizers.Adam(learning_rate=1e-3, epsilon = 1e-8, 
                                          beta_1 = .9, beta_2 = .999, decay=3e-8)#(learning_rate=lr_schedule, )    
    return optimizer



model, plot_results=train_part34(model_init_fn, optimizer_init_fn, num_epochs=num_epochs, is_training=True)


#%% Compile the trained model, and conduct the testing

# compile the model, so can used for prediction
model.compile(optimizer=optimizer,
              loss=tf.keras.losses.CategoricalCrossentropy(from_logits=True),
              metrics=['accuracy'])



# running through each sub testing data sets, and conduct the evaluation of the model
test_set = 0
overall_acc=0
for x_np, y_np in test_dset: 
    
    pred_scores = model.predict(x_np)
    pred_locs=np.argmax(pred_scores, axis=1)  # turn predict scores into predict classes
    acc = len( np.where(pred_locs==y_np)[0] )/len(y_np)  # calcualte the predict accuracy for this small testing data sets
    
    overall_acc+=acc
    test_set +=1
    
        
    template = 'For test data set {}, test accuracy is {}%.'
    print(template.format(test_set+1, 
                          '%.2f'%(acc*100)) )    
# print out the overall testing accuracy for all the data sets    
overall_acc = overall_acc/test_set 
template = 'Overall test accuracy is {}%.'
print('\n')
print(template.format('%.2f'%(overall_acc*100)) )   
    

# this will print out the summary information about the model
model.summary()

 
#%%

# summarize history for accuracy
plt.figure()
plt.plot(plot_results['train_accuracy_results'],'.--')
plt.plot(plot_results['val_accuracy_results'],'.--')
plt.title('model accuracy')

plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['train', 'val'], loc='upper left')
plt.show()


# summarize history for loss
plt.figure()
plt.plot(plot_results['train_loss_results'],'.--')
plt.plot(plot_results['val_loss_results'],'.--')
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'val'], loc='upper left')
plt.show()    
    

#%%  save the pre-trained model and load the model to use later
###############################################################################
#                                                                             #
#                    SAVE AND LOAD THE PRETRAINED DATA                        #
#                                                                             #
###############################################################################


# use current time stamps to create a new folder, and save the model into that model
import datetime
dateinfo = str( datetime.datetime.now() )                   # get current time staps 
new_folder = dateinfo[0:4] +  dateinfo[5:7]+ dateinfo[8:10] + '-' + dateinfo[11:13] + dateinfo[14:16]

filepath = 'C:/Users/Mingming/Desktop/Work projects/Machine_learning/mymodels/' + new_folder +'/'  # create a new folder each time for new training
tf.saved_model.save(model, filepath)  # this will create a folder and save models there


loaded = tf.saved_model.load(filepath)
print(list(loaded.signatures.keys()))  # ["serving_default"]



infer = loaded.signatures["serving_default"]
labeling_scores = infer(tf.constant(x_np))['output_1']

pred_locs=np.argmax(labeling_scores, axis=1)  # turn predict scores into predict classes
acc = len( np.where(pred_locs==y_np)[0] )/len(y_np)  # calcualte the predict accuracy for this small testing data sets















