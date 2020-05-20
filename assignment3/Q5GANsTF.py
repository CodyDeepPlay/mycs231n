# -*- coding: utf-8 -*-
"""
Created on Fri May 15 14:48:02 2020

@author: Mingming

cs231n Assignment3 Q5, GANs


In this notebook, we will expand our repetoire, 
and build generative models using neural networks. Specifically, 
we will learn how to build models which generate novel images that resemble 
a set of training images.
"""

import tensorflow as tf
import numpy as np
import os

import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

#%matplotlib inline
plt.rcParams['figure.figsize'] = (10.0, 8.0) # set default size of plots
plt.rcParams['image.interpolation'] = 'nearest'
plt.rcParams['image.cmap'] = 'gray'

# A bunch of utility functions

def show_images(images):
    images = np.reshape(images, [images.shape[0], -1])  # images reshape to (batch_size, D)
    sqrtn = int(np.ceil(np.sqrt(images.shape[0])))
    sqrtimg = int(np.ceil(np.sqrt(images.shape[1])))

    fig = plt.figure(figsize=(sqrtn, sqrtn))
    gs = gridspec.GridSpec(sqrtn, sqrtn)
    gs.update(wspace=0.05, hspace=0.05)

    for i, img in enumerate(images):
        ax = plt.subplot(gs[i])
        plt.axis('off')
        ax.set_xticklabels([])
        ax.set_yticklabels([])
        ax.set_aspect('equal')
        plt.imshow(img.reshape([sqrtimg,sqrtimg]))
    return

def preprocess_img(x):
    return 2 * x - 1.0

def deprocess_img(x):
    return (x + 1.0) / 2.0

def rel_error(x,y):
    return np.max(np.abs(x - y) / (np.maximum(1e-8, np.abs(x) + np.abs(y))))

def count_params(model):
    """Count the number of parameters in the current TensorFlow graph """
    param_count = np.sum([np.prod(p.shape) for p in model.weights])
    return param_count

answers = np.load('gan-checks-tf.npz')

NOISE_DIM = 96


#%%
'''
GANs are notoriously finicky with hyperparameters, and also require many training epochs. 
In order to make this assignment approachable without a GPU, we will be working on the MNIST dataset, 
which is 60,000 training and 10,000 test images. Each picture contains a centered image of white digit 
on black background (0 through 9). This was one of the first datasets used to train convolutional neural networks
 and it is fairly easy -- a standard CNN model can easily exceed 99% accuracy. 
Heads-up: Our MNIST wrapper returns images as vectors. That is, they're size (batch, 784). 
If you want to treat them as images, we have to resize them to (batch,28,28) or (batch,28,28,1). 
They are also type np.float32 and bounded [0,1]. 
'''
class MNIST(object):
    def __init__(self, batch_size, shuffle=False):
        """
        Construct an iterator object over the MNIST data
        
        Inputs:
        - batch_size: Integer giving number of elements per minibatch
        - shuffle: (optional) Boolean, whether to shuffle the data on each epoch
        """
        train, _ = tf.keras.datasets.mnist.load_data()
        X, y = train
        X = X.astype(np.float32)/255
        X = X.reshape((X.shape[0], -1))
        self.X, self.y = X, y
        self.batch_size, self.shuffle = batch_size, shuffle

    def __iter__(self):
        N, B = self.X.shape[0], self.batch_size
        idxs = np.arange(N)
        if self.shuffle:
            np.random.shuffle(idxs)
        return iter((self.X[i:i+B], self.y[i:i+B]) for i in range(0, N, B)) 
    

# show a batch
mnist = MNIST(batch_size=16) 
show_images(mnist.X[:16])   
    
#%% implement leaky relu
#-----------------------------------------------------------------------------#
#                               Leaky ReLu                                    #
def leaky_relu(x, alpha=0.01):
    """Compute the leaky ReLU activation function.
    
    Inputs:
    - x: TensorFlow Tensor with arbitrary shape
    - alpha: leak parameter for leaky ReLU
    
    Returns:
    TensorFlow Tensor with the same shape as x
    """
    # TODO: implement leaky ReLU
    # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
    zero_matrx = tf.zeros_like(x)
    smaller_0  = -( tf.maximum(zero_matrx,-x)*alpha)  # find where the elements are smaller than 0, and apply the small weight     
    bigger_0   = tf.maximum(zero_matrx, x)            # preserve the elements are bigger than 0.  
    out_put_x  = tf.add(smaller_0, bigger_0)
    
    
    # for those where elements are smaller than 0, apply the leaky parameter    
    pass

    # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
    return out_put_x


def test_leaky_relu(x, y_true):
    x = tf.constant(x)
    y = leaky_relu(x)
    print('Maximum error: %g'%rel_error(y_true, y))

test_leaky_relu(answers['lrelu_x'], answers['lrelu_y'])
    
    
    
#%%
#-----------------------------------------------------------------------------#
#                              Random Noise                                   #

def sample_noise(batch_size, dim):
    """Generate random uniform noise from -1 to 1.
    
    Inputs:
    - batch_size: integer giving the batch size of noise to generate
    - dim: integer giving the dimension of the noise to generate
    
    Returns:
    TensorFlow Tensor containing uniform noise in [-1, 1] with shape [batch_size, dim]
    """
    # TODO: sample and return noise
    # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
    output_noise = tf.random.uniform(shape=[batch_size, dim], minval=-1, maxval=1)
    pass

    # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
    return output_noise
  
def test_sample_noise():
    batch_size = 3
    dim = 4
    z = sample_noise(batch_size, dim)
    # Check z has the correct shape
    assert z.get_shape().as_list() == [batch_size, dim]
    # Make sure z is a Tensor and not a numpy array
    assert isinstance(z, tf.Tensor)
    # Check that we get different noise for different evaluations
    z1 = sample_noise(batch_size, dim)
    z2 = sample_noise(batch_size, dim)
    assert not np.array_equal(z1, z2)
    # Check that we get the correct range
    assert np.all(z1 >= -1.0) and np.all(z1 <= 1.0)
    print("All tests passed!")
    
test_sample_noise()   

#%%
###############################################################################
#                                Discriminator                                #
###############################################################################

'''
Architecture:
Fully connected layer with input size 784 and output size 256
LeakyReLU with alpha 0.01
Fully connected layer with output size 256
LeakyReLU with alpha 0.01
Fully connected layer with output size 1 
'''

def discriminator():
    """Compute discriminator score for a batch of input images.
    
    Inputs:
    - x: TensorFlow Tensor of flattened input images, shape [batch_size, 784]
    
    Returns:
        
    TensorFlow Tensor with shape [batch_size, 1], containing the score 
    for an image being real for each input image.
    """
    #input_shape = x.shape
    model = tf.keras.Sequential([
        # TODO: implement architecture
        # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
        #tf.keras.layers.Flatten(input_shape=input_shape),
        tf.keras.layers.InputLayer((784)),
        # 1st fully connected layer with leaky relu
        tf.keras.layers.Dense(256, use_bias=True),
        tf.keras.layers.LeakyReLU(alpha=0.01),
       
        # 2nd fully connected layer with leaky relu
        tf.keras.layers.Dense(256, use_bias=True),
        tf.keras.layers.LeakyReLU(alpha=0.01),
        # 2rd fully connected layer
        tf.keras.layers.Dense(1, use_bias=True), #, kernel_initializer='glorot_uniform',bias_initializer='zeros'),

        # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
        ])
    return model


def test_discriminator(true_count=267009):
    model = discriminator()
    cur_count = count_params(model)
    if cur_count != true_count:
        print('Incorrect number of parameters in discriminator. {0} instead of {1}. Check your achitecture.'.format(cur_count,true_count))
    else:
        print('Correct number of parameters in discriminator.')
        
test_discriminator()


#%%

###############################################################################
#                                Generator                                    #
###############################################################################


def generator(noise_dim=NOISE_DIM):
    """Generate images from a random noise vector.
    
    Inputs:
    - z: TensorFlow Tensor of random noise with shape [batch_size, noise_dim]
    
    Returns:
    TensorFlow Tensor of generated images, with shape [batch_size, 784].
    """
    
    model = tf.keras.models.Sequential([
        # TODO: implement architecture
        # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
        #tf.keras.layers.Flatten(input_shape=(NOISE_DIM,)),
        tf.keras.layers.InputLayer(noise_dim),
        tf.keras.layers.Dense(1024, use_bias=True,),
        tf.keras.layers.ReLU(),
        
        tf.keras.layers.Dense(1024, use_bias=True,),
        tf.keras.layers.ReLU(),
        
        tf.keras.layers.Dense(784, use_bias=True, activation='tanh'),

        # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
        ])
    return model


def test_generator(true_count=1858320):
    model = generator(4)
    cur_count = count_params(model)
    if cur_count != true_count:
        print('Incorrect number of parameters in generator. {0} instead of {1}. Check your achitecture.'.format(cur_count,true_count))
    else:
        print('Correct number of parameters in generator.')
        
test_generator()




#%%

###############################################################################
#                                  GAN Loss                                   #
###############################################################################

def discriminator_loss(logits_real, logits_fake):
    """
    Computes the discriminator loss described above.
    
    Inputs:
    - logits_real: Tensor of shape (N, 1) giving scores for the real data.
    - logits_fake: Tensor of shape (N, 1) giving scores for the fake data.
    
    Returns:
    - loss: Tensor containing (scalar) the loss for the discriminator.
    """
    loss = None
    # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
    
    # compute the loss between the true labels and predicted labels.
    # also allows to use the logits/scores.
    cross_entropy = tf.keras.losses.BinaryCrossentropy(from_logits=True)
    
    # data part
    loss_data = cross_entropy(tf.ones(logits_real.shape), logits_real)
    
    # noise part
    loss_noise = cross_entropy(tf.zeros(logits_fake.shape), logits_fake)
     
    loss = loss_data + loss_noise
    pass

    # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
    return loss



def generator_loss(logits_fake):
    """
    Computes the generator loss described above.

    Inputs:
    - logits_fake: PyTorch Tensor of shape (N,) giving scores for the fake data.
    
    Returns:
    - loss: PyTorch Tensor containing the (scalar) loss for the generator.
    """
    loss = None
    # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

    cross_entropy = tf.keras.losses.BinaryCrossentropy(from_logits=True)
    loss = cross_entropy(tf.ones(logits_fake.shape), logits_fake)
    '''
    Generator is generating image with input fake data, the goal is trying to make it fool the discriminator,
    so that the discriminator thinks all the images are in the training set.
    So here use tf.ones() to generate labels for fake data, because we want fake data to pass ALL!
    '''
    
    pass

    # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
    return loss


# Test your GAN loss. Make sure both the generator and discriminator loss are correct. 
# You should see errors less than 1e-8.

def test_discriminator_loss(logits_real, logits_fake, d_loss_true):
    d_loss = discriminator_loss(tf.constant(logits_real),
                                tf.constant(logits_fake))
    print("Maximum error in d_loss: %g"%rel_error(d_loss_true, d_loss))

test_discriminator_loss(answers['logits_real'], answers['logits_fake'],
                        answers['d_loss_true'])


def test_generator_loss(logits_fake, g_loss_true):
    g_loss = generator_loss(tf.constant(logits_fake))
    print("Maximum error in g_loss: %g"%rel_error(g_loss_true, g_loss))

test_generator_loss(answers['logits_fake'], answers['g_loss_true'])


#%%
###############################################################################
#                               Optimizing our loss                           #
###############################################################################

# TODO: create an AdamOptimizer for D_solver and G_solver
def get_solvers(learning_rate=1e-3, beta1=0.5):
    """Create solvers for GAN training.
    
    Inputs:
    - learning_rate: learning rate to use for both solvers
    - beta1: beta1 parameter for both solvers (first moment decay)
    
    Returns:
    - D_solver: instance of tf.optimizers.Adam with correct learning_rate and beta1
    - G_solver: instance of tf.optimizers.Adam with correct learning_rate and beta1
    """
    D_solver = None
    G_solver = None
    # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

    D_solver = tf.keras.optimizers.Adam(learning_rate, beta1)
    G_solver = tf.keras.optimizers.Adam(learning_rate, beta1)
    
    pass

    # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
    return D_solver, G_solver



# a giant helper function
def run_a_gan(D, G, D_solver, G_solver, discriminator_loss, generator_loss,
              show_every=20, print_every=20, batch_size=128, num_epochs=10, noise_size=96):
    """Train a GAN for a certain number of epochs.
    
    Inputs:
    - D: Discriminator model
    - G: Generator model
    - D_solver: an Optimizer for Discriminator
    - G_solver: an Optimizer for Generator
    - generator_loss: Generator loss
    - discriminator_loss: Discriminator loss
    Returns:
        Nothing
    """
    mnist = MNIST(batch_size=batch_size, shuffle=True)
    
    iter_count = 0
    for epoch in range(num_epochs):
        for (x, _) in mnist:
            with tf.GradientTape() as tape:
                real_data = x
                logits_real = D(preprocess_img(real_data))

                g_fake_seed = sample_noise(batch_size, noise_size)
                fake_images = G(g_fake_seed)
                logits_fake = D(tf.reshape(fake_images, [batch_size, 784]))

                d_total_error = discriminator_loss(logits_real, logits_fake)
                d_gradients = tape.gradient(d_total_error, D.trainable_variables)      
                D_solver.apply_gradients(zip(d_gradients, D.trainable_variables))
            
            with tf.GradientTape() as tape:
                g_fake_seed = sample_noise(batch_size, noise_size)
                fake_images = G(g_fake_seed)

                gen_logits_fake = D(tf.reshape(fake_images, [batch_size, 784]))
                g_error = generator_loss(gen_logits_fake)
                g_gradients = tape.gradient(g_error, G.trainable_variables)      
                G_solver.apply_gradients(zip(g_gradients, G.trainable_variables))

            if (iter_count % show_every == 0):
                print('Epoch: {}, Iter: {}, D: {:.4}, G:{:.4}'.format(epoch, iter_count,d_total_error,g_error))
                #imgs_numpy = fake_images.cpu().numpy()
                #show_images(imgs_numpy[0:16])
                #plt.show()
            iter_count += 1
    
    # random noise fed into our generator
    z = sample_noise(batch_size, noise_size)
    # generated images
    G_sample = G(z)
    print('Final images')
    show_images(G_sample[:16])
    plt.show()

#%%

# Make the discriminator
D = discriminator()

# Make the generator
G = generator()

# Use the function you wrote earlier to get optimizers for the Discriminator and the Generator
D_solver, G_solver = get_solvers()

# Run it!
run_a_gan(D, G, D_solver, G_solver, discriminator_loss, generator_loss)


#%%

###############################################################################
#                              Least Squares GAN                              #
###############################################################################


def ls_discriminator_loss(scores_real, scores_fake):
    """
    Compute the Least-Squares GAN loss for the discriminator.
    
    Inputs:
    - scores_real: Tensor of shape (N, 1) giving scores for the real data.
    - scores_fake: Tensor of shape (N, 1) giving scores for the fake data.
    
    Outputs:
    - loss: A Tensor containing the loss.
    """
    loss = None
    # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
    
    loss_noise = 0.5*tf.reduce_mean((scores_fake)**2)    
    loss_data  = 0.5*tf.reduce_mean((scores_real-1)**2)
    loss = loss_noise + loss_data

    pass

    # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
    return loss




def ls_generator_loss(scores_fake):
    """
    Computes the Least-Squares GAN loss for the generator.
    
    Inputs:
    - scores_fake: Tensor of shape (N, 1) giving scores for the fake data.
    
    Outputs:
    - loss: A Tensor containing the loss.
    """
    loss = None
    # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

    loss = 0.5*tf.reduce_mean((scores_fake-1)**2)

     
    pass

    # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
    return loss

# Test your LSGAN loss. You should see errors less than 1e-8.

def test_lsgan_loss(score_real, score_fake, d_loss_true, g_loss_true):
    
    d_loss = ls_discriminator_loss(tf.constant(score_real), tf.constant(score_fake))
    g_loss = ls_generator_loss(tf.constant(score_fake))
    print("Maximum error in d_loss: %g"%rel_error(d_loss_true, d_loss))
    print("Maximum error in g_loss: %g"%rel_error(g_loss_true, g_loss))

test_lsgan_loss(answers['logits_real'], answers['logits_fake'],
                answers['d_loss_lsgan_true'], answers['g_loss_lsgan_true'])


#%%
# Make the discriminator
D = discriminator()

# Make the generator
G = generator()

# Use the function you wrote earlier to get optimizers for the Discriminator and the Generator
D_solver, G_solver = get_solvers()

# Run it!
run_a_gan(D, G, D_solver, G_solver, ls_discriminator_loss, ls_generator_loss)







#%%

###############################################################################
#                             Deep Convolutional GANs                         #
###############################################################################



def discriminator():
    """Compute discriminator score for a batch of input images.
    
    Inputs:
    - x: TensorFlow Tensor of flattened input images, shape [batch_size, 784]
    
    Returns:
    TensorFlow Tensor with shape [batch_size, 1], containing the score 
    for an image being real for each input image.
    """
        
    model = tf.keras.models.Sequential([
        # TODO: implement architecture
        # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
        tf.keras.layers.InputLayer( input_shape=(1,784) ),
        tf.keras.layers.Reshape( (28,28,1) ),
        tf.keras.layers.Conv2D(filters=32,             # number of conv filters
                                     kernel_size=(5,5),      # the filter dimensions along time domain, the other dimension set to 1
                                     strides=1,
                                     padding='valid', use_bias=True),
        tf.keras.layers.LeakyReLU(alpha=0.01),
        tf.keras.layers.MaxPool2D(pool_size=(2, 2), strides=2),
        tf.keras.layers.Conv2D(filters=64,             # number of conv filters
                               kernel_size=(5,5),      # the filter dimensions along time domain, the other dimension set to 1
                               strides=1,
                               padding='valid', use_bias=True),
        tf.keras.layers.LeakyReLU(alpha=0.01),
        tf.keras.layers.MaxPool2D(pool_size=(2, 2), strides=2),
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(4*4*64, use_bias=True),
        tf.keras.layers.LeakyReLU(alpha=0.01),

        tf.keras.layers.Dense(1, use_bias=True),
        # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
        ])
    return model

model = discriminator()
test_discriminator(1102721)



def generator(noise_dim=NOISE_DIM):
    """Generate images from a random noise vector.
    
    Inputs:
    - z: TensorFlow Tensor of random noise with shape [batch_size, noise_dim]
    
    Returns:
    TensorFlow Tensor of generated images, with shape [batch_size, 784].
    """
    model = tf.keras.models.Sequential()
    # TODO: implement architecture
    # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
    
    model.add(tf.keras.layers.InputLayer(noise_dim))
    model.add(tf.keras.layers.Dense(1024, use_bias=True))
    model.add(tf.keras.layers.ReLU())
    model.add(tf.keras.layers.BatchNormalization(trainable=True))
    model.add(tf.keras.layers.Dense(7*7*128, use_bias=True))
    model.add(tf.keras.layers.ReLU())
    model.add(tf.keras.layers.BatchNormalization(trainable=True))
    model.add(tf.keras.layers.Reshape((7,7,128)))
    model.add(tf.keras.layers.Conv2DTranspose(filters=64, padding='same',
                                        kernel_size = (4,4),
                                        strides=2, use_bias=True))
    model.add(tf.keras.layers.ReLU())
    model.add(tf.keras.layers.BatchNormalization(trainable=True))
    model.add(tf.keras.layers.Conv2DTranspose(filters=1, padding='same',
                                        kernel_size = (4,4),
                                        strides=2, use_bias=True))
    model.add(tf.keras.layers.Activation('tanh') )

    # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
    return model
test_generator(6595521)



#%% Train and evaluate a DCGAN
# Make the discriminator
D = discriminator()

# Make the generator
G = generator()

# Use the function you wrote earlier to get optimizers for the Discriminator and the Generator
D_solver, G_solver = get_solvers()

# Run it!
run_a_gan(D, G, D_solver, G_solver, discriminator_loss, generator_loss, num_epochs=3)


