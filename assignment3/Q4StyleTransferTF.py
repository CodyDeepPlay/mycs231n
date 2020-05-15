# -*- coding: utf-8 -*-
"""
Created on Tue May 12 17:35:17 2020

@author: Mingming


cs231 assignment3 Q4 style transfer
"""

import os
import numpy as np
from scipy.misc import imread, imresize
import matplotlib.pyplot as plt
import tensorflow as tf

# Helper functions to deal with image preprocessing
from cs231n.image_utils import load_image, preprocess_image, deprocess_image
from cs231n.classifiers.squeezenet import SqueezeNet


def rel_error(x,y):
    return np.max(np.abs(x - y) / (np.maximum(1e-8, np.abs(x) + np.abs(y))))

# Older versions of scipy.misc.imresize yield different results
# from newer versions, so we check to make sure scipy is up to date.
def check_scipy():
    import scipy
    version = scipy.__version__.split('.')
    if int(version[0]) < 1:
        assert int(version[1]) >= 16, "You must install SciPy >= 0.16.0 to complete this notebook."

check_scipy()

#%%


# Load pretrained SqueezeNet model
SAVE_PATH = 'cs231n/datasets/squeezenet.ckpt'
if not os.path.exists(SAVE_PATH + ".index"):
    raise ValueError("You need to download SqueezeNet!")

model=SqueezeNet()
model.load_weights(SAVE_PATH)
model.trainable=False

filename = 'styles/tubingen.jpg'
X = load_image(filename, size=192)

# Load data for testing
content_img_test = preprocess_image(load_image('styles/tubingen.jpg', size=192))[None]
style_img_test = preprocess_image(load_image('styles/starry_night.jpg', size=192))[None]
answers = np.load('style-transfer-checks-tf.npz')

#%%

###############################################################################
#                                Content Loss                                 #
###############################################################################

def content_loss(content_weight, content_current, content_original):
    """
    Compute the content loss for style transfer.
    
    Inputs:
    - content_weight: scalar constant we multiply the content_loss by.
    - content_current: features of the current image, Tensor with shape [1, height, width, channels]
    - content_original: features of the content image, Tensor with shape [1, height, width, channels]
    
    Returns:
    - scalar content loss
    """
    # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
    #[N, height, width, channels] = content_current.shape  # here N should be 1    
    # current_vectorized   = tf.reshape( content_current,  [height*width, channels] )
    # original_vectorized  = tf.reshape( content_original, [height*width, channels] )   
    # contentloss = content_weight* tf.reduce_sum( (current_vectorized - original_vectorized)**2 )
 
    diff = content_current - content_original
    diff = diff*diff
    contentloss = content_weight* tf.reduce_sum(diff)
              
    pass
    # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
    return contentloss


# We provide this helper code which takes an image, a model (cnn), and returns a list of
# feature maps, one per layer.
def extract_features(x, cnn):
    """
    Use the CNN to extract features from the input image x.
    
    Inputs:
    - x: A Tensor of shape (N, H, W, C) holding a minibatch of images that
      will be fed to the CNN.
    - cnn: A Tensorflow model that we will use to extract features.
    
    Returns:
    - features: A list of feature for the input images x extracted using the cnn model.
      features[i] is a Tensor of shape (N, H_i, W_i, C_i); recall that features
      from different layers of the network may have different numbers of channels (C_i) and
      spatial dimensions (H_i, W_i).
    """
    features = []
    prev_feat = x
    for i, layer in enumerate(cnn.net.layers[:-2]):
        next_feat = layer(prev_feat)
        features.append(next_feat)
        prev_feat = next_feat
    return features


def content_loss_test(correct):
    content_layer = 2
    content_weight = 6e-2
    c_feats = extract_features(content_img_test, model)[content_layer]
    bad_img = tf.zeros(content_img_test.shape)
    feats   = extract_features(bad_img, model)[content_layer]
    student_output = content_loss(content_weight, c_feats, feats)
    error = rel_error(correct, student_output)
    print('Maximum error is {:.8f}'.format(error))

content_loss_test(answers['cl_out'])


#%% 
###############################################################################
#                                Style Loss                                   #
###############################################################################


def gram_matrix(features, normalize=True):
    """
    Compute the Gram matrix from features.
    
    Inputs:
    - features: Tensor of shape (1, H, W, C) giving features for
      a single image.
    - normalize: optional, whether to normalize the Gram matrix
        If True, divide the Gram matrix by the number of neurons (H * W * C)
    
    Returns:
    - gram: Tensor of shape (C, C) giving the (optionally normalized)
      Gram matrices for the input image.
    """
    # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

    '''
    [N, height, width, channels] = features.shape  # here N should be 1
    f_vectorized  = tf.reshape( features,  [height*width, channels] )
    matrx = np.zeros((channels, channels))
    
    for a_col in range(channels):     
        a = f_vectorized[:, a_col]            # get an column vector from the vectorized matrix       
        f_col_matrx    = np.ones_like(f_vectorized)*a[:,np.newaxis]  # copy this column to each column, so that this matrix has the same size with vertorized matrix, but only repeat the selected column
        all_cols       = f_col_matrx* f_vectorized                   # all the columns were multiplied by the selected column, the matrix operation gets rid of using an loop to compute
        matrx[a_col,:] = np.sum(all_cols, axis=0)                    # calculate the multiplied results for each column
        #matrx[a_col,:] = tf.math.reduce_sum(all_cols, axis=0)   
       
    if normalize:  matrx = matrx/(height*width*channels)
    '''
    
    s = tf.shape(features)
    H = s[1]
    W = s[2]
    C = s[3]
    F = tf.reshape(features, (H * W, C))
    G = tf.matmul(F, F, transpose_a=True)
    if normalize: matrx = tf.divide(G, tf.cast(H * W * C, G.dtype))  # make sure the tensor is the tf tensor type. 
 
        
    pass
    return matrx
    # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****


def gram_matrix_test(correct):
    gram = gram_matrix(extract_features(style_img_test, model)[4]) ### 4 instead of 5 - second MaxPooling layer
    error = rel_error(correct, gram)
    print('Maximum error is {:.8f}'.format(error))

gram_matrix_test(answers['gm_out'])



def style_loss(feats, style_layers, style_targets, style_weights):
    """
    Computes the style loss at a set of layers.
    
    Inputs:
    - feats: list of the features at every layer of the current image, as produced by the extract_features function.
    
    - style_layers: List of layer indices into feats giving the layers to include in the style loss.
    
    - style_targets: List of the same length as style_layers, where style_targets[i] is
      a Tensor giving the Gram matrix of the source style image computed at layer style_layers[i].
    
    - style_weights: List of the same length as style_layers, where style_weights[i]
      is a scalar giving the weight for the style loss at layer style_layers[i].
      
    Returns:
    - style_loss: A Tensor containing the scalar style loss.
    """
    # Hint: you can do this with one for loop over the style layers, and should
    # not be short code (~5 lines). You will need to use your gram_matrix function.
    # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****   
    styleloss = 0
    for i in range(len(style_layers)):      
        current_img_matrx = gram_matrix(feats[style_layers[i]], normalize=True)   # at a given layer, Gram Matrix from the feature map of current image
        styleloss += style_weights[i]* tf.reduce_sum( (current_img_matrx - style_targets[i])**2 )
         
    pass

    # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
    return styleloss



def style_loss_test(correct):
    style_layers = [0, 3, 5, 6]
    style_weights = [300000, 1000, 15, 3]
    
    c_feats = extract_features(content_img_test, model)
    feats = extract_features(style_img_test, model)
    style_targets = []
    for idx in style_layers:
        style_targets.append(gram_matrix(feats[idx]))
                             
    s_loss = style_loss(c_feats, style_layers, style_targets, style_weights)
    error = rel_error(correct, s_loss)
    print('Error is {:.6f}'.format(error))

style_loss_test(answers['sl_out'])

#%%
###############################################################################
#                       Total variation regularization                        #
###############################################################################

'''
It turns out that it's helpful to also encourage smoothness in the image. 
We can do this by adding another term to our loss that penalizes wiggles or 
"total variation" in the pixel values. 
'''


def tv_loss(img, tv_weight):
    """
    Compute total variation loss.
    
    Inputs:
    - img: Tensor of shape (1, H, W, 3) holding an input image.
    - tv_weight: Scalar giving the weight w_t to use for the TV loss.
    
    Returns:
    - loss: Tensor holding a scalar giving the total variation loss
      for img weighted by tv_weight.
    """
    # Your implementation should be vectorized and not require any loops!
    # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

    (N, H, W, C)    = img.shape
    L_H = tf.reduce_sum( (img[:,0:H-1, :,:] - img[:,1:H, :,:])**2 )
    L_W = tf.reduce_sum( (img[:,:,0:W-1,:]  - img[:,:,1:W,:])**2 )
    
    LW = tv_weight*(L_H+L_W)
        
    pass
    # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

    return LW



def tv_loss_test(correct):
    tv_weight = 2e-2
    t_loss = tv_loss(content_img_test, tv_weight)
    error = rel_error(correct, t_loss)
    print('Error is {:.6f}'.format(error))

tv_loss_test(answers['tv_out'])

#%%
###############################################################################
#                                Style transfer                               #
###############################################################################


'''
Lets put it all together and make some beautiful images! The style_transfer function 
below combines all the losses you coded up above and optimizes for an image that 
minimizes the total loss.
'''

def style_transfer(content_image, style_image, image_size, style_size, content_layer, content_weight,
                   style_layers, style_weights, tv_weight, init_random = False):
    """Run style transfer!
    
    Inputs:
    - content_image: filename of content image
    - style_image: filename of style image
    - image_size: size of smallest image dimension (used for content loss and generated image)
    - style_size: size of smallest style image dimension
    - content_layer: layer to use for content loss
    - content_weight: weighting on content loss
    - style_layers: list of layers to use for style loss
    - style_weights: list of weights to use for each layer in style_layers
    - tv_weight: weight of total variation regularization term
    - init_random: initialize the starting image to uniform random noise
    """
    # Extract features from the content image
    content_img = preprocess_image(load_image(content_image, size=image_size))
    feats = extract_features(content_img[None], model)
    content_target = feats[content_layer]
    
    # Extract features from the style image
    style_img = preprocess_image(load_image(style_image, size=style_size))
    s_feats = extract_features(style_img[None], model)
    style_targets = []
    # Compute list of TensorFlow Gram matrices
    for idx in style_layers:
        style_targets.append(gram_matrix(s_feats[idx]))
    
    # Set up optimization hyperparameters
    initial_lr = 3.0
    decayed_lr = 0.1
    decay_lr_at = 180
    max_iter = 200
    
    step = tf.Variable(0, trainable=False)
    boundaries = [decay_lr_at]
    values = [initial_lr, decayed_lr]
    learning_rate_fn = tf.keras.optimizers.schedules.PiecewiseConstantDecay(boundaries, values)

    # Later, whenever we perform an optimization step, we pass in the step.
    learning_rate = learning_rate_fn(step)

    optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)
        
    # Initialize the generated image and optimization variables
    
    f, axarr = plt.subplots(1,2)
    axarr[0].axis('off')
    axarr[1].axis('off')
    axarr[0].set_title('Content Source Img.')
    axarr[1].set_title('Style Source Img.')
    axarr[0].imshow(deprocess_image(content_img))
    axarr[1].imshow(deprocess_image(style_img))
    plt.show()
    plt.figure()
    
    # Initialize generated image to content image
    if init_random:
        initializer = tf.random_uniform_initializer(0, 1)
        img = initializer(shape=content_img[None].shape)
        img_var = tf.Variable(img)
        print("Intializing randomly.")
    else:
        img_var = tf.Variable(content_img[None])
        print("Initializing with content image.")
        
    for t in range(max_iter):
        with tf.GradientTape() as tape:
            tape.watch(img_var)
            feats = extract_features(img_var, model)
            # Compute loss
            c_loss = content_loss(content_weight, feats[content_layer], content_target)
            s_loss = style_loss(feats, style_layers, style_targets, style_weights)
            t_loss = tv_loss(img_var, tv_weight)
            loss = c_loss + s_loss + t_loss
        # Compute gradient
        grad = tape.gradient(loss, img_var)
        optimizer.apply_gradients([(grad, img_var)])
        
        img_var.assign(tf.clip_by_value(img_var, -1.5, 1.5))
            
        if t % 10 == 0:
            print('Iteration {}'.format(t))
            #plt.imshow(deprocess_image(img_var[0].numpy(), rescale=True))
            #plt.axis('off')
            #plt.show()
            
    print('Iteration {}'.format(t))    
    plt.imshow(deprocess_image(img_var[0].numpy(), rescale=True))
    plt.axis('off')
    plt.show()



#%%

content_image='styles/tubingen.jpg'
style_image  ='styles/composition_vii.jpg'
image_size= 192
style_size= 512
content_layer=2
content_weight=5e-2 
style_layers=(0, 3, 5, 6)
style_weights=(20000, 500, 12, 1)
tv_weight=5e-2
init_random = False




#%%



# Composition VII + Tubingen
params1 = {
    'content_image' : 'styles/tubingen.jpg',
    'style_image' : 'styles/composition_vii.jpg',
    'image_size' : 192,
    'style_size' : 512,
    'content_layer' : 2,
    'content_weight' : 5e-2, 
    'style_layers' : (0, 3, 5, 6),
    'style_weights' : (20000, 500, 12, 1),
    'tv_weight' : 5e-2
}

style_transfer(**params1)



# Scream + Tubingen
params2 = {
    'content_image':'styles/tubingen.jpg',
    'style_image':'styles/the_scream.jpg',
    'image_size':192,
    'style_size':224,
    'content_layer':2,
    'content_weight':3e-2,
    'style_layers':[0, 3, 5, 6],
    'style_weights':[200000, 800, 12, 1],
    'tv_weight':2e-2
}

style_transfer(**params2)



# Starry Night + Tubingen
params3 = {
    'content_image' : 'styles/tubingen.jpg',
    'style_image' : 'styles/starry_night.jpg',
    'image_size' : 192,
    'style_size' : 192,
    'content_layer' : 2,
    'content_weight' : 6e-2,
    'style_layers' : [0, 3, 5, 6],
    'style_weights' : [300000, 1000, 15, 3],
    'tv_weight' : 2e-2
}

style_transfer(**params3)



# Feature Inversion -- Starry Night + Tubingen
params_inv = {
    'content_image' : 'styles/tubingen.jpg',
    'style_image' : 'styles/starry_night.jpg',
    'image_size' : 192,
    'style_size' : 192,
    'content_layer' : 2,
    'content_weight' : 6e-2,
    'style_layers' : [0, 3, 5, 6],
    'style_weights' : [0, 0, 0, 0], # we discard any contributions from style to the loss
    'tv_weight' : 2e-2,
    'init_random': True # we want to initialize our image to be random
}

style_transfer(**params_inv)