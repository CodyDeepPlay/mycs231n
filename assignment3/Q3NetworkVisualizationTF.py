# -*- coding: utf-8 -*-
"""
Created on Fri May  8 14:11:38 2020

@author: Mingming

cs231n assignment3, Q3, Network visualization
Using tensorflow
"""

# As usual, a bit of setup
import sys
sys.path.append('../')
import time, os, json
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf

from cs231n.classifiers.squeezenet import SqueezeNet
from cs231n.data_utils import load_tiny_imagenet
from cs231n.image_utils import preprocess_image, deprocess_image
from cs231n.image_utils import SQUEEZENET_MEAN, SQUEEZENET_STD

#%matplotlib inline
plt.rcParams['figure.figsize'] = (10.0, 8.0) # set default size of plots
plt.rcParams['image.interpolation'] = 'nearest'
plt.rcParams['image.cmap'] = 'gray'

# for auto-reloading external modules
# see http://stackoverflow.com/questions/1907993/autoreload-of-modules-in-ipython
#%load_ext autoreload
#%autoreload 2

#%%  Load the pretrained model. 
# this model is downloaded from here "http://cs231n.stanford.edu/squeezenet_tf2.zip"

SAVE_PATH = 'cs231n/datasets/squeezenet.ckpt'

if not os.path.exists(SAVE_PATH + ".index"):
    raise ValueError("You need to download SqueezeNet!")

model = SqueezeNet()
status = model.load_weights(SAVE_PATH)

model.trainable = False


#%%  Load some ImageNet images
from cs231n.data_utils import load_imagenet_val
X_raw, y, class_names = load_imagenet_val(num=5)

plt.figure(figsize=(12, 6))
for i in range(5):
    plt.subplot(1, 5, i + 1)
    plt.imshow(X_raw[i])
    plt.title(class_names[y[i]])
    plt.axis('off')
plt.gcf().tight_layout()

#%%  Preprocess images 
'''
The input to the pretrained model is expected to be normalized, so we first preprocess 
the images by subtracting the pixelwise mean and dividing by the pixelwise standard deviation.
'''
X = np.array([preprocess_image(img) for img in X_raw])



#%% 
##########################################################################################
#                                      Saliency Maps                                     #
##########################################################################################

'''
A saliency map tells us the degree to which each pixel in the image affects the 
classification score for that image. To compute it, we compute the gradient of the 
unnormalized score corresponding to the correct class (which is a scalar) with respect 
to the pixels of the image. If the image has shape (H, W, 3) then this gradient will
also have shape (H, W, 3); for each pixel in the image, this gradient tells us the amount 
by which the classification score will change if the pixel changes by a small amount. 
To compute the saliency map, we take the absolute value of this gradient, then take the 
maximum value over the 3 input channels; the final saliency map thus has shape (H, W) and all
entries are nonnegative.
'''

def compute_saliency_maps(X, y, model):
    """
    Compute a class saliency map using the model for images X and labels y.

    Input:
    - X: Input images, numpy array of shape (N, H, W, 3)
    - y: Labels for X, numpy of shape (N,)
    - model: A SqueezeNet model that will be used to compute the saliency map.

    Returns:
    - saliency: A numpy array of shape (N, H, W) giving the saliency maps for the
    input images.
    """
    saliency = None
    # Compute the score of the correct class for each example.
    # This gives a Tensor with shape [N], the number of examples.
    #
    # Note: this is equivalent to scores[np.arange(N), y] we used in NumPy
    # for computing vectorized losses.
    
    ###############################################################################
    # TODO: Produce the saliency maps over a batch of images.                     #
    #                                                                             #
    # 1) Define a gradient tape object and watch input Image variable             #
    # 2) Compute the “loss” for the batch of given input images.                  #
    #    - get scores output by the model for the given batch of input images     #
    #    - use tf.gather_nd or tf.gather to get correct scores                    #
    # 3) Use the gradient() method of the gradient tape object to compute the     #
    #    gradient of the loss with respect to the image                           #
    # 4) Finally, process the returned gradient to compute the saliency map.      #
    ###############################################################################
    # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

    (N, H, W, C) = X.shape
    myX = tf.convert_to_tensor(X)
    # 1) Define a gradient tape object and watch input Image variable 
    with tf.GradientTape() as t:   
        t.watch(myX)    # here watch the input image variable. the input needs to be tf tensor type   
    
    # 2) Compute the “loss” for the batch of given input images.
    #   - get scores output by the model for the given batch of input images
        scores = model.call(myX)  # defined in SqueezeNet() Class, which is in squeezenet.py
    #   - use tf.gather_nd or tf.gather to get correct scores        
        correct_scores = tf.gather_nd(scores, tf.stack((tf.range(N), y), axis=1))
       
    # 3) Use the gradient() method of the gradient tape object to compute the gradient of the loss with respect to the image 
    dmyX = t.gradient(correct_scores, myX)
    
    # 4) Finally, process the returned gradient to compute the saliency map.   
    saliency = np.max(abs(dmyX), axis=3)

    pass

    # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
    ##############################################################################
    #                             END OF YOUR CODE                               #
    ##############################################################################
    return saliency



def show_saliency_maps(X, y, mask):
    mask = np.asarray(mask)
    Xm = X[mask]
    ym = y[mask]

    saliency = compute_saliency_maps(Xm, ym, model)

    for i in range(mask.size):
        plt.subplot(2, mask.size, i + 1)
        plt.imshow(deprocess_image(Xm[i]))
        plt.axis('off')
        plt.title(class_names[ym[i]])
        plt.subplot(2, mask.size, mask.size + i + 1)
        plt.title(mask[i])
        plt.imshow(saliency[i], cmap=plt.cm.hot)
        plt.axis('off')
        plt.gcf().set_size_inches(10, 4)
    plt.show()

mask = np.arange(5)
show_saliency_maps(X, y, mask)


#%%
##########################################################################################
#                                    Fooling images                                      #
##########################################################################################

'''
Given an image and a target class, we can perform gradient ascent over the image to maximize 
the target class, stopping when the network classifies the image as the target class. 
'''

def make_fooling_image(X, target_y, model):
    """
    Generate a fooling image that is close to X, but that the model classifies
    as target_y.

    Inputs:
    - X: Input image, a numpy array of shape (1, 224, 224, 3)
    - target_y: An integer in the range [0, 1000)
    - model: Pretrained SqueezeNet model

    Returns:
    - X_fooling: An image that is close to X, but that is classifed as target_y
    by the model.
    """
    
    # Make a copy of the input that we will modify
    X_fooling = X.copy()
    
    # Step size for the update
    learning_rate = 1
    
    ##############################################################################
    # TODO: Generate a fooling image X_fooling that the model will classify as   #
    # the class target_y. Use gradient *ascent* on the target class score, using #
    # the model.scores Tensor to get the class scores for the model.image.       #
    # When computing an update step, first normalize the gradient:               #
    #   dX = learning_rate * g / ||g||_2                                         #
    #                                                                            #
    # You should write a training loop, where in each iteration, you make an     #
    # update to the input image X_fooling (don't modify X). The loop should      #
    # stop when the predicted class for the input is the same as target_y.       #
    #                                                                            #
    # HINT: Use tf.GradientTape() to keep track of your gradients and            #
    # use tape.gradient to get the actual gradient with respect to X_fooling.    #
    #                                                                            #
    # HINT 2: For most examples, you should be able to generate a fooling image  #
    # in fewer than 100 iterations of gradient ascent. You can print your        #
    # progress over iterations to check your algorithm.                          #
    ##############################################################################
    # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
    
    #(N, H, W, C) = X.shape
    myX_fooling = tf.convert_to_tensor(X_fooling)  # model.call() requires the input to be tf.tensor type.
    
    #model.compile()
    predict_scores = model.predict(myX_fooling)
    predict_y = np.argmax(predict_scores)
    
    track_process = 0
    
    while (predict_y != target_y):
    #while (track_process != 500):    
        
        # 1) Define a gradient tape object and watch input Image variable 
        with tf.GradientTape() as t:   
            t.watch(myX_fooling)    # here watch the input image variable. the input needs to be tf tensor type   
        
        # 2) Compute the “loss” for the batch of given input images.
        #   - get scores output by the model for the given batch of input images
            scores1 = model.call(myX_fooling)  # defined in SqueezeNet() Class, which is in squeezenet.py
        #   - get correct score        
            correct_scores = scores1[:,target_y]          
        # 3) Use the gradient() method of the gradient tape object to compute the gradient of the loss with respect to the image 
        dX_fooling = t.gradient(correct_scores, myX_fooling)
        
        #myX_fooling += learning_rate/np.abs(dX_fooling).mean() * dX_fooling  #np.linalg.norm(dX_fooling)
        myX_fooling += learning_rate/np.linalg.norm(dX_fooling) * dX_fooling  #np.linalg.norm(dX_fooling)
         
        # conduct the prediction again after updating the image
        predict_scores = model.predict(myX_fooling)
        predict_y      = np.argmax(predict_scores)
             
        track_process +=1
        if (track_process%10 == 0):  # print progress every 10 updates
            print('Progress is at which updates:', track_process)
                
    pass
    print('Total iterations are X updates:', track_process)
    
    X_fooling = myX_fooling  

    # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
    ##############################################################################
    #                             END OF YOUR CODE                               #
    ##############################################################################
    return X_fooling




idx = 3
Xi = X[idx][None]
target_y = 6
X_fooling = make_fooling_image(Xi, target_y, model)

# Make sure that X_fooling is classified as y_target
scores = model(X_fooling)
assert tf.math.argmax(scores[0]).numpy() == target_y, 'The network is not fooled!'

# Show original image, fooling image, and difference
orig_img = deprocess_image(Xi[0])
fool_img = deprocess_image(X_fooling[0])
plt.figure(figsize=(12, 6))

# Rescale 
plt.subplot(1, 4, 1)
plt.imshow(orig_img)
plt.axis('off')
plt.title(class_names[y[idx]])
plt.subplot(1, 4, 2)
plt.imshow(fool_img)
plt.title(class_names[target_y])
plt.axis('off')
plt.subplot(1, 4, 3)
plt.title('Difference')
plt.imshow(deprocess_image((Xi-X_fooling)[0]))
plt.axis('off')
plt.subplot(1, 4, 4)
plt.title('Magnified difference (10x)')
plt.imshow(deprocess_image(10 * (Xi-X_fooling)[0]))
plt.axis('off')
plt.gcf().tight_layout()


#%%
##########################################################################################
#                                    Class Visualization                                 #
##########################################################################################

from scipy.ndimage.filters import gaussian_filter1d
def blur_image(X, sigma=1):
    X = gaussian_filter1d(X, sigma, axis=1)
    X = gaussian_filter1d(X, sigma, axis=2)
    return X


def jitter(X, ox, oy):
    """
    Helper function to randomly jitter an image.
    
    Inputs
    - X: Tensor of shape (N, H, W, C)
    - ox, oy: Integers giving number of pixels to jitter along W and H axes
    
    Returns: A new Tensor of shape (N, H, W, C)
    """
    if ox != 0:
        left = X[:, :, :-ox]
        right = X[:, :, -ox:]
        X = tf.concat([right, left], axis=2)
    if oy != 0:
        top = X[:, :-oy]
        bottom = X[:, -oy:]
        X = tf.concat([bottom, top], axis=1)
    return X

#%%

def create_class_visualization(target_y, model, **kwargs):
    """
    Generate an image to maximize the score of target_y under a pretrained model.
    
    Inputs:
    - target_y: Integer in the range [0, 1000) giving the index of the class
    - model: A pretrained CNN that will be used to generate the image
    
    Keyword arguments:
    - l2_reg: Strength of L2 regularization on the image
    - learning_rate: How big of a step to take
    - num_iterations: How many iterations to use
    - blur_every: How often to blur the image as an implicit regularizer
    - max_jitter: How much to jitter the image as an implicit regularizer
    - show_every: How often to show the intermediate result
    """
    l2_reg = kwargs.pop('l2_reg', 1e-3)
    learning_rate = kwargs.pop('learning_rate', 25)
    num_iterations = kwargs.pop('num_iterations', 200)
    blur_every = kwargs.pop('blur_every', 10)
    max_jitter = kwargs.pop('max_jitter', 16)
    show_every = kwargs.pop('show_every', 25)
    
    # We use a single image of random noise as a starting point
    X = 255 * np.random.rand(224, 224, 3)
    X = preprocess_image(X)[None]

    loss = None # scalar loss
    grad = None # gradient of loss with respect to model.image, same size as model.image
    
    X = tf.Variable(X)
    for t in range(num_iterations):
        # Randomly jitter the image a bit; this gives slightly nicer results
        ox, oy = np.random.randint(0, max_jitter, 2)
        X = jitter(X, ox, oy)
        
        ########################################################################
        # TODO: Compute the value of the gradient of the score for             #
        # class target_y with respect to the pixels of the image, and make a   #
        # gradient step on the image using the learning rate. You should use   #
        # the tf.GradientTape() and tape.gradient to compute gradients.        #
        #                                                                      #
        # Be very careful about the signs of elements in your code.            #
        ########################################################################
        # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
        #X = tf.convert_to_tensor(X)
        
        # 1) Define a gradient tape object and watch input Image variable 
        with tf.GradientTape() as tg:   
            tg.watch(X)    # here watch the input image variable. the input needs to be tf tensor type   
            # 2) Compute the “loss” for the batch of given input images.
            #   - get scores output by the model for the given batch of input images
            scores1 = model.call(X)  # defined in SqueezeNet() Class, which is in squeezenet.py
            #   - get correct score 
            correct_scores = scores1[:,target_y]  # get the correct score, here there is only one score, because target_y is only one class
        
            #SyI = np.argmax([correct_scores , -l2_reg*np.sum(X*X)])
             
        # 3) Use the gradient() method of the gradient tape object to compute the gradient of the loss with respect to the image 
        dX = tg.gradient(correct_scores, X)
        dX += l2_reg*2*X   # add L2 regularization to the image gradient
        
        X += learning_rate*dX
        
        pass

        # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
        ############################################################################
        #                             END OF YOUR CODE                             #
        ############################################################################
        
        # Undo the jitter
        X = jitter(X, -ox, -oy)
        # As a regularizer, clip and periodically blur
        
        if (t%20 == 0):  # print progress every 10 updates
            template = 'Training progress is at {}th iteration out of {} iterations.'
            print(template.format(t, num_iterations))
        
        
        X = tf.clip_by_value(X, -SQUEEZENET_MEAN/SQUEEZENET_STD, (1.0 - SQUEEZENET_MEAN)/SQUEEZENET_STD)
        if t % blur_every == 0:
            X = blur_image(X, sigma=0.5)

        # Periodically show the image
        if t == 0 or (t + 1) % show_every == 0 or t == num_iterations - 1:
            plt.imshow(deprocess_image(X[0]))
            class_name = class_names[target_y]
            plt.title('%s\nIteration %d / %d' % (class_name, t + 1, num_iterations))
            plt.gcf().set_size_inches(4, 4)
            plt.axis('off')
            plt.show()
    return X

'''

Once you have completed the implementation in the cell above, run the following cell to 
generate an image of Tarantula:
'''
plt.figure()
target_y = 76 # Tarantula
out = create_class_visualization(target_y, model)

target_y = np.random.randint(1000)
# target_y = 78 # Tick
# target_y = 187 # Yorkshire Terrier
# target_y = 683 # Oboe
# target_y = 366 # Gorilla
# target_y = 604 # Hourglass
print(class_names[target_y])
plt.figure()
X = create_class_visualization(target_y, model)














