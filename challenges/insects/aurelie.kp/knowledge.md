---
title: Insects classification
authors:
- "Aur\xE9lie"
tags:
- RAMP
- ConvNets
- Challenge
created_at: 2016-11-08 00:00:00
updated_at: 2016-12-11 13:47:31.194458
tldr: Insect classification challenge
---
# Pollenating Insects classification

## Context - aim
The aim is to build an image classification model allowing to predict species of insects from pictures. 

The dataset consists in 20348 pictures of insects from different species gathered from the SPIPOLL project and labeled. There are 18 different classes each one corresponding to a different insect specie. Each picture is a 64x64 colored image, which makes a total of 64×64×3=12288 features per picture. 

*Reference* : http://www.spipoll.org/

## Image recognition and neural networks

Over the past few years, neural nets have proven to be very efficient as regards image classification. As an example, "basic" multilayer perceptrons can yield very good results in terms of classification errors on the well-known handwritten digits [MNIST dataset](http://yann.lecun.com/exdb/mnist/) (ref : [D. C. Ciresan et al. (2010)](https://arxiv.org/pdf/1003.0358.pdf))

*To get familiar with neural networks (with a nice tutorial on the handwritten digits classification problem) : * http://neuralnetworksanddeeplearning.com/chap1.html


Object (or insect...) recognition using "real life" images can however prove to be tricky, and this for many reasons. A few are listed below :
- The separation between the object and its background is not necessarily obvious
- Several pictures of a same object can actually look quite different the one from the other. For example, the object's location in the image or the illumination can vary, which means the classification model needs to be invariant under certain transformations (translational symmetry for example)
- Efficient computer vision requires models that are able to exploit the 2D topology of pixels, as well as locality.


Because of their particular properties, convolutional neural networks (CNNs) allow to address the issues listed above.


### Convolutional neural networks

By construction, CNNs are well suited for image classification :
- from one convolutional layer (CL) to the next one, only a few units are connected together, which allows local treatment of subsets of pixels
- parameter sharing in one given CL contributes to translational invariance of the model
- In practice, the two constraints listed above reduce drastically the number of model parameters to be computed, and then allow to train quite complex models in a reasonable time.

*Some useful reference to gain knowledge of CNNs : * 
http://cs231n.github.io/convolutional-networks/


A basic CNN consists in successions of convolutional layers (CL) and pooling layers (PL), the latter allowing to reduce the number of parameters to be computed in the network. Those successions of CLs and PLs allow to perform feature extraction. For image classification, the output layer is a fully connected NN layer with a number of units equal to the number of classes. The output layer activation is a softmax, so that the i$^{th}$ output unit activation is consistent with the probability that the image belongs to class i.

It's also common to see in a CNN, the CLs and PLs being combined with some rectification (non-linearities) and normalization layers that can drastically improve the classification accuracy ([Jarrett et al. (2009)](http://cs.nyu.edu/~koray/publis/jarrett-iccv-09.pdf))



## Building CNNs with Keras

Below are loaded some useful libraries for building, training and evaluating neural nets.
- [Keras](https://keras.io/) is a python library running either on [Tensorflow](https://www.tensorflow.org/) or [Theano](http://deeplearning.net/software/theano/). The following pieces of codes are valid for a Tensorflow implementation. [Here are some instructions to install Tensorflow](https://github.com/tensorflow/tensorflow#download-and-setup). As training neural nets can be quite computationally costly, it is recommended to install the gpu version of tensorflow (obviously, it's possible only if you have a dedicated GPU!).


- In what's next we'll use some methods that are implemented in the well-known machine learning library [scikit-learn](http://scikit-learn.org/stable/). In particular, the methods cross_val_score() and GridSearchCV() will be used, respectively to apply some unit tests to the models and to perform grid searches on model hyperparameters. For those functions to be called on Keras models, those latter will be wrapped into classes that are "compatible" with scikit-learn. [Here is some useful tutorial to build scitkit-learn wrappers for estimators](http://danielhnyk.cz/creating-your-own-estimator-scikit-learn/).


```python
from __future__ import print_function
import time

import numpy as np

from sklearn.base import BaseEstimator

from sklearn.preprocessing import LabelEncoder

from sklearn.cross_validation import StratifiedShuffleSplit
from sklearn.cross_validation import cross_val_score
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import classification_report
from sklearn.externals import joblib

from keras.models import Sequential
from keras.layers import Dense, Convolution2D, MaxPooling2D, AveragePooling2D, Flatten, Activation
from keras.callbacks import EarlyStopping
from keras.utils import np_utils

```
## Loading dataset
Remark : 10% of the examples are staged as a test set that will be used to evaluate classification accuracy with the model chosen from hyperparameter tuning.


```python
f = np.load('train_64x64.npz')

X = f['X']
print('X shape : ', X.shape)
X_flat = X.reshape((20348,64*64*3))
print('X_flat shape : ', X_flat.shape)

y = f['y']
print(y.shape)
# encode class values
encoder = LabelEncoder()
encoder.fit(y)
encoded_y = encoder.transform(y)
print('y shape : ', encoded_y.shape)

X_train, X_test, y_train, y_test = train_test_split(X_flat, encoded_y, test_size=0.1, random_state=0, stratify=encoded_y)
print('Train set size : ', X_train.shape[0])
print('Test set size : ', X_test.shape[0])
```
    X shape :  (20348, 64, 64, 3)
    X_flat shape :  (20348, 12288)
    (20348,)
    y shape :  (20348,)
    Train set size :  18313
    Test set size :  2035


## Pipeline
The pipeline to perform model selection is inspired from [Jarrett et al. (2009)](http://cs.nyu.edu/~koray/publis/jarrett-iccv-09.pdf).

### Model architecture

In this paper, the impact of the following model properties on object classification accuracy is investigated :
- number of convolutional layers (CL) needed to perform feature extraction
- type of pooling (PL) used (average pooling vs. max pooling)
- role of rectification layers (RL)


In the following, those criteria will be tested so as to find the model "architecture" that is best suited for our classification problem. This will be done by training different models (with different numbers of CLs, and varied types of PL / RL) and evaluating classification accuracy by cross-validation.


### Hyperparameter tuning

Once the model architecture is determined, some hyperparameters tuning is performed by using grid search. The concerned hyperparameters are :
- number of feature maps in CLs
- dimensions of feature maps in CLs
- dimensions of pooling matrices in PLs.


NB : In an ideal world the model "architecture" could also be tuned with grid search, together with the hyperparameters listed above. To avoid exploding the parameters space, grid search was however performed by focusing only on the number and dimensions of feature maps in convolutional layers, and the dimensions of the pooling matrices.

### Define a unit_test function to extract cross-validated score for model architecture selection


```python
def unit_test(classifier, nb_iter=3):
    test_size = 0.2
    random_state = 15
    cv = StratifiedShuffleSplit(encoded_y, nb_iter,test_size=test_size,random_state=random_state)
    clf = classifier()
    scores = cross_val_score(clf, X=X_flat, y=encoded_y, scoring='accuracy', cv=cv)
    return scores
```
### Define a hyperparameter_optim function to perform grid search 


```python
def hyperparameter_optim(classifier, params, cv=3):

    clf = GridSearchCV(classifier(), params, cv=cv, scoring='accuracy')
    clf.fit(X_train, y_train)

    print("Best parameters set found:")
    print(clf.best_params_)
    print()
    print("Grid scores:")
    means = clf.cv_results_['mean_test_score']
    stds = clf.cv_results_['std_test_score']
    for mean, std, params in zip(means, stds, clf.cv_results_['params']):
        print("%0.3f (+/-%0.03f) for %r"
                % (mean, std * 2, params))
    print()
    
    return clf
```
## Determining best model architecture

### "Basic" model with only one convolutional layer

#### Building corresponding classifier inheriting from sklearn.BaseEstimator
##### Default hyperparameters
- nb_filters = 32, filter_size = (3,3) in CL
- pool_size = (2,2) in PL
- nb_epochs = 10

##### Early stopping
- An early stopping condition based on the monitoring of the validation set accuracy is used so as to avoid overfitting and improve a bit the training time.





```python
class Classifier(BaseEstimator):  

    def __init__(self, nb_filters=32, filter_size=3, pool_size=2):
        self.nb_filters = nb_filters
        self.filter_size = filter_size
        self.pool_size = pool_size
        
    def preprocess(self, X):
        X = X.reshape((X.shape[0],64,64,3))
        X = (X / 255.)
        X = X.astype(np.float32)
        return X
    
    def preprocess_y(self, y):
        return np_utils.to_categorical(y)
    
    def fit(self, X, y):
        X = self.preprocess(X)
        y = self.preprocess_y(y)
        
        hyper_parameters = dict(
        nb_filters = self.nb_filters,
        filter_size = self.filter_size,
        pool_size = self.pool_size 
        )
        
        print("FIT PARAMS : ")
        print(hyper_parameters)
        
        self.model = build_model(hyper_parameters)
        
        earlyStopping = EarlyStopping(monitor='val_loss', patience=1, verbose=1, mode='auto')
        self.model.fit(X, y, nb_epoch=10, verbose=1, callbacks=[earlyStopping], validation_split=0.1, 
                       validation_data=None, shuffle=True)
        return self

    def predict(self, X):
        X = self.preprocess(X)
        return self.model.predict_classes(X)

    def predict_proba(self, X):
        X = self.preprocess(X)
        return self.model.predict(X)
    
    def score(self, X, y):
        print(self.model.evaluate(self, X, y, batch_size=32, verbose=1, sample_weight=None))
        return self.model.evaluate(self, X, y, batch_size=32, verbose=1, sample_weight=None)
```
#### One CL combined to one PL (average pooling / no rectification)


```python
def build_model(hp):
    net = Sequential()
    net.add(Convolution2D(hp['nb_filters'], hp['filter_size'], hp['filter_size'], border_mode='same', 
                          input_shape=(64,64,3)))
    net.add(AveragePooling2D(pool_size=(hp['pool_size'],hp['pool_size'])))
    net.add(Flatten())
    net.add(Dense(output_dim=18))
    net.add(Activation("softmax"))
    
    net.compile(loss='categorical_crossentropy', optimizer='adadelta', metrics=['accuracy'])
    
    return net

print(unit_test(Classifier,nb_iter=3))
```
    FIT PARAMS : 
    {'filter_size': 3, 'nb_filters': 32, 'pool_size': 2}
    Train on 14650 samples, validate on 1628 samples
    Epoch 1/10
    14650/14650 [==============================] - 31s - loss: 2.3809 - acc: 0.3158 - val_loss: 2.2046 - val_acc: 0.3649
    Epoch 2/10
    14650/14650 [==============================] - 31s - loss: 2.1510 - acc: 0.3605 - val_loss: 2.2075 - val_acc: 0.3526
    Epoch 3/10
    14650/14650 [==============================] - 30s - loss: 2.0355 - acc: 0.3792 - val_loss: 2.1813 - val_acc: 0.3636
    Epoch 4/10
    14650/14650 [==============================] - 30s - loss: 1.9465 - acc: 0.4004 - val_loss: 2.2266 - val_acc: 0.3495
    Epoch 5/10
    14624/14650 [============================>.] - ETA: 0s - loss: 1.8660 - acc: 0.4229Epoch 00004: early stopping
    14650/14650 [==============================] - 30s - loss: 1.8663 - acc: 0.4226 - val_loss: 2.2773 - val_acc: 0.3372
    4064/4070 [============================>.] - ETA: 0sFIT PARAMS : 
    {'filter_size': 3, 'nb_filters': 32, 'pool_size': 2}
    Train on 14650 samples, validate on 1628 samples
    Epoch 1/10
    14650/14650 [==============================] - 32s - loss: 2.3719 - acc: 0.3146 - val_loss: 2.3389 - val_acc: 0.3071
    Epoch 2/10
    14650/14650 [==============================] - 32s - loss: 2.1393 - acc: 0.3584 - val_loss: 2.2357 - val_acc: 0.3409
    Epoch 3/10
    14650/14650 [==============================] - 31s - loss: 2.0210 - acc: 0.3870 - val_loss: 2.2526 - val_acc: 0.3428
    Epoch 4/10
    14624/14650 [============================>.] - ETA: 0s - loss: 1.9310 - acc: 0.4045Epoch 00003: early stopping
    14650/14650 [==============================] - 31s - loss: 1.9320 - acc: 0.4042 - val_loss: 2.3660 - val_acc: 0.2985
    4064/4070 [============================>.] - ETA: 0sFIT PARAMS : 
    {'filter_size': 3, 'nb_filters': 32, 'pool_size': 2}
    Train on 14650 samples, validate on 1628 samples
    Epoch 1/10
    14650/14650 [==============================] - 32s - loss: 2.3874 - acc: 0.3148 - val_loss: 2.2172 - val_acc: 0.3722
    Epoch 2/10
    14650/14650 [==============================] - 31s - loss: 2.1571 - acc: 0.3599 - val_loss: 2.1146 - val_acc: 0.3753
    Epoch 3/10
    14650/14650 [==============================] - 30s - loss: 2.0444 - acc: 0.3813 - val_loss: 2.1243 - val_acc: 0.3753
    Epoch 4/10
    14624/14650 [============================>.] - ETA: 0s - loss: 1.9541 - acc: 0.4001Epoch 00003: early stopping
    14650/14650 [==============================] - 30s - loss: 1.9540 - acc: 0.4000 - val_loss: 2.1610 - val_acc: 0.3630
    4064/4070 [============================>.] - ETA: 0s[ 0.33120393  0.32014742  0.34668305]


#### One CL combined to one PL (max pooling / no rectification)


```python
def build_model(hp):
    net = Sequential()
    net.add(Convolution2D(hp['nb_filters'], hp['filter_size'], hp['filter_size'], border_mode='same', 
                          input_shape=(64,64,3)))
    net.add(MaxPooling2D(pool_size=(hp['pool_size'],hp['pool_size'])))
    net.add(Flatten())
    net.add(Dense(output_dim=18))
    net.add(Activation("softmax"))
    
    net.compile(loss='categorical_crossentropy', optimizer='adadelta', metrics=['accuracy'])
    
    return net

print(unit_test(Classifier,nb_iter=3))
```
    FIT PARAMS : 
    {'filter_size': 3, 'nb_filters': 32, 'pool_size': 2}
    Train on 14650 samples, validate on 1628 samples
    Epoch 1/10
    14650/14650 [==============================] - 48s - loss: 2.3369 - acc: 0.3217 - val_loss: 2.1589 - val_acc: 0.3741
    Epoch 2/10
    14650/14650 [==============================] - 48s - loss: 2.0477 - acc: 0.3777 - val_loss: 2.1185 - val_acc: 0.3771
    Epoch 3/10
    14650/14650 [==============================] - 49s - loss: 1.8741 - acc: 0.4244 - val_loss: 2.0649 - val_acc: 0.3845
    Epoch 4/10
    14650/14650 [==============================] - 48s - loss: 1.7407 - acc: 0.4704 - val_loss: 2.0838 - val_acc: 0.3747
    Epoch 5/10
    14624/14650 [============================>.] - ETA: 0s - loss: 1.6169 - acc: 0.5086Epoch 00004: early stopping
    14650/14650 [==============================] - 48s - loss: 1.6169 - acc: 0.5087 - val_loss: 2.1108 - val_acc: 0.3814
    4070/4070 [==============================] - 5s     
    FIT PARAMS : 
    {'filter_size': 3, 'nb_filters': 32, 'pool_size': 2}
    Train on 14650 samples, validate on 1628 samples
    Epoch 1/10
    14650/14650 [==============================] - 49s - loss: 2.3222 - acc: 0.3258 - val_loss: 2.2157 - val_acc: 0.3434
    Epoch 2/10
    14650/14650 [==============================] - 48s - loss: 2.0546 - acc: 0.3782 - val_loss: 2.1354 - val_acc: 0.3544
    Epoch 3/10
    14650/14650 [==============================] - 48s - loss: 1.8970 - acc: 0.4192 - val_loss: 2.1762 - val_acc: 0.3520
    Epoch 4/10
    14650/14650 [==============================] - 48s - loss: 1.7670 - acc: 0.4611 - val_loss: 2.1107 - val_acc: 0.3753
    Epoch 5/10
    14650/14650 [==============================] - 48s - loss: 1.6501 - acc: 0.4999 - val_loss: 2.1381 - val_acc: 0.3704
    Epoch 6/10
    14624/14650 [============================>.] - ETA: 0s - loss: 1.5398 - acc: 0.5356Epoch 00005: early stopping
    14650/14650 [==============================] - 49s - loss: 1.5399 - acc: 0.5354 - val_loss: 2.1903 - val_acc: 0.3587
    4070/4070 [==============================] - 5s     
    FIT PARAMS : 
    {'filter_size': 3, 'nb_filters': 32, 'pool_size': 2}
    Train on 14650 samples, validate on 1628 samples
    Epoch 1/10
    14650/14650 [==============================] - 48s - loss: 2.3454 - acc: 0.3203 - val_loss: 2.2178 - val_acc: 0.3729
    Epoch 2/10
    14650/14650 [==============================] - 48s - loss: 2.0830 - acc: 0.3751 - val_loss: 2.0535 - val_acc: 0.3900
    Epoch 3/10
    14650/14650 [==============================] - 48s - loss: 1.9202 - acc: 0.4206 - val_loss: 2.0252 - val_acc: 0.3931
    Epoch 4/10
    14650/14650 [==============================] - 49s - loss: 1.7835 - acc: 0.4621 - val_loss: 2.0245 - val_acc: 0.4036
    Epoch 5/10
    14650/14650 [==============================] - 48s - loss: 1.6629 - acc: 0.4982 - val_loss: 2.0862 - val_acc: 0.3722
    Epoch 6/10
    14624/14650 [============================>.] - ETA: 0s - loss: 1.5483 - acc: 0.5345Epoch 00005: early stopping
    14650/14650 [==============================] - 49s - loss: 1.5484 - acc: 0.5345 - val_loss: 2.1178 - val_acc: 0.3857
    4070/4070 [==============================] - 5s     
    [ 0.37149877  0.36732187  0.36388206]


From the above we know that max pooling performs better than average pooling, as suggested in [Jarrett et al. (2009)](http://cs.nyu.edu/~koray/publis/jarrett-iccv-09.pdf).

This very basic first model is composed of one CL followed by one PL(max) for feature extraction, and a single fully-connected layer with softmax activation for the classification step. The cross-validated accuracy obtained with this model is $\sim$ 36%.

In the following, max pooling will systematically be used.

#### One CL combined to one PL (with sigmoid non-linearity)


```python
def build_model(hp):
    net = Sequential()
    net.add(Convolution2D(hp['nb_filters'], hp['filter_size'], hp['filter_size'], border_mode='same', 
                          input_shape=(64,64,3)))
    net.add(Activation("sigmoid"))
    net.add(MaxPooling2D(pool_size=(hp['pool_size'],hp['pool_size'])))
    net.add(Flatten())
    net.add(Dense(output_dim=18))
    net.add(Activation("softmax"))
    
    net.compile(loss='categorical_crossentropy', optimizer='adadelta', metrics=['accuracy'])
    
    return net

print(unit_test(Classifier,nb_iter=3))
```
    FIT PARAMS : 
    {'filter_size': 3, 'nb_filters': 32, 'pool_size': 2}
    Train on 14650 samples, validate on 1628 samples
    Epoch 1/10
    14650/14650 [==============================] - 36s - loss: 11.2415 - acc: 0.2704 - val_loss: 11.1777 - val_acc: 0.3065
    Epoch 2/10
    14650/14650 [==============================] - 36s - loss: 11.4906 - acc: 0.2871 - val_loss: 11.1777 - val_acc: 0.3065
    Epoch 3/10
    14624/14650 [============================>.] - ETA: 0s - loss: 11.4890 - acc: 0.2872Epoch 00002: early stopping
    14650/14650 [==============================] - 35s - loss: 11.4906 - acc: 0.2871 - val_loss: 11.1777 - val_acc: 0.3065
    4070/4070 [==============================] - 4s     
    FIT PARAMS : 
    {'filter_size': 3, 'nb_filters': 32, 'pool_size': 2}
    Train on 14650 samples, validate on 1628 samples
    Epoch 1/10
    14650/14650 [==============================] - 36s - loss: 11.3950 - acc: 0.2883 - val_loss: 11.7322 - val_acc: 0.2721
    Epoch 2/10
    14650/14650 [==============================] - 35s - loss: 11.4290 - acc: 0.2909 - val_loss: 11.7322 - val_acc: 0.2721
    Epoch 3/10
    14624/14650 [============================>.] - ETA: 0s - loss: 11.4273 - acc: 0.2910Epoch 00002: early stopping
    14650/14650 [==============================] - 36s - loss: 11.4290 - acc: 0.2909 - val_loss: 11.7322 - val_acc: 0.2721
    4064/4070 [============================>.] - ETA: 0sFIT PARAMS : 
    {'filter_size': 3, 'nb_filters': 32, 'pool_size': 2}
    Train on 14650 samples, validate on 1628 samples
    Epoch 1/10
    14650/14650 [==============================] - 36s - loss: 9.5754 - acc: 0.2368 - val_loss: 8.0837 - val_acc: 0.3237
    Epoch 2/10
    14650/14650 [==============================] - 35s - loss: 6.9748 - acc: 0.2493 - val_loss: 3.2529 - val_acc: 0.3360
    Epoch 3/10
    14650/14650 [==============================] - 35s - loss: 2.7146 - acc: 0.2749 - val_loss: 2.3388 - val_acc: 0.3280
    Epoch 4/10
    14650/14650 [==============================] - 36s - loss: 2.4289 - acc: 0.3030 - val_loss: 2.3514 - val_acc: 0.3268
    Epoch 5/10
    14650/14650 [==============================] - 35s - loss: 2.3317 - acc: 0.3300 - val_loss: 2.2995 - val_acc: 0.3673
    Epoch 6/10
    14650/14650 [==============================] - 35s - loss: 2.2715 - acc: 0.3403 - val_loss: 2.3420 - val_acc: 0.3004
    Epoch 7/10
    14624/14650 [============================>.] - ETA: 0s - loss: 2.2215 - acc: 0.3526Epoch 00006: early stopping
    14650/14650 [==============================] - 35s - loss: 2.2216 - acc: 0.3525 - val_loss: 2.3333 - val_acc: 0.3649
    4064/4070 [============================>.] - ETA: 0s[ 0.28894349  0.28894349  0.34103194]


#### One CL combined to one PL (with relu non-linearity)


```python
def build_model(hp):
    net = Sequential()
    net.add(Convolution2D(hp['nb_filters'], hp['filter_size'], hp['filter_size'], border_mode='same', 
                          input_shape=(64,64,3)))
    net.add(Activation("relu"))
    net.add(MaxPooling2D(pool_size=(hp['pool_size'],hp['pool_size'])))
    net.add(Flatten())
    net.add(Dense(output_dim=18))
    net.add(Activation("softmax"))
    
    net.compile(loss='categorical_crossentropy', optimizer='adadelta', metrics=['accuracy'])
    
    return net

print(unit_test(Classifier,nb_iter=3))
```
    FIT PARAMS : 
    {'filter_size': 3, 'nb_filters': 32, 'pool_size': 2}
    Train on 14650 samples, validate on 1628 samples
    Epoch 1/10
    14650/14650 [==============================] - 34s - loss: 2.2736 - acc: 0.3410 - val_loss: 2.1303 - val_acc: 0.3643
    Epoch 2/10
    14650/14650 [==============================] - 34s - loss: 1.9507 - acc: 0.4123 - val_loss: 1.9629 - val_acc: 0.4226
    Epoch 3/10
    14650/14650 [==============================] - 34s - loss: 1.7659 - acc: 0.4626 - val_loss: 1.8785 - val_acc: 0.4527
    Epoch 4/10
    14650/14650 [==============================] - 34s - loss: 1.6240 - acc: 0.5059 - val_loss: 1.9083 - val_acc: 0.4324
    Epoch 5/10
    14650/14650 [==============================] - 35s - loss: 1.5049 - acc: 0.5431 - val_loss: 1.8683 - val_acc: 0.4453
    Epoch 6/10
    14650/14650 [==============================] - 35s - loss: 1.3842 - acc: 0.5810 - val_loss: 1.8628 - val_acc: 0.4613
    Epoch 7/10
    14650/14650 [==============================] - 36s - loss: 1.2716 - acc: 0.6192 - val_loss: 1.8882 - val_acc: 0.4521
    Epoch 8/10
    14624/14650 [============================>.] - ETA: 0s - loss: 1.1662 - acc: 0.6528Epoch 00007: early stopping
    14650/14650 [==============================] - 36s - loss: 1.1663 - acc: 0.6527 - val_loss: 1.9630 - val_acc: 0.4281
    4064/4070 [============================>.] - ETA: 0sFIT PARAMS : 
    {'filter_size': 3, 'nb_filters': 32, 'pool_size': 2}
    Train on 14650 samples, validate on 1628 samples
    Epoch 1/10
    14650/14650 [==============================] - 37s - loss: 2.2900 - acc: 0.3421 - val_loss: 2.1442 - val_acc: 0.3612
    Epoch 2/10
    14650/14650 [==============================] - 36s - loss: 1.9574 - acc: 0.4091 - val_loss: 1.9871 - val_acc: 0.4042
    Epoch 3/10
    14650/14650 [==============================] - 35s - loss: 1.7785 - acc: 0.4545 - val_loss: 1.9729 - val_acc: 0.4054
    Epoch 4/10
    14650/14650 [==============================] - 36s - loss: 1.6450 - acc: 0.4984 - val_loss: 1.8864 - val_acc: 0.4324
    Epoch 5/10
    14650/14650 [==============================] - 36s - loss: 1.5276 - acc: 0.5360 - val_loss: 1.8952 - val_acc: 0.4490
    Epoch 6/10
    14650/14650 [==============================] - 36s - loss: 1.4183 - acc: 0.5717 - val_loss: 1.8703 - val_acc: 0.4435
    Epoch 7/10
    14650/14650 [==============================] - 37s - loss: 1.3157 - acc: 0.6026 - val_loss: 1.9247 - val_acc: 0.4251
    Epoch 8/10
    14624/14650 [============================>.] - ETA: 0s - loss: 1.2166 - acc: 0.6351Epoch 00007: early stopping
    14650/14650 [==============================] - 36s - loss: 1.2162 - acc: 0.6353 - val_loss: 1.9340 - val_acc: 0.4337
    4064/4070 [============================>.] - ETA: 0sFIT PARAMS : 
    {'filter_size': 3, 'nb_filters': 32, 'pool_size': 2}
    Train on 14650 samples, validate on 1628 samples
    Epoch 1/10
    14650/14650 [==============================] - 37s - loss: 2.2984 - acc: 0.3393 - val_loss: 2.0347 - val_acc: 0.4048
    Epoch 2/10
    14650/14650 [==============================] - 36s - loss: 1.9431 - acc: 0.4197 - val_loss: 1.9551 - val_acc: 0.4306
    Epoch 3/10
    14650/14650 [==============================] - 36s - loss: 1.7485 - acc: 0.4719 - val_loss: 1.8781 - val_acc: 0.4361
    Epoch 4/10
    14650/14650 [==============================] - 35s - loss: 1.6004 - acc: 0.5141 - val_loss: 1.8985 - val_acc: 0.4257
    Epoch 5/10
    14624/14650 [============================>.] - ETA: 0s - loss: 1.4636 - acc: 0.5570Epoch 00004: early stopping
    14650/14650 [==============================] - 36s - loss: 1.4645 - acc: 0.5569 - val_loss: 1.8977 - val_acc: 0.4392
    4064/4070 [============================>.] - ETA: 0s[ 0.43562654  0.45651106  0.42776413]


From the above we know that adding sigmoid non-linearities deteriorate the performances, whereas relu rectification improves the classification accuracy.

In what follows, relu activations will be used as rectification layers.

### Model with two convolutional layers

#### Building corresponding classifier inheriting from sklearn.BaseEstimator
##### Default hyperparameters
- nb_filters_1 = 32, filter_size_1 = (3,3) in 1st CL
- pool_size_1 = (2,2) in 1st PL
- nb_filters_2 = 32, filter_size_2 = (3,3) in 2nd CL
- pool_size_2 = (2,2) in 2nd PL
- nb_epochs = 10

##### Early stopping
- An early stopping condition based on the monitoring of the validation set accuracy is used so as to avoid overfitting and improve a bit the training time.





```python
class Classifier(BaseEstimator):  

    def __init__(self, nb_filters_1=32, filter_size_1=3, pool_size_1=2,
                 nb_filters_2=32, filter_size_2=3, pool_size_2=2):
        self.nb_filters_1 = nb_filters_1
        self.filter_size_1 = filter_size_1
        self.pool_size_1 = pool_size_1
        self.nb_filters_2 = nb_filters_2
        self.filter_size_2 = filter_size_2
        self.pool_size_2 = pool_size_2
        
    def preprocess(self, X):
        X = X.reshape((X.shape[0],64,64,3))
        X = (X / 255.)
        X = X.astype(np.float32)
        return X
    
    def preprocess_y(self, y):
        return np_utils.to_categorical(y)
    
    def fit(self, X, y):
        X = self.preprocess(X)
        y = self.preprocess_y(y)
        
        hyper_parameters = dict(
        nb_filters_1 = self.nb_filters_1,
        filter_size_1 = self.filter_size_1,
        pool_size_1 = self.pool_size_1,
        nb_filters_2 = self.nb_filters_2,
        filter_size_2 = self.filter_size_2,
        pool_size_2 = self.pool_size_2
        )
        
        print("FIT PARAMS : ")
        print(hyper_parameters)
        
        self.model = build_model(hyper_parameters)
        
        earlyStopping = EarlyStopping(monitor='val_loss', patience=1, verbose=1, mode='auto')
        self.model.fit(X, y, nb_epoch=10, verbose=1, callbacks=[earlyStopping], validation_split=0.1, 
                       validation_data=None, shuffle=True)
        return self

    def predict(self, X):
        print("PREDICT")
        X = self.preprocess(X)
        return self.model.predict_classes(X)

    def predict_proba(self, X):
        X = self.preprocess(X)
        return self.model.predict(X)
    
    def score(self, X, y):
        print("SCORE")
        print(self.model.evaluate(self, X, y, batch_size=32, verbose=1, sample_weight=None))
        return self.model.evaluate(self, X, y, batch_size=32, verbose=1, sample_weight=None) 
    
```
#### Two CLs/PLs (no rectification layer)


```python
def build_model(hp):
    net = Sequential()
    net.add(Convolution2D(hp['nb_filters_1'], hp['filter_size_1'], hp['filter_size_1'], border_mode='same', 
                          input_shape=(64,64,3)))
    net.add(MaxPooling2D(pool_size=(hp['pool_size_1'],hp['pool_size_1'])))
    net.add(Convolution2D(hp['nb_filters_2'], hp['filter_size_2'], hp['filter_size_2'], border_mode='same'))
    net.add(MaxPooling2D(pool_size=(hp['pool_size_2'],hp['pool_size_2'])))
    net.add(Flatten())
    net.add(Dense(output_dim=18))
    net.add(Activation("softmax"))
    
    net.compile(loss='categorical_crossentropy', optimizer='adadelta', metrics=['accuracy'])
    
    return net

print(unit_test(Classifier,nb_iter=3))
```
    FIT PARAMS : 
    {'nb_filters_1': 32, 'nb_filters_2': 32, 'filter_size_2': 3, 'pool_size_1': 2, 'pool_size_2': 2, 'filter_size_1': 3}
    Train on 14650 samples, validate on 1628 samples
    Epoch 1/10
    14650/14650 [==============================] - 52s - loss: 2.3540 - acc: 0.3214 - val_loss: 2.1798 - val_acc: 0.3649
    Epoch 2/10
    14650/14650 [==============================] - 53s - loss: 2.0866 - acc: 0.3746 - val_loss: 2.0392 - val_acc: 0.4097
    Epoch 3/10
    14650/14650 [==============================] - 52s - loss: 1.9604 - acc: 0.4173 - val_loss: 1.9712 - val_acc: 0.4220
    Epoch 4/10
    14650/14650 [==============================] - 50s - loss: 1.8878 - acc: 0.4433 - val_loss: 1.9656 - val_acc: 0.4232
    Epoch 5/10
    14650/14650 [==============================] - 50s - loss: 1.8248 - acc: 0.4629 - val_loss: 1.9560 - val_acc: 0.4337
    Epoch 6/10
    14650/14650 [==============================] - 51s - loss: 1.7627 - acc: 0.4747 - val_loss: 1.9769 - val_acc: 0.4189
    Epoch 7/10
    14624/14650 [============================>.] - ETA: 0s - loss: 1.6997 - acc: 0.4941Epoch 00006: early stopping
    14650/14650 [==============================] - 50s - loss: 1.6998 - acc: 0.4940 - val_loss: 1.9715 - val_acc: 0.4318
    PREDICT
    4064/4070 [============================>.] - ETA: 0sFIT PARAMS : 
    {'nb_filters_1': 32, 'nb_filters_2': 32, 'filter_size_2': 3, 'pool_size_1': 2, 'pool_size_2': 2, 'filter_size_1': 3}
    Train on 14650 samples, validate on 1628 samples
    Epoch 1/10
    14650/14650 [==============================] - 49s - loss: 2.3705 - acc: 0.3212 - val_loss: 2.2507 - val_acc: 0.3323
    Epoch 2/10
    14650/14650 [==============================] - 49s - loss: 2.1223 - acc: 0.3710 - val_loss: 2.1468 - val_acc: 0.3722
    Epoch 3/10
    14650/14650 [==============================] - 49s - loss: 1.9834 - acc: 0.4035 - val_loss: 2.0264 - val_acc: 0.4122
    Epoch 4/10
    14650/14650 [==============================] - 50s - loss: 1.8932 - acc: 0.4386 - val_loss: 2.0172 - val_acc: 0.4085
    Epoch 5/10
    14650/14650 [==============================] - 50s - loss: 1.8237 - acc: 0.4552 - val_loss: 1.9837 - val_acc: 0.4343
    Epoch 6/10
    14650/14650 [==============================] - 50s - loss: 1.7567 - acc: 0.4784 - val_loss: 1.9523 - val_acc: 0.4269
    Epoch 7/10
    14650/14650 [==============================] - 49s - loss: 1.6919 - acc: 0.4930 - val_loss: 1.9674 - val_acc: 0.4324
    Epoch 8/10
    14624/14650 [============================>.] - ETA: 0s - loss: 1.6344 - acc: 0.5113Epoch 00007: early stopping
    14650/14650 [==============================] - 49s - loss: 1.6344 - acc: 0.5112 - val_loss: 1.9930 - val_acc: 0.4232
    PREDICT
    4070/4070 [==============================] - 5s     
    FIT PARAMS : 
    {'nb_filters_1': 32, 'nb_filters_2': 32, 'filter_size_2': 3, 'pool_size_1': 2, 'pool_size_2': 2, 'filter_size_1': 3}
    Train on 14650 samples, validate on 1628 samples
    Epoch 1/10
    14650/14650 [==============================] - 50s - loss: 2.3576 - acc: 0.3230 - val_loss: 2.1339 - val_acc: 0.3649
    Epoch 2/10
    14650/14650 [==============================] - 50s - loss: 2.1038 - acc: 0.3779 - val_loss: 2.0697 - val_acc: 0.3956
    Epoch 3/10
    14650/14650 [==============================] - 50s - loss: 1.9819 - acc: 0.4128 - val_loss: 1.9789 - val_acc: 0.4244
    Epoch 4/10
    14650/14650 [==============================] - 51s - loss: 1.9015 - acc: 0.4388 - val_loss: 1.9572 - val_acc: 0.4294
    Epoch 5/10
    14650/14650 [==============================] - 52s - loss: 1.8424 - acc: 0.4576 - val_loss: 1.9195 - val_acc: 0.4392
    Epoch 6/10
    14650/14650 [==============================] - 51s - loss: 1.7767 - acc: 0.4769 - val_loss: 1.9543 - val_acc: 0.4060
    Epoch 7/10
    14650/14650 [==============================] - 52s - loss: 1.7134 - acc: 0.4937 - val_loss: 1.9149 - val_acc: 0.4410
    Epoch 8/10
    14650/14650 [==============================] - 51s - loss: 1.6503 - acc: 0.5091 - val_loss: 1.9294 - val_acc: 0.4324
    Epoch 9/10
    14624/14650 [============================>.] - ETA: 0s - loss: 1.5899 - acc: 0.5223Epoch 00008: early stopping
    14650/14650 [==============================] - 53s - loss: 1.5893 - acc: 0.5226 - val_loss: 1.9329 - val_acc: 0.4545
    PREDICT
    4064/4070 [============================>.] - ETA: 0s[ 0.42653563  0.42997543  0.41965602]


#### Two CLs/PLs (with relu layers)


```python
def build_model(hp):
    net = Sequential()
    net.add(Convolution2D(hp['nb_filters_1'], hp['filter_size_1'], hp['filter_size_1'], border_mode='same', 
                          input_shape=(64,64,3)))
    net.add(Activation("relu"))
    net.add(MaxPooling2D(pool_size=(hp['pool_size_1'],hp['pool_size_1'])))
    net.add(Convolution2D(hp['nb_filters_2'], hp['filter_size_2'], hp['filter_size_2'], border_mode='same'))
    net.add(Activation("relu"))
    net.add(MaxPooling2D(pool_size=(hp['pool_size_2'],hp['pool_size_2'])))
    net.add(Flatten())
    net.add(Dense(output_dim=18))
    net.add(Activation("softmax"))
    
    net.compile(loss='categorical_crossentropy', optimizer='adadelta', metrics=['accuracy'])
    
    return net

print(unit_test(Classifier,nb_iter=3))
```
    FIT PARAMS : 
    {'nb_filters_1': 32, 'nb_filters_2': 32, 'filter_size_2': 3, 'pool_size_1': 2, 'pool_size_2': 2, 'filter_size_1': 3}
    Train on 14650 samples, validate on 1628 samples
    Epoch 1/10
    14650/14650 [==============================] - 55s - loss: 2.3761 - acc: 0.3235 - val_loss: 2.1801 - val_acc: 0.3925
    Epoch 2/10
    14650/14650 [==============================] - 56s - loss: 2.0877 - acc: 0.3816 - val_loss: 1.9894 - val_acc: 0.4122
    Epoch 3/10
    14650/14650 [==============================] - 57s - loss: 1.8972 - acc: 0.4303 - val_loss: 1.8884 - val_acc: 0.4441
    Epoch 4/10
    14650/14650 [==============================] - 55s - loss: 1.7812 - acc: 0.4644 - val_loss: 1.8573 - val_acc: 0.4527
    Epoch 5/10
    14650/14650 [==============================] - 56s - loss: 1.7079 - acc: 0.4852 - val_loss: 1.7866 - val_acc: 0.4674
    Epoch 6/10
    14650/14650 [==============================] - 55s - loss: 1.6492 - acc: 0.5005 - val_loss: 1.7709 - val_acc: 0.4810
    Epoch 7/10
    14650/14650 [==============================] - 55s - loss: 1.5961 - acc: 0.5145 - val_loss: 1.7410 - val_acc: 0.4877
    Epoch 8/10
    14650/14650 [==============================] - 56s - loss: 1.5426 - acc: 0.5287 - val_loss: 1.7394 - val_acc: 0.4828
    Epoch 9/10
    14650/14650 [==============================] - 57s - loss: 1.4929 - acc: 0.5443 - val_loss: 1.7156 - val_acc: 0.4902
    Epoch 10/10
    14650/14650 [==============================] - 53s - loss: 1.4405 - acc: 0.5597 - val_loss: 1.6891 - val_acc: 0.5055
    PREDICT
    4070/4070 [==============================] - 5s     
    FIT PARAMS : 
    {'nb_filters_1': 32, 'nb_filters_2': 32, 'filter_size_2': 3, 'pool_size_1': 2, 'pool_size_2': 2, 'filter_size_1': 3}
    Train on 14650 samples, validate on 1628 samples
    Epoch 1/10
    14650/14650 [==============================] - 53s - loss: 2.3415 - acc: 0.3358 - val_loss: 2.2248 - val_acc: 0.3575
    Epoch 2/10
    14650/14650 [==============================] - 52s - loss: 2.0613 - acc: 0.3870 - val_loss: 2.0163 - val_acc: 0.4023
    Epoch 3/10
    14650/14650 [==============================] - 52s - loss: 1.8915 - acc: 0.4302 - val_loss: 1.9275 - val_acc: 0.4306
    Epoch 4/10
    14650/14650 [==============================] - 53s - loss: 1.7808 - acc: 0.4612 - val_loss: 1.8434 - val_acc: 0.4441
    Epoch 5/10
    14650/14650 [==============================] - 53s - loss: 1.7028 - acc: 0.4814 - val_loss: 1.8097 - val_acc: 0.4496
    Epoch 6/10
    14650/14650 [==============================] - 53s - loss: 1.6396 - acc: 0.5046 - val_loss: 1.7716 - val_acc: 0.4693
    Epoch 7/10
    14650/14650 [==============================] - 53s - loss: 1.5811 - acc: 0.5189 - val_loss: 1.7617 - val_acc: 0.4668
    Epoch 8/10
    14650/14650 [==============================] - 53s - loss: 1.5258 - acc: 0.5349 - val_loss: 1.6793 - val_acc: 0.4914
    Epoch 9/10
    14650/14650 [==============================] - 53s - loss: 1.4722 - acc: 0.5523 - val_loss: 1.7343 - val_acc: 0.4607
    Epoch 10/10
    14624/14650 [============================>.] - ETA: 0s - loss: 1.4210 - acc: 0.5671Epoch 00009: early stopping
    14650/14650 [==============================] - 53s - loss: 1.4208 - acc: 0.5672 - val_loss: 1.6891 - val_acc: 0.4926
    PREDICT
    4070/4070 [==============================] - 5s     
    FIT PARAMS : 
    {'nb_filters_1': 32, 'nb_filters_2': 32, 'filter_size_2': 3, 'pool_size_1': 2, 'pool_size_2': 2, 'filter_size_1': 3}
    Train on 14650 samples, validate on 1628 samples
    Epoch 1/10
    14650/14650 [==============================] - 53s - loss: 2.3550 - acc: 0.3280 - val_loss: 2.1037 - val_acc: 0.3993
    Epoch 2/10
    14650/14650 [==============================] - 53s - loss: 2.0513 - acc: 0.3925 - val_loss: 1.9989 - val_acc: 0.4072
    Epoch 3/10
    14650/14650 [==============================] - 53s - loss: 1.9162 - acc: 0.4244 - val_loss: 1.8623 - val_acc: 0.4447
    Epoch 4/10
    14650/14650 [==============================] - 54s - loss: 1.8059 - acc: 0.4592 - val_loss: 1.7845 - val_acc: 0.4699
    Epoch 5/10
    14650/14650 [==============================] - 53s - loss: 1.7251 - acc: 0.4807 - val_loss: 1.7375 - val_acc: 0.4859
    Epoch 6/10
    14650/14650 [==============================] - 54s - loss: 1.6505 - acc: 0.5009 - val_loss: 1.7458 - val_acc: 0.4625
    Epoch 7/10
    14650/14650 [==============================] - 56s - loss: 1.5868 - acc: 0.5176 - val_loss: 1.6881 - val_acc: 0.4902
    Epoch 8/10
    14650/14650 [==============================] - 55s - loss: 1.5262 - acc: 0.5347 - val_loss: 1.6865 - val_acc: 0.4945
    Epoch 9/10
    14650/14650 [==============================] - 53s - loss: 1.4699 - acc: 0.5533 - val_loss: 1.6581 - val_acc: 0.5000
    Epoch 10/10
    14650/14650 [==============================] - 56s - loss: 1.4124 - acc: 0.5663 - val_loss: 1.9262 - val_acc: 0.4214
    PREDICT
    4070/4070 [==============================] - 6s     
    [ 0.5014742   0.5036855   0.44348894]


### Model architecture choice
- From the results of the tests listed above, we retain as best model architecture for feature extraction : CL/relu/PL(max)/CL/relu/PL(max).
- With this architecture, grid search will be performed to tune the number of filters in the CLs as well as their sizes, and the sizes of the pooling matrices.

## Hyperparameter optimization

### Using grid search to tune the number of filters, and the size of the filters / pooling matrices


```python
def build_model(hp):
    net = Sequential()
    net.add(Convolution2D(hp['nb_filters_1'], hp['filter_size_1'], hp['filter_size_1'], border_mode='same', 
                          input_shape=(64,64,3)))
    net.add(Activation("relu"))
    net.add(MaxPooling2D(pool_size=(hp['pool_size_1'],hp['pool_size_1'])))
    net.add(Convolution2D(hp['nb_filters_2'], hp['filter_size_2'], hp['filter_size_2'], border_mode='same'))
    net.add(Activation("relu"))
    net.add(MaxPooling2D(pool_size=(hp['pool_size_2'],hp['pool_size_2'])))
    net.add(Flatten())
    net.add(Dense(output_dim=18))
    net.add(Activation("softmax"))
    
    net.compile(loss='categorical_crossentropy', optimizer='adadelta', metrics=['accuracy'])
    
    return net

class Classifier(BaseEstimator):  

    def __init__(self, nb_filters_1=32, filter_size_1=3, pool_size_1=2,
                 nb_filters_2=32, filter_size_2=3, pool_size_2=2):
        self.nb_filters_1 = nb_filters_1
        self.filter_size_1 = filter_size_1
        self.pool_size_1 = pool_size_1
        self.nb_filters_2 = nb_filters_2
        self.filter_size_2 = filter_size_2
        self.pool_size_2 = pool_size_2
        
    def preprocess(self, X):
        X = X.reshape((X.shape[0],64,64,3))
        X = (X / 255.)
        X = X.astype(np.float32)
        return X
    
    def preprocess_y(self, y):
        return np_utils.to_categorical(y)
    
    def fit(self, X, y):
        X = self.preprocess(X)
        y = self.preprocess_y(y)
        
        hyper_parameters = dict(
        nb_filters_1 = self.nb_filters_1,
        filter_size_1 = self.filter_size_1,
        pool_size_1 = self.pool_size_1,
        nb_filters_2 = self.nb_filters_2,
        filter_size_2 = self.filter_size_2,
        pool_size_2 = self.pool_size_2
        )
        
        print("FIT PARAMS : ")
        print(hyper_parameters)
        
        self.model = build_model(hyper_parameters)
        
        earlyStopping = EarlyStopping(monitor='val_loss', patience=1, verbose=1, mode='auto')
        self.model.fit(X, y, nb_epoch=20, verbose=2, callbacks=[earlyStopping], validation_split=0.1, 
                       validation_data=None, shuffle=True)
        time.sleep(0.1)
        return self

    def predict(self, X):
        print("PREDICT")
        X = self.preprocess(X)
        return self.model.predict_classes(X)

    def predict_proba(self, X):
        X = self.preprocess(X)
        return self.model.predict(X)
    
    def score(self, X, y):
        print("SCORE")
        print(self.model.evaluate(self, X, y, batch_size=32, verbose=1, sample_weight=None))
        return self.model.evaluate(self, X, y, batch_size=32, verbose=1, sample_weight=None) 
    


params = {
    'nb_filters_1': [32,64],
    'filter_size_1': [3,6],
    'pool_size_1': [2,4],
    'nb_filters_2': [32,64],
    'filter_size_2': [3,6],
    'pool_size_2': [2,4]
}
clf = hyperparameter_optim(Classifier,params)

print("Detailed classification report:")
print()
y_true, y_pred = y_test, clf.predict(X_test)
print(classification_report(y_true, y_pred))
```
    FIT PARAMS : 
    {'nb_filters_1': 32, 'nb_filters_2': 32, 'filter_size_2': 3, 'pool_size_1': 2, 'pool_size_2': 2, 'filter_size_1': 3}
    Train on 10987 samples, validate on 1221 samples
    Epoch 1/20
    6s - loss: 2.4039 - acc: 0.3143 - val_loss: 2.2342 - val_acc: 0.3587
    Epoch 2/20
    6s - loss: 2.1300 - acc: 0.3816 - val_loss: 2.0607 - val_acc: 0.3784
    Epoch 3/20
    6s - loss: 1.9607 - acc: 0.4113 - val_loss: 1.9362 - val_acc: 0.4357
    Epoch 4/20
    6s - loss: 1.8300 - acc: 0.4507 - val_loss: 1.8738 - val_acc: 0.4455
    Epoch 5/20
    6s - loss: 1.7400 - acc: 0.4738 - val_loss: 1.8514 - val_acc: 0.4611
    Epoch 6/20
    6s - loss: 1.6644 - acc: 0.4934 - val_loss: 1.8412 - val_acc: 0.4578
    Epoch 7/20
    6s - loss: 1.5986 - acc: 0.5159 - val_loss: 1.9484 - val_acc: 0.3964
    Epoch 8/20
    5s - loss: 1.5498 - acc: 0.5257 - val_loss: 1.7547 - val_acc: 0.4840
    Epoch 9/20
    6s - loss: 1.4968 - acc: 0.5429 - val_loss: 1.8006 - val_acc: 0.4808
    Epoch 10/20
    Epoch 00009: early stopping
    5s - loss: 1.4418 - acc: 0.5577 - val_loss: 1.8119 - val_acc: 0.4668
    PREDICT
    6105/6105 [==============================] - 1s     
    PREDICT
    12192/12208 [============================>.] - ETA: 0sFIT PARAMS : 
    {'nb_filters_1': 32, 'nb_filters_2': 32, 'filter_size_2': 3, 'pool_size_1': 2, 'pool_size_2': 2, 'filter_size_1': 3}
    Train on 10988 samples, validate on 1221 samples
    Epoch 1/20
    6s - loss: 2.3886 - acc: 0.3181 - val_loss: 2.1972 - val_acc: 0.3686
    Epoch 2/20
    5s - loss: 2.1023 - acc: 0.3855 - val_loss: 2.0830 - val_acc: 0.3702
    Epoch 3/20
    5s - loss: 1.9618 - acc: 0.4089 - val_loss: 1.9376 - val_acc: 0.4333
    Epoch 4/20
    5s - loss: 1.8556 - acc: 0.4387 - val_loss: 1.8704 - val_acc: 0.4480
    Epoch 5/20
    6s - loss: 1.7725 - acc: 0.4649 - val_loss: 1.8190 - val_acc: 0.4726
    Epoch 6/20
    5s - loss: 1.7045 - acc: 0.4831 - val_loss: 1.7835 - val_acc: 0.4898
    Epoch 7/20
    5s - loss: 1.6417 - acc: 0.5035 - val_loss: 1.7641 - val_acc: 0.4889
    Epoch 8/20
    5s - loss: 1.5864 - acc: 0.5146 - val_loss: 1.7717 - val_acc: 0.4939
    Epoch 9/20
    5s - loss: 1.5307 - acc: 0.5346 - val_loss: 1.7448 - val_acc: 0.4996
    Epoch 10/20
    5s - loss: 1.4829 - acc: 0.5450 - val_loss: 1.7209 - val_acc: 0.5004
    Epoch 11/20
    5s - loss: 1.4297 - acc: 0.5654 - val_loss: 1.6990 - val_acc: 0.5053
    Epoch 12/20
    5s - loss: 1.3762 - acc: 0.5744 - val_loss: 1.6935 - val_acc: 0.4939
    Epoch 13/20
    5s - loss: 1.3298 - acc: 0.5918 - val_loss: 1.7646 - val_acc: 0.4693
    Epoch 14/20
    5s - loss: 1.2746 - acc: 0.6077 - val_loss: 1.6829 - val_acc: 0.5176
    Epoch 15/20
    5s - loss: 1.2243 - acc: 0.6203 - val_loss: 1.8035 - val_acc: 0.4496
    Epoch 16/20
    Epoch 00015: early stopping
    5s - loss: 1.1762 - acc: 0.6375 - val_loss: 1.7732 - val_acc: 0.4644
    PREDICT
    6104/6104 [==============================] - 1s     
    PREDICT
    12192/12209 [============================>.] - ETA: 0sFIT PARAMS : 
    {'nb_filters_1': 32, 'nb_filters_2': 32, 'filter_size_2': 3, 'pool_size_1': 2, 'pool_size_2': 2, 'filter_size_1': 3}
    Train on 10988 samples, validate on 1221 samples
    Epoch 1/20
    6s - loss: 2.3903 - acc: 0.3182 - val_loss: 2.2172 - val_acc: 0.3694
    Epoch 2/20
    5s - loss: 2.1067 - acc: 0.3823 - val_loss: 2.0580 - val_acc: 0.3890
    Epoch 3/20
    5s - loss: 1.9421 - acc: 0.4168 - val_loss: 1.9302 - val_acc: 0.4382
    Epoch 4/20
    5s - loss: 1.8208 - acc: 0.4517 - val_loss: 1.8764 - val_acc: 0.4455
    Epoch 5/20
    5s - loss: 1.7356 - acc: 0.4748 - val_loss: 1.8097 - val_acc: 0.4693
    Epoch 6/20
    5s - loss: 1.6635 - acc: 0.4938 - val_loss: 1.7998 - val_acc: 0.4676
    Epoch 7/20
    5s - loss: 1.6020 - acc: 0.5116 - val_loss: 1.7531 - val_acc: 0.4848
    Epoch 8/20
    5s - loss: 1.5484 - acc: 0.5276 - val_loss: 1.7234 - val_acc: 0.5037
    Epoch 9/20
    5s - loss: 1.4909 - acc: 0.5428 - val_loss: 1.7879 - val_acc: 0.4562
    Epoch 10/20
    Epoch 00009: early stopping
    5s - loss: 1.4421 - acc: 0.5597 - val_loss: 1.7297 - val_acc: 0.5143
    PREDICT
    6104/6104 [==============================] - 1s     
    PREDICT
    12192/12209 [============================>.] - ETA: 0sFIT PARAMS : 
    {'nb_filters_1': 32, 'nb_filters_2': 32, 'filter_size_2': 3, 'pool_size_1': 2, 'pool_size_2': 4, 'filter_size_1': 3}
    Train on 10987 samples, validate on 1221 samples
    Epoch 1/20
    6s - loss: 2.4184 - acc: 0.3127 - val_loss: 2.2901 - val_acc: 0.3579
    Epoch 2/20
    5s - loss: 2.1787 - acc: 0.3757 - val_loss: 2.1901 - val_acc: 0.3808
    Epoch 3/20
    5s - loss: 2.0649 - acc: 0.3929 - val_loss: 2.0441 - val_acc: 0.3964
    Epoch 4/20
    5s - loss: 1.9579 - acc: 0.4120 - val_loss: 1.9736 - val_acc: 0.4373
    Epoch 5/20
    5s - loss: 1.8791 - acc: 0.4324 - val_loss: 1.9036 - val_acc: 0.4464
    Epoch 6/20
    5s - loss: 1.8189 - acc: 0.4515 - val_loss: 1.8659 - val_acc: 0.4554
    Epoch 7/20
    5s - loss: 1.7637 - acc: 0.4665 - val_loss: 1.8474 - val_acc: 0.4554
    Epoch 8/20
    5s - loss: 1.7194 - acc: 0.4821 - val_loss: 1.8235 - val_acc: 0.4750
    Epoch 9/20
    5s - loss: 1.6762 - acc: 0.4935 - val_loss: 1.7785 - val_acc: 0.4799
    Epoch 10/20
    5s - loss: 1.6367 - acc: 0.5050 - val_loss: 1.7783 - val_acc: 0.4799
    Epoch 11/20
    5s - loss: 1.5982 - acc: 0.5183 - val_loss: 1.7427 - val_acc: 0.4914
    Epoch 12/20
    5s - loss: 1.5603 - acc: 0.5271 - val_loss: 1.7501 - val_acc: 0.4889
    Epoch 13/20
    5s - loss: 1.5235 - acc: 0.5344 - val_loss: 1.7310 - val_acc: 0.4758
    Epoch 14/20
    5s - loss: 1.4895 - acc: 0.5455 - val_loss: 1.7012 - val_acc: 0.4980
    Epoch 15/20
    5s - loss: 1.4591 - acc: 0.5542 - val_loss: 1.7215 - val_acc: 0.4824
    Epoch 16/20
    5s - loss: 1.4307 - acc: 0.5645 - val_loss: 1.6716 - val_acc: 0.4963
    Epoch 17/20
    5s - loss: 1.3949 - acc: 0.5746 - val_loss: 1.7081 - val_acc: 0.5004
    Epoch 18/20
    5s - loss: 1.3707 - acc: 0.5821 - val_loss: 1.6278 - val_acc: 0.5143
    Epoch 19/20
    5s - loss: 1.3381 - acc: 0.5863 - val_loss: 1.6448 - val_acc: 0.5012
    Epoch 20/20
    5s - loss: 1.3141 - acc: 0.5986 - val_loss: 1.6244 - val_acc: 0.5135
    PREDICT
    6105/6105 [==============================] - 1s     
    PREDICT
    12208/12208 [==============================] - 2s     
    FIT PARAMS : 
    {'nb_filters_1': 32, 'nb_filters_2': 32, 'filter_size_2': 3, 'pool_size_1': 2, 'pool_size_2': 4, 'filter_size_1': 3}
    Train on 10988 samples, validate on 1221 samples
    Epoch 1/20
    7s - loss: 2.4452 - acc: 0.3036 - val_loss: 2.2437 - val_acc: 0.3612
    Epoch 2/20
    5s - loss: 2.1629 - acc: 0.3737 - val_loss: 2.0670 - val_acc: 0.3923
    Epoch 3/20
    5s - loss: 2.0170 - acc: 0.4027 - val_loss: 2.0325 - val_acc: 0.4357
    Epoch 4/20
    5s - loss: 1.9323 - acc: 0.4147 - val_loss: 1.9238 - val_acc: 0.4316
    Epoch 5/20
    5s - loss: 1.8665 - acc: 0.4359 - val_loss: 1.8908 - val_acc: 0.4349
    Epoch 6/20
    5s - loss: 1.8082 - acc: 0.4511 - val_loss: 1.8181 - val_acc: 0.4537
    Epoch 7/20
    5s - loss: 1.7487 - acc: 0.4697 - val_loss: 1.7973 - val_acc: 0.4750
    Epoch 8/20
    5s - loss: 1.7005 - acc: 0.4859 - val_loss: 1.8094 - val_acc: 0.4816
    Epoch 9/20
    5s - loss: 1.6564 - acc: 0.4973 - val_loss: 1.7223 - val_acc: 0.5004
    Epoch 10/20
    5s - loss: 1.6170 - acc: 0.5048 - val_loss: 1.6943 - val_acc: 0.5020
    Epoch 11/20
    5s - loss: 1.5782 - acc: 0.5209 - val_loss: 1.6753 - val_acc: 0.5119
    Epoch 12/20
    5s - loss: 1.5484 - acc: 0.5237 - val_loss: 1.6863 - val_acc: 0.4808
    Epoch 13/20
    5s - loss: 1.5150 - acc: 0.5372 - val_loss: 1.6505 - val_acc: 0.5192
    Epoch 14/20
    5s - loss: 1.4859 - acc: 0.5424 - val_loss: 1.7120 - val_acc: 0.5061
    Epoch 15/20
    5s - loss: 1.4585 - acc: 0.5491 - val_loss: 1.6190 - val_acc: 0.5209
    Epoch 16/20
    5s - loss: 1.4309 - acc: 0.5617 - val_loss: 1.6041 - val_acc: 0.5340
    Epoch 17/20
    5s - loss: 1.4072 - acc: 0.5643 - val_loss: 1.5984 - val_acc: 0.5364
    Epoch 18/20
    5s - loss: 1.3797 - acc: 0.5748 - val_loss: 1.5922 - val_acc: 0.5111
    Epoch 19/20
    5s - loss: 1.3544 - acc: 0.5863 - val_loss: 1.6270 - val_acc: 0.5192
    Epoch 20/20
    Epoch 00019: early stopping
    5s - loss: 1.3306 - acc: 0.5917 - val_loss: 1.6046 - val_acc: 0.5242
    PREDICT
    6104/6104 [==============================] - 1s     
    PREDICT
    12192/12209 [============================>.] - ETA: 0sFIT PARAMS : 
    {'nb_filters_1': 32, 'nb_filters_2': 32, 'filter_size_2': 3, 'pool_size_1': 2, 'pool_size_2': 4, 'filter_size_1': 3}
    Train on 10988 samples, validate on 1221 samples
    Epoch 1/20
    6s - loss: 2.4236 - acc: 0.3102 - val_loss: 2.2458 - val_acc: 0.3825
    Epoch 2/20
    5s - loss: 2.2020 - acc: 0.3682 - val_loss: 2.1324 - val_acc: 0.3948
    Epoch 3/20
    5s - loss: 2.0905 - acc: 0.3806 - val_loss: 2.0092 - val_acc: 0.4169
    Epoch 4/20
    5s - loss: 1.9680 - acc: 0.4086 - val_loss: 1.9574 - val_acc: 0.4161
    Epoch 5/20
    5s - loss: 1.8889 - acc: 0.4224 - val_loss: 1.9135 - val_acc: 0.4267
    Epoch 6/20
    5s - loss: 1.8321 - acc: 0.4432 - val_loss: 1.8739 - val_acc: 0.4603
    Epoch 7/20
    5s - loss: 1.7798 - acc: 0.4586 - val_loss: 1.8371 - val_acc: 0.4513
    Epoch 8/20
    5s - loss: 1.7332 - acc: 0.4745 - val_loss: 1.7971 - val_acc: 0.4865
    Epoch 9/20
    5s - loss: 1.6888 - acc: 0.4861 - val_loss: 1.7463 - val_acc: 0.5020
    Epoch 10/20
    5s - loss: 1.6529 - acc: 0.4994 - val_loss: 1.7258 - val_acc: 0.4963
    Epoch 11/20
    5s - loss: 1.6167 - acc: 0.5083 - val_loss: 1.7205 - val_acc: 0.4873
    Epoch 12/20
    5s - loss: 1.5827 - acc: 0.5190 - val_loss: 1.7243 - val_acc: 0.4930
    Epoch 13/20
    5s - loss: 1.5485 - acc: 0.5276 - val_loss: 1.6558 - val_acc: 0.5291
    Epoch 14/20
    5s - loss: 1.5197 - acc: 0.5362 - val_loss: 1.7532 - val_acc: 0.4791
    Epoch 15/20
    Epoch 00014: early stopping
    5s - loss: 1.4864 - acc: 0.5489 - val_loss: 1.6647 - val_acc: 0.5094
    PREDICT
    6104/6104 [==============================] - 1s     
    PREDICT
    12192/12209 [============================>.] - ETA: 0sFIT PARAMS : 
    {'nb_filters_1': 32, 'nb_filters_2': 32, 'filter_size_2': 3, 'pool_size_1': 4, 'pool_size_2': 2, 'filter_size_1': 3}
    Train on 10987 samples, validate on 1221 samples
    Epoch 1/20
    5s - loss: 2.4469 - acc: 0.2980 - val_loss: 2.2702 - val_acc: 0.3579
    Epoch 2/20
    4s - loss: 2.1842 - acc: 0.3744 - val_loss: 2.0939 - val_acc: 0.3874
    Epoch 3/20
    4s - loss: 2.0402 - acc: 0.3960 - val_loss: 1.9895 - val_acc: 0.4046
    Epoch 4/20
    4s - loss: 1.9468 - acc: 0.4188 - val_loss: 1.9948 - val_acc: 0.4480
    Epoch 5/20
    4s - loss: 1.8829 - acc: 0.4388 - val_loss: 1.8906 - val_acc: 0.4423
    Epoch 6/20
    4s - loss: 1.8278 - acc: 0.4606 - val_loss: 1.8659 - val_acc: 0.4595
    Epoch 7/20
    4s - loss: 1.7808 - acc: 0.4724 - val_loss: 1.8953 - val_acc: 0.4423
    Epoch 8/20
    4s - loss: 1.7400 - acc: 0.4863 - val_loss: 1.7869 - val_acc: 0.4750
    Epoch 9/20
    4s - loss: 1.7058 - acc: 0.4932 - val_loss: 1.7854 - val_acc: 0.4914
    Epoch 10/20
    4s - loss: 1.6740 - acc: 0.5018 - val_loss: 1.7812 - val_acc: 0.4701
    Epoch 11/20
    4s - loss: 1.6410 - acc: 0.5084 - val_loss: 1.7535 - val_acc: 0.4980
    Epoch 12/20
    4s - loss: 1.6149 - acc: 0.5126 - val_loss: 1.7121 - val_acc: 0.4996
    Epoch 13/20
    4s - loss: 1.5879 - acc: 0.5205 - val_loss: 1.7485 - val_acc: 0.4922
    Epoch 14/20
    Epoch 00013: early stopping
    4s - loss: 1.5609 - acc: 0.5287 - val_loss: 1.7792 - val_acc: 0.4808
    PREDICT
    6105/6105 [==============================] - 1s     
    PREDICT
    12192/12208 [============================>.] - ETA: 0sFIT PARAMS : 
    {'nb_filters_1': 32, 'nb_filters_2': 32, 'filter_size_2': 3, 'pool_size_1': 4, 'pool_size_2': 2, 'filter_size_1': 3}
    Train on 10988 samples, validate on 1221 samples
    Epoch 1/20
    5s - loss: 2.4599 - acc: 0.2899 - val_loss: 2.3206 - val_acc: 0.3464
    Epoch 2/20
    4s - loss: 2.2327 - acc: 0.3611 - val_loss: 2.1594 - val_acc: 0.3726
    Epoch 3/20
    4s - loss: 2.0958 - acc: 0.3832 - val_loss: 2.0657 - val_acc: 0.4111
    Epoch 4/20
    4s - loss: 1.9944 - acc: 0.4018 - val_loss: 1.9616 - val_acc: 0.4177
    Epoch 5/20
    4s - loss: 1.9217 - acc: 0.4212 - val_loss: 1.9220 - val_acc: 0.4242
    Epoch 6/20
    4s - loss: 1.8644 - acc: 0.4351 - val_loss: 1.8759 - val_acc: 0.4464
    Epoch 7/20
    4s - loss: 1.8188 - acc: 0.4511 - val_loss: 1.8639 - val_acc: 0.4447
    Epoch 8/20
    4s - loss: 1.7743 - acc: 0.4652 - val_loss: 1.8015 - val_acc: 0.4767
    Epoch 9/20
    4s - loss: 1.7379 - acc: 0.4730 - val_loss: 1.7640 - val_acc: 0.4767
    Epoch 10/20
    4s - loss: 1.7044 - acc: 0.4813 - val_loss: 1.7427 - val_acc: 0.4996
    Epoch 11/20
    4s - loss: 1.6750 - acc: 0.4921 - val_loss: 1.7319 - val_acc: 0.4799
    Epoch 12/20
    4s - loss: 1.6467 - acc: 0.4966 - val_loss: 1.7189 - val_acc: 0.5078
    Epoch 13/20
    4s - loss: 1.6223 - acc: 0.5063 - val_loss: 1.6774 - val_acc: 0.5061
    Epoch 14/20
    4s - loss: 1.5970 - acc: 0.5167 - val_loss: 1.6859 - val_acc: 0.5061
    Epoch 15/20
    4s - loss: 1.5761 - acc: 0.5165 - val_loss: 1.6564 - val_acc: 0.5102
    Epoch 16/20
    4s - loss: 1.5510 - acc: 0.5218 - val_loss: 1.6746 - val_acc: 0.4971
    Epoch 17/20
    Epoch 00016: early stopping
    4s - loss: 1.5315 - acc: 0.5335 - val_loss: 1.6616 - val_acc: 0.5233
    PREDICT
    6104/6104 [==============================] - 1s     
    PREDICT
    12192/12209 [============================>.] - ETA: 0sFIT PARAMS : 
    {'nb_filters_1': 32, 'nb_filters_2': 32, 'filter_size_2': 3, 'pool_size_1': 4, 'pool_size_2': 2, 'filter_size_1': 3}
    Train on 10988 samples, validate on 1221 samples
    Epoch 1/20
    5s - loss: 2.4634 - acc: 0.2929 - val_loss: 2.3194 - val_acc: 0.3726
    Epoch 2/20
    4s - loss: 2.2421 - acc: 0.3586 - val_loss: 2.1858 - val_acc: 0.3866
    Epoch 3/20
    4s - loss: 2.1339 - acc: 0.3746 - val_loss: 2.0902 - val_acc: 0.3956
    Epoch 4/20
    4s - loss: 2.0402 - acc: 0.3945 - val_loss: 2.0357 - val_acc: 0.4054
    Epoch 5/20
    4s - loss: 1.9456 - acc: 0.4188 - val_loss: 1.9070 - val_acc: 0.4480
    Epoch 6/20
    4s - loss: 1.8581 - acc: 0.4448 - val_loss: 1.8385 - val_acc: 0.4570
    Epoch 7/20
    4s - loss: 1.7992 - acc: 0.4590 - val_loss: 1.8152 - val_acc: 0.4889
    Epoch 8/20
    4s - loss: 1.7543 - acc: 0.4746 - val_loss: 1.8094 - val_acc: 0.4865
    Epoch 9/20
    4s - loss: 1.7189 - acc: 0.4826 - val_loss: 1.7892 - val_acc: 0.4783
    Epoch 10/20
    4s - loss: 1.6895 - acc: 0.4894 - val_loss: 1.7441 - val_acc: 0.4881
    Epoch 11/20
    4s - loss: 1.6594 - acc: 0.4947 - val_loss: 1.7555 - val_acc: 0.4824
    Epoch 12/20
    4s - loss: 1.6307 - acc: 0.5072 - val_loss: 1.7308 - val_acc: 0.5037
    Epoch 13/20
    4s - loss: 1.6046 - acc: 0.5125 - val_loss: 1.8148 - val_acc: 0.4701
    Epoch 14/20
    4s - loss: 1.5778 - acc: 0.5177 - val_loss: 1.6871 - val_acc: 0.5143
    Epoch 15/20
    4s - loss: 1.5535 - acc: 0.5233 - val_loss: 1.7003 - val_acc: 0.5037
    Epoch 16/20
    4s - loss: 1.5318 - acc: 0.5344 - val_loss: 1.6424 - val_acc: 0.5201
    Epoch 17/20
    4s - loss: 1.5089 - acc: 0.5390 - val_loss: 1.6944 - val_acc: 0.5037
    Epoch 18/20
    Epoch 00017: early stopping
    4s - loss: 1.4899 - acc: 0.5439 - val_loss: 1.6962 - val_acc: 0.5143
    PREDICT
    6104/6104 [==============================] - 1s     
    PREDICT
    12192/12209 [============================>.] - ETA: 0sFIT PARAMS : 
    {'nb_filters_1': 32, 'nb_filters_2': 32, 'filter_size_2': 3, 'pool_size_1': 4, 'pool_size_2': 4, 'filter_size_1': 3}
    Train on 10987 samples, validate on 1221 samples
    Epoch 1/20
    5s - loss: 2.4766 - acc: 0.2874 - val_loss: 2.3989 - val_acc: 0.2998
    Epoch 2/20
    4s - loss: 2.2747 - acc: 0.3606 - val_loss: 2.1711 - val_acc: 0.3686
    Epoch 3/20
    4s - loss: 2.1369 - acc: 0.3862 - val_loss: 2.0727 - val_acc: 0.3989
    Epoch 4/20
    4s - loss: 2.0618 - acc: 0.3968 - val_loss: 2.0362 - val_acc: 0.3989
    Epoch 5/20
    4s - loss: 2.0112 - acc: 0.4034 - val_loss: 1.9848 - val_acc: 0.4062
    Epoch 6/20
    4s - loss: 1.9653 - acc: 0.4160 - val_loss: 1.9606 - val_acc: 0.4316
    Epoch 7/20
    4s - loss: 1.9180 - acc: 0.4265 - val_loss: 1.8948 - val_acc: 0.4259
    Epoch 8/20
    4s - loss: 1.8724 - acc: 0.4386 - val_loss: 1.8834 - val_acc: 0.4382
    Epoch 9/20
    4s - loss: 1.8307 - acc: 0.4504 - val_loss: 1.9390 - val_acc: 0.4570
    Epoch 10/20
    4s - loss: 1.7924 - acc: 0.4618 - val_loss: 1.8342 - val_acc: 0.4709
    Epoch 11/20
    4s - loss: 1.7565 - acc: 0.4732 - val_loss: 1.7964 - val_acc: 0.4636
    Epoch 12/20
    4s - loss: 1.7259 - acc: 0.4842 - val_loss: 1.7743 - val_acc: 0.4676
    Epoch 13/20
    4s - loss: 1.6954 - acc: 0.4909 - val_loss: 1.7503 - val_acc: 0.4767
    Epoch 14/20
    4s - loss: 1.6706 - acc: 0.4943 - val_loss: 1.7045 - val_acc: 0.4832
    Epoch 15/20
    4s - loss: 1.6451 - acc: 0.5065 - val_loss: 1.7025 - val_acc: 0.4848
    Epoch 16/20
    4s - loss: 1.6227 - acc: 0.5137 - val_loss: 1.7006 - val_acc: 0.5061
    Epoch 17/20
    4s - loss: 1.6010 - acc: 0.5180 - val_loss: 1.6747 - val_acc: 0.4988
    Epoch 18/20
    4s - loss: 1.5788 - acc: 0.5228 - val_loss: 1.6806 - val_acc: 0.5160
    Epoch 19/20
    4s - loss: 1.5602 - acc: 0.5320 - val_loss: 1.6353 - val_acc: 0.5094
    Epoch 20/20
    4s - loss: 1.5430 - acc: 0.5306 - val_loss: 1.6384 - val_acc: 0.5242
    PREDICT
    6105/6105 [==============================] - 1s     
    PREDICT
    12192/12208 [============================>.] - ETA: 0sFIT PARAMS : 
    {'nb_filters_1': 32, 'nb_filters_2': 32, 'filter_size_2': 3, 'pool_size_1': 4, 'pool_size_2': 4, 'filter_size_1': 3}
    Train on 10988 samples, validate on 1221 samples
    Epoch 1/20
    5s - loss: 2.4755 - acc: 0.2886 - val_loss: 2.3861 - val_acc: 0.2981
    Epoch 2/20
    4s - loss: 2.2990 - acc: 0.3469 - val_loss: 2.1994 - val_acc: 0.3628
    Epoch 3/20
    4s - loss: 2.1676 - acc: 0.3746 - val_loss: 2.1083 - val_acc: 0.3833
    Epoch 4/20
    4s - loss: 2.0849 - acc: 0.3892 - val_loss: 2.0415 - val_acc: 0.4038
    Epoch 5/20
    4s - loss: 2.0223 - acc: 0.3963 - val_loss: 1.9890 - val_acc: 0.4062
    Epoch 6/20
    4s - loss: 1.9742 - acc: 0.4073 - val_loss: 1.9516 - val_acc: 0.4136
    Epoch 7/20
    4s - loss: 1.9297 - acc: 0.4172 - val_loss: 1.9256 - val_acc: 0.4242
    Epoch 8/20
    4s - loss: 1.8855 - acc: 0.4327 - val_loss: 1.8774 - val_acc: 0.4373
    Epoch 9/20
    4s - loss: 1.8472 - acc: 0.4444 - val_loss: 1.8400 - val_acc: 0.4423
    Epoch 10/20
    4s - loss: 1.8120 - acc: 0.4532 - val_loss: 1.8191 - val_acc: 0.4455
    Epoch 11/20
    4s - loss: 1.7767 - acc: 0.4652 - val_loss: 1.7767 - val_acc: 0.4644
    Epoch 12/20
    4s - loss: 1.7473 - acc: 0.4742 - val_loss: 1.7743 - val_acc: 0.4668
    Epoch 13/20
    4s - loss: 1.7186 - acc: 0.4824 - val_loss: 1.7450 - val_acc: 0.4627
    Epoch 14/20
    4s - loss: 1.6939 - acc: 0.4895 - val_loss: 1.7221 - val_acc: 0.4709
    Epoch 15/20
    4s - loss: 1.6669 - acc: 0.4916 - val_loss: 1.7075 - val_acc: 0.4988
    Epoch 16/20
    4s - loss: 1.6467 - acc: 0.5004 - val_loss: 1.6813 - val_acc: 0.4873
    Epoch 17/20
    4s - loss: 1.6264 - acc: 0.5063 - val_loss: 1.6471 - val_acc: 0.4988
    Epoch 18/20
    4s - loss: 1.6031 - acc: 0.5121 - val_loss: 1.6334 - val_acc: 0.5053
    Epoch 19/20
    4s - loss: 1.5841 - acc: 0.5177 - val_loss: 1.6147 - val_acc: 0.5004
    Epoch 20/20
    4s - loss: 1.5672 - acc: 0.5248 - val_loss: 1.6010 - val_acc: 0.5184
    PREDICT
    6104/6104 [==============================] - 1s     
    PREDICT
    12192/12209 [============================>.] - ETA: 0sFIT PARAMS : 
    {'nb_filters_1': 32, 'nb_filters_2': 32, 'filter_size_2': 3, 'pool_size_1': 4, 'pool_size_2': 4, 'filter_size_1': 3}
    Train on 10988 samples, validate on 1221 samples
    Epoch 1/20
    5s - loss: 2.4786 - acc: 0.2846 - val_loss: 2.3916 - val_acc: 0.3047
    Epoch 2/20
    4s - loss: 2.2946 - acc: 0.3477 - val_loss: 2.2003 - val_acc: 0.3948
    Epoch 3/20
    4s - loss: 2.1409 - acc: 0.3799 - val_loss: 2.0763 - val_acc: 0.4079
    Epoch 4/20
    4s - loss: 2.0717 - acc: 0.3905 - val_loss: 2.0275 - val_acc: 0.4185
    Epoch 5/20
    4s - loss: 2.0215 - acc: 0.3977 - val_loss: 1.9888 - val_acc: 0.4242
    Epoch 6/20
    4s - loss: 1.9804 - acc: 0.4029 - val_loss: 1.9914 - val_acc: 0.4079
    Epoch 7/20
    4s - loss: 1.9397 - acc: 0.4141 - val_loss: 1.9301 - val_acc: 0.4251
    Epoch 8/20
    4s - loss: 1.9014 - acc: 0.4226 - val_loss: 1.9371 - val_acc: 0.4226
    Epoch 9/20
    4s - loss: 1.8683 - acc: 0.4342 - val_loss: 1.8701 - val_acc: 0.4570
    Epoch 10/20
    4s - loss: 1.8327 - acc: 0.4446 - val_loss: 1.8359 - val_acc: 0.4693
    Epoch 11/20
    4s - loss: 1.8029 - acc: 0.4540 - val_loss: 1.8227 - val_acc: 0.4586
    Epoch 12/20
    4s - loss: 1.7748 - acc: 0.4656 - val_loss: 1.7918 - val_acc: 0.4717
    Epoch 13/20
    4s - loss: 1.7455 - acc: 0.4738 - val_loss: 1.7888 - val_acc: 0.4685
    Epoch 14/20
    4s - loss: 1.7180 - acc: 0.4794 - val_loss: 1.7852 - val_acc: 0.4742
    Epoch 15/20
    4s - loss: 1.6985 - acc: 0.4849 - val_loss: 1.7513 - val_acc: 0.4824
    Epoch 16/20
    4s - loss: 1.6726 - acc: 0.4941 - val_loss: 1.7187 - val_acc: 0.4906
    Epoch 17/20
    4s - loss: 1.6560 - acc: 0.4979 - val_loss: 1.6906 - val_acc: 0.4996
    Epoch 18/20
    4s - loss: 1.6345 - acc: 0.5031 - val_loss: 1.7457 - val_acc: 0.4627
    Epoch 19/20
    Epoch 00018: early stopping
    4s - loss: 1.6149 - acc: 0.5105 - val_loss: 1.7210 - val_acc: 0.5004
    PREDICT
    6104/6104 [==============================] - 1s     
    PREDICT
    12192/12209 [============================>.] - ETA: 0sFIT PARAMS : 
    {'nb_filters_1': 32, 'nb_filters_2': 64, 'filter_size_2': 3, 'pool_size_1': 2, 'pool_size_2': 2, 'filter_size_1': 3}
    Train on 10987 samples, validate on 1221 samples
    Epoch 1/20
    8s - loss: 2.3714 - acc: 0.3227 - val_loss: 2.1621 - val_acc: 0.3866
    Epoch 2/20
    7s - loss: 2.0265 - acc: 0.4042 - val_loss: 1.9896 - val_acc: 0.4439
    Epoch 3/20
    7s - loss: 1.8395 - acc: 0.4504 - val_loss: 1.8505 - val_acc: 0.4578
    Epoch 4/20
    7s - loss: 1.7208 - acc: 0.4836 - val_loss: 1.7966 - val_acc: 0.4824
    Epoch 5/20
    7s - loss: 1.6371 - acc: 0.5118 - val_loss: 1.7525 - val_acc: 0.4939
    Epoch 6/20
    7s - loss: 1.5678 - acc: 0.5241 - val_loss: 1.7421 - val_acc: 0.4930
    Epoch 7/20
    7s - loss: 1.4943 - acc: 0.5448 - val_loss: 1.7376 - val_acc: 0.4939
    Epoch 8/20
    7s - loss: 1.4235 - acc: 0.5658 - val_loss: 1.6847 - val_acc: 0.5111
    Epoch 9/20
    7s - loss: 1.3547 - acc: 0.5840 - val_loss: 1.7051 - val_acc: 0.5012
    Epoch 10/20
    Epoch 00009: early stopping
    7s - loss: 1.2879 - acc: 0.6027 - val_loss: 1.7407 - val_acc: 0.4758
    PREDICT
    6105/6105 [==============================] - 2s     
    PREDICT
    12208/12208 [==============================] - 3s     
    FIT PARAMS : 
    {'nb_filters_1': 32, 'nb_filters_2': 64, 'filter_size_2': 3, 'pool_size_1': 2, 'pool_size_2': 2, 'filter_size_1': 3}
    Train on 10988 samples, validate on 1221 samples
    Epoch 1/20
    8s - loss: 2.3651 - acc: 0.3278 - val_loss: 2.1788 - val_acc: 0.3677
    Epoch 2/20
    7s - loss: 2.0949 - acc: 0.3825 - val_loss: 2.0540 - val_acc: 0.3718
    Epoch 3/20
    7s - loss: 1.9061 - acc: 0.4240 - val_loss: 1.9155 - val_acc: 0.4423
    Epoch 4/20
    7s - loss: 1.7808 - acc: 0.4609 - val_loss: 1.8024 - val_acc: 0.4717
    Epoch 5/20
    7s - loss: 1.6822 - acc: 0.4904 - val_loss: 1.7739 - val_acc: 0.4808
    Epoch 6/20
    7s - loss: 1.6043 - acc: 0.5113 - val_loss: 1.7325 - val_acc: 0.4783
    Epoch 7/20
    7s - loss: 1.5243 - acc: 0.5355 - val_loss: 1.8330 - val_acc: 0.4488
    Epoch 8/20
    7s - loss: 1.4530 - acc: 0.5560 - val_loss: 1.6974 - val_acc: 0.4947
    Epoch 9/20
    7s - loss: 1.3773 - acc: 0.5752 - val_loss: 1.6842 - val_acc: 0.5004
    Epoch 10/20
    7s - loss: 1.3044 - acc: 0.5989 - val_loss: 1.7739 - val_acc: 0.4955
    Epoch 11/20
    7s - loss: 1.2234 - acc: 0.6247 - val_loss: 1.6731 - val_acc: 0.5209
    Epoch 12/20
    7s - loss: 1.1472 - acc: 0.6432 - val_loss: 1.6703 - val_acc: 0.4971
    Epoch 13/20
    7s - loss: 1.0714 - acc: 0.6645 - val_loss: 1.7488 - val_acc: 0.4930
    Epoch 14/20
    Epoch 00013: early stopping
    7s - loss: 0.9982 - acc: 0.6912 - val_loss: 1.7624 - val_acc: 0.5012
    PREDICT
    6104/6104 [==============================] - 2s     
    PREDICT
    12209/12209 [==============================] - 3s     
    FIT PARAMS : 
    {'nb_filters_1': 32, 'nb_filters_2': 64, 'filter_size_2': 3, 'pool_size_1': 2, 'pool_size_2': 2, 'filter_size_1': 3}
    Train on 10988 samples, validate on 1221 samples
    Epoch 1/20
    8s - loss: 2.3825 - acc: 0.3191 - val_loss: 2.1252 - val_acc: 0.4062
    Epoch 2/20
    7s - loss: 2.0636 - acc: 0.3925 - val_loss: 1.9943 - val_acc: 0.4218
    Epoch 3/20
    7s - loss: 1.8867 - acc: 0.4331 - val_loss: 1.9266 - val_acc: 0.4218
    Epoch 4/20
    7s - loss: 1.7752 - acc: 0.4686 - val_loss: 1.7712 - val_acc: 0.4914
    Epoch 5/20
    7s - loss: 1.6883 - acc: 0.4902 - val_loss: 1.7553 - val_acc: 0.4947
    Epoch 6/20
    7s - loss: 1.6132 - acc: 0.5104 - val_loss: 1.9789 - val_acc: 0.4300
    Epoch 7/20
    Epoch 00006: early stopping
    7s - loss: 1.5474 - acc: 0.5273 - val_loss: 1.8140 - val_acc: 0.4676
    PREDICT
    6104/6104 [==============================] - 2s     
    PREDICT
    12192/12209 [============================>.] - ETA: 0sFIT PARAMS : 
    {'nb_filters_1': 32, 'nb_filters_2': 64, 'filter_size_2': 3, 'pool_size_1': 2, 'pool_size_2': 4, 'filter_size_1': 3}
    Train on 10987 samples, validate on 1221 samples
    Epoch 1/20
    7s - loss: 2.4183 - acc: 0.3113 - val_loss: 2.2232 - val_acc: 0.3702
    Epoch 2/20
    6s - loss: 2.1503 - acc: 0.3799 - val_loss: 2.0948 - val_acc: 0.3808
    Epoch 3/20
    6s - loss: 1.9923 - acc: 0.4060 - val_loss: 1.9761 - val_acc: 0.4136
    Epoch 4/20
    6s - loss: 1.8914 - acc: 0.4289 - val_loss: 1.9374 - val_acc: 0.4275
    Epoch 5/20
    6s - loss: 1.8184 - acc: 0.4492 - val_loss: 1.8625 - val_acc: 0.4554
    Epoch 6/20
    7s - loss: 1.7509 - acc: 0.4696 - val_loss: 1.8519 - val_acc: 0.4619
    Epoch 7/20
    7s - loss: 1.6890 - acc: 0.4889 - val_loss: 1.8982 - val_acc: 0.4390
    Epoch 8/20
    6s - loss: 1.6276 - acc: 0.5044 - val_loss: 1.7390 - val_acc: 0.4791
    Epoch 9/20
    6s - loss: 1.5751 - acc: 0.5213 - val_loss: 1.7773 - val_acc: 0.4848
    Epoch 10/20
    6s - loss: 1.5262 - acc: 0.5406 - val_loss: 1.7050 - val_acc: 0.5029
    Epoch 11/20
    6s - loss: 1.4877 - acc: 0.5498 - val_loss: 1.6865 - val_acc: 0.4996
    Epoch 12/20
    6s - loss: 1.4424 - acc: 0.5609 - val_loss: 1.6727 - val_acc: 0.5029
    Epoch 13/20
    7s - loss: 1.4104 - acc: 0.5730 - val_loss: 1.6905 - val_acc: 0.5012
    Epoch 14/20
    6s - loss: 1.3703 - acc: 0.5807 - val_loss: 1.6318 - val_acc: 0.4963
    Epoch 15/20
    6s - loss: 1.3348 - acc: 0.5907 - val_loss: 1.6521 - val_acc: 0.5119
    Epoch 16/20
    6s - loss: 1.2992 - acc: 0.6019 - val_loss: 1.6170 - val_acc: 0.5225
    Epoch 17/20
    6s - loss: 1.2650 - acc: 0.6090 - val_loss: 1.5731 - val_acc: 0.5381
    Epoch 18/20
    6s - loss: 1.2312 - acc: 0.6208 - val_loss: 1.6445 - val_acc: 0.5160
    Epoch 19/20
    Epoch 00018: early stopping
    6s - loss: 1.2013 - acc: 0.6258 - val_loss: 1.6035 - val_acc: 0.5274
    PREDICT
    6105/6105 [==============================] - 2s     
    PREDICT
    12208/12208 [==============================] - 3s     
    FIT PARAMS : 
    {'nb_filters_1': 32, 'nb_filters_2': 64, 'filter_size_2': 3, 'pool_size_1': 2, 'pool_size_2': 4, 'filter_size_1': 3}
    Train on 10988 samples, validate on 1221 samples
    Epoch 1/20
    8s - loss: 2.4115 - acc: 0.3146 - val_loss: 2.2316 - val_acc: 0.3694
    Epoch 2/20
    7s - loss: 2.1370 - acc: 0.3730 - val_loss: 2.1063 - val_acc: 0.3759
    Epoch 3/20
    6s - loss: 1.9974 - acc: 0.3980 - val_loss: 1.9577 - val_acc: 0.4185
    Epoch 4/20
    7s - loss: 1.8963 - acc: 0.4241 - val_loss: 1.8832 - val_acc: 0.4537
    Epoch 5/20
    6s - loss: 1.8097 - acc: 0.4498 - val_loss: 1.8101 - val_acc: 0.4627
    Epoch 6/20
    7s - loss: 1.7369 - acc: 0.4716 - val_loss: 1.7610 - val_acc: 0.4709
    Epoch 7/20
    7s - loss: 1.6757 - acc: 0.4906 - val_loss: 1.8102 - val_acc: 0.4701
    Epoch 8/20
    6s - loss: 1.6245 - acc: 0.5050 - val_loss: 1.6948 - val_acc: 0.5078
    Epoch 9/20
    6s - loss: 1.5703 - acc: 0.5264 - val_loss: 1.7494 - val_acc: 0.4889
    Epoch 10/20
    7s - loss: 1.5203 - acc: 0.5336 - val_loss: 1.6744 - val_acc: 0.5037
    Epoch 11/20
    7s - loss: 1.4733 - acc: 0.5495 - val_loss: 1.6337 - val_acc: 0.5192
    Epoch 12/20
    7s - loss: 1.4340 - acc: 0.5636 - val_loss: 1.6062 - val_acc: 0.5242
    Epoch 13/20
    7s - loss: 1.3934 - acc: 0.5727 - val_loss: 1.6127 - val_acc: 0.5045
    Epoch 14/20
    6s - loss: 1.3555 - acc: 0.5873 - val_loss: 1.6045 - val_acc: 0.5324
    Epoch 15/20
    7s - loss: 1.3177 - acc: 0.5938 - val_loss: 1.5945 - val_acc: 0.5389
    Epoch 16/20
    7s - loss: 1.2821 - acc: 0.6079 - val_loss: 1.5625 - val_acc: 0.5242
    Epoch 17/20
    6s - loss: 1.2424 - acc: 0.6157 - val_loss: 1.6007 - val_acc: 0.5102
    Epoch 18/20
    Epoch 00017: early stopping
    6s - loss: 1.2113 - acc: 0.6282 - val_loss: 1.5828 - val_acc: 0.5266
    PREDICT
    6104/6104 [==============================] - 2s     
    PREDICT
    12209/12209 [==============================] - 3s     
    FIT PARAMS : 
    {'nb_filters_1': 32, 'nb_filters_2': 64, 'filter_size_2': 3, 'pool_size_1': 2, 'pool_size_2': 4, 'filter_size_1': 3}
    Train on 10988 samples, validate on 1221 samples
    Epoch 1/20
    8s - loss: 2.4128 - acc: 0.3175 - val_loss: 2.2666 - val_acc: 0.3849
    Epoch 2/20
    6s - loss: 2.1834 - acc: 0.3699 - val_loss: 2.1389 - val_acc: 0.3776
    Epoch 3/20
    7s - loss: 2.0262 - acc: 0.3967 - val_loss: 2.0262 - val_acc: 0.4144
    Epoch 4/20
    6s - loss: 1.9043 - acc: 0.4284 - val_loss: 1.9186 - val_acc: 0.4283
    Epoch 5/20
    6s - loss: 1.8165 - acc: 0.4510 - val_loss: 1.8321 - val_acc: 0.4701
    Epoch 6/20
    7s - loss: 1.7516 - acc: 0.4732 - val_loss: 1.8413 - val_acc: 0.4644
    Epoch 7/20
    6s - loss: 1.6941 - acc: 0.4858 - val_loss: 1.7893 - val_acc: 0.4758
    Epoch 8/20
    6s - loss: 1.6424 - acc: 0.5061 - val_loss: 1.7574 - val_acc: 0.4930
    Epoch 9/20
    7s - loss: 1.5911 - acc: 0.5165 - val_loss: 1.7126 - val_acc: 0.5045
    Epoch 10/20
    6s - loss: 1.5495 - acc: 0.5294 - val_loss: 1.6530 - val_acc: 0.5160
    Epoch 11/20
    7s - loss: 1.5058 - acc: 0.5390 - val_loss: 1.6883 - val_acc: 0.5143
    Epoch 12/20
    Epoch 00011: early stopping
    7s - loss: 1.4651 - acc: 0.5519 - val_loss: 1.6834 - val_acc: 0.5217
    PREDICT
    6104/6104 [==============================] - 2s     
    PREDICT
    12192/12209 [============================>.] - ETA: 0sFIT PARAMS : 
    {'nb_filters_1': 32, 'nb_filters_2': 64, 'filter_size_2': 3, 'pool_size_1': 4, 'pool_size_2': 2, 'filter_size_1': 3}
    Train on 10987 samples, validate on 1221 samples
    Epoch 1/20
    5s - loss: 2.4144 - acc: 0.3148 - val_loss: 2.2058 - val_acc: 0.3702
    Epoch 2/20
    4s - loss: 2.1504 - acc: 0.3785 - val_loss: 2.1063 - val_acc: 0.3825
    Epoch 3/20
    4s - loss: 1.9988 - acc: 0.4082 - val_loss: 1.9720 - val_acc: 0.4169
    Epoch 4/20
    4s - loss: 1.8902 - acc: 0.4351 - val_loss: 1.8928 - val_acc: 0.4406
    Epoch 5/20
    4s - loss: 1.8045 - acc: 0.4576 - val_loss: 1.8405 - val_acc: 0.4505
    Epoch 6/20
    4s - loss: 1.7449 - acc: 0.4797 - val_loss: 1.8147 - val_acc: 0.4529
    Epoch 7/20
    4s - loss: 1.6922 - acc: 0.4898 - val_loss: 1.7849 - val_acc: 0.4816
    Epoch 8/20
    4s - loss: 1.6492 - acc: 0.5016 - val_loss: 1.7355 - val_acc: 0.4873
    Epoch 9/20
    4s - loss: 1.6112 - acc: 0.5146 - val_loss: 1.7039 - val_acc: 0.5029
    Epoch 10/20
    4s - loss: 1.5686 - acc: 0.5294 - val_loss: 1.6981 - val_acc: 0.4939
    Epoch 11/20
    4s - loss: 1.5345 - acc: 0.5398 - val_loss: 1.7000 - val_acc: 0.5029
    Epoch 12/20
    4s - loss: 1.4980 - acc: 0.5463 - val_loss: 1.6467 - val_acc: 0.5168
    Epoch 13/20
    4s - loss: 1.4613 - acc: 0.5586 - val_loss: 1.6204 - val_acc: 0.5266
    Epoch 14/20
    4s - loss: 1.4305 - acc: 0.5698 - val_loss: 1.6051 - val_acc: 0.5291
    Epoch 15/20
    4s - loss: 1.3941 - acc: 0.5779 - val_loss: 1.6177 - val_acc: 0.5266
    Epoch 16/20
    Epoch 00015: early stopping
    4s - loss: 1.3655 - acc: 0.5804 - val_loss: 1.6239 - val_acc: 0.5225
    PREDICT
    6105/6105 [==============================] - 2s     
    PREDICT
    12192/12208 [============================>.] - ETA: 0sFIT PARAMS : 
    {'nb_filters_1': 32, 'nb_filters_2': 64, 'filter_size_2': 3, 'pool_size_1': 4, 'pool_size_2': 2, 'filter_size_1': 3}
    Train on 10988 samples, validate on 1221 samples
    Epoch 1/20
    5s - loss: 2.4200 - acc: 0.3144 - val_loss: 2.2607 - val_acc: 0.3579
    Epoch 2/20
    4s - loss: 2.1637 - acc: 0.3762 - val_loss: 2.0772 - val_acc: 0.3964
    Epoch 3/20
    4s - loss: 2.0065 - acc: 0.4038 - val_loss: 1.9570 - val_acc: 0.4324
    Epoch 4/20
    4s - loss: 1.9028 - acc: 0.4280 - val_loss: 1.9412 - val_acc: 0.4439
    Epoch 5/20
    4s - loss: 1.8206 - acc: 0.4522 - val_loss: 1.8783 - val_acc: 0.4644
    Epoch 6/20
    4s - loss: 1.7632 - acc: 0.4731 - val_loss: 1.7831 - val_acc: 0.4685
    Epoch 7/20
    4s - loss: 1.7121 - acc: 0.4823 - val_loss: 1.7695 - val_acc: 0.4726
    Epoch 8/20
    4s - loss: 1.6720 - acc: 0.4889 - val_loss: 1.7148 - val_acc: 0.4988
    Epoch 9/20
    4s - loss: 1.6304 - acc: 0.5049 - val_loss: 1.7133 - val_acc: 0.4980
    Epoch 10/20
    4s - loss: 1.5972 - acc: 0.5112 - val_loss: 1.6913 - val_acc: 0.5078
    Epoch 11/20
    4s - loss: 1.5593 - acc: 0.5260 - val_loss: 1.6440 - val_acc: 0.5299
    Epoch 12/20
    4s - loss: 1.5257 - acc: 0.5336 - val_loss: 1.6320 - val_acc: 0.5274
    Epoch 13/20
    4s - loss: 1.4908 - acc: 0.5443 - val_loss: 1.6385 - val_acc: 0.5192
    Epoch 14/20
    Epoch 00013: early stopping
    4s - loss: 1.4582 - acc: 0.5501 - val_loss: 1.6585 - val_acc: 0.5192
    PREDICT
    6104/6104 [==============================] - 2s     
    PREDICT
    12192/12209 [============================>.] - ETA: 0sFIT PARAMS : 
    {'nb_filters_1': 32, 'nb_filters_2': 64, 'filter_size_2': 3, 'pool_size_1': 4, 'pool_size_2': 2, 'filter_size_1': 3}
    Train on 10988 samples, validate on 1221 samples
    Epoch 1/20
    5s - loss: 2.4490 - acc: 0.3011 - val_loss: 2.2990 - val_acc: 0.3776
    Epoch 2/20
    4s - loss: 2.1924 - acc: 0.3660 - val_loss: 2.0838 - val_acc: 0.4005
    Epoch 3/20
    4s - loss: 2.0239 - acc: 0.3996 - val_loss: 1.9536 - val_acc: 0.4292
    Epoch 4/20
    4s - loss: 1.9023 - acc: 0.4316 - val_loss: 1.8754 - val_acc: 0.4529
    Epoch 5/20
    4s - loss: 1.8177 - acc: 0.4509 - val_loss: 1.8011 - val_acc: 0.4750
    Epoch 6/20
    4s - loss: 1.7599 - acc: 0.4651 - val_loss: 1.8239 - val_acc: 0.4808
    Epoch 7/20
    Epoch 00006: early stopping
    4s - loss: 1.7122 - acc: 0.4816 - val_loss: 1.8188 - val_acc: 0.4758
    PREDICT
    6104/6104 [==============================] - 2s     
    PREDICT
    12192/12209 [============================>.] - ETA: 0sFIT PARAMS : 
    {'nb_filters_1': 32, 'nb_filters_2': 64, 'filter_size_2': 3, 'pool_size_1': 4, 'pool_size_2': 4, 'filter_size_1': 3}
    Train on 10987 samples, validate on 1221 samples
    Epoch 1/20
    5s - loss: 2.4618 - acc: 0.2894 - val_loss: 2.3906 - val_acc: 0.3464
    Epoch 2/20
    4s - loss: 2.2409 - acc: 0.3679 - val_loss: 2.1430 - val_acc: 0.3948
    Epoch 3/20
    4s - loss: 2.0706 - acc: 0.3945 - val_loss: 2.0143 - val_acc: 0.4120
    Epoch 4/20
    4s - loss: 1.9825 - acc: 0.4087 - val_loss: 1.9577 - val_acc: 0.4185
    Epoch 5/20
    4s - loss: 1.9200 - acc: 0.4236 - val_loss: 1.9095 - val_acc: 0.4275
    Epoch 6/20
    4s - loss: 1.8627 - acc: 0.4408 - val_loss: 1.9304 - val_acc: 0.4275
    Epoch 7/20
    4s - loss: 1.8116 - acc: 0.4541 - val_loss: 1.8421 - val_acc: 0.4627
    Epoch 8/20
    4s - loss: 1.7684 - acc: 0.4679 - val_loss: 1.7814 - val_acc: 0.4660
    Epoch 9/20
    4s - loss: 1.7252 - acc: 0.4790 - val_loss: 1.7550 - val_acc: 0.4742
    Epoch 10/20
    4s - loss: 1.6860 - acc: 0.4928 - val_loss: 1.7472 - val_acc: 0.4848
    Epoch 11/20
    4s - loss: 1.6555 - acc: 0.5025 - val_loss: 1.7444 - val_acc: 0.4758
    Epoch 12/20
    4s - loss: 1.6248 - acc: 0.5122 - val_loss: 1.6853 - val_acc: 0.4873
    Epoch 13/20
    4s - loss: 1.5960 - acc: 0.5176 - val_loss: 1.7081 - val_acc: 0.4840
    Epoch 14/20
    4s - loss: 1.5686 - acc: 0.5272 - val_loss: 1.6579 - val_acc: 0.5037
    Epoch 15/20
    4s - loss: 1.5434 - acc: 0.5355 - val_loss: 1.6592 - val_acc: 0.5020
    Epoch 16/20
    4s - loss: 1.5149 - acc: 0.5391 - val_loss: 1.6012 - val_acc: 0.5307
    Epoch 17/20
    4s - loss: 1.4937 - acc: 0.5505 - val_loss: 1.6366 - val_acc: 0.5176
    Epoch 18/20
    Epoch 00017: early stopping
    4s - loss: 1.4713 - acc: 0.5538 - val_loss: 1.6116 - val_acc: 0.5225
    PREDICT
    6105/6105 [==============================] - 2s     
    PREDICT
    12192/12208 [============================>.] - ETA: 0sFIT PARAMS : 
    {'nb_filters_1': 32, 'nb_filters_2': 64, 'filter_size_2': 3, 'pool_size_1': 4, 'pool_size_2': 4, 'filter_size_1': 3}
    Train on 10988 samples, validate on 1221 samples
    Epoch 1/20
    5s - loss: 2.4316 - acc: 0.3082 - val_loss: 2.2470 - val_acc: 0.3669
    Epoch 2/20
    4s - loss: 2.2211 - acc: 0.3670 - val_loss: 2.1839 - val_acc: 0.3726
    Epoch 3/20
    4s - loss: 2.1397 - acc: 0.3740 - val_loss: 2.0962 - val_acc: 0.3890
    Epoch 4/20
    4s - loss: 2.0476 - acc: 0.3913 - val_loss: 1.9917 - val_acc: 0.4062
    Epoch 5/20
    4s - loss: 1.9666 - acc: 0.4124 - val_loss: 1.9258 - val_acc: 0.4169
    Epoch 6/20
    4s - loss: 1.8959 - acc: 0.4306 - val_loss: 1.8734 - val_acc: 0.4201
    Epoch 7/20
    4s - loss: 1.8374 - acc: 0.4452 - val_loss: 1.8214 - val_acc: 0.4529
    Epoch 8/20
    4s - loss: 1.7901 - acc: 0.4564 - val_loss: 1.8123 - val_acc: 0.4668
    Epoch 9/20
    4s - loss: 1.7505 - acc: 0.4680 - val_loss: 1.7657 - val_acc: 0.4709
    Epoch 10/20
    4s - loss: 1.7109 - acc: 0.4778 - val_loss: 1.7151 - val_acc: 0.4898
    Epoch 11/20
    4s - loss: 1.6796 - acc: 0.4870 - val_loss: 1.7049 - val_acc: 0.4906
    Epoch 12/20
    4s - loss: 1.6518 - acc: 0.4969 - val_loss: 1.6747 - val_acc: 0.5004
    Epoch 13/20
    4s - loss: 1.6223 - acc: 0.5046 - val_loss: 1.6478 - val_acc: 0.5094
    Epoch 14/20
    4s - loss: 1.5910 - acc: 0.5164 - val_loss: 1.6513 - val_acc: 0.5160
    Epoch 15/20
    4s - loss: 1.5685 - acc: 0.5168 - val_loss: 1.6098 - val_acc: 0.5201
    Epoch 16/20
    4s - loss: 1.5428 - acc: 0.5285 - val_loss: 1.5824 - val_acc: 0.5348
    Epoch 17/20
    4s - loss: 1.5172 - acc: 0.5374 - val_loss: 1.5895 - val_acc: 0.5266
    Epoch 18/20
    Epoch 00017: early stopping
    4s - loss: 1.4938 - acc: 0.5405 - val_loss: 1.6134 - val_acc: 0.5225
    PREDICT
    6104/6104 [==============================] - 2s     
    PREDICT
    12192/12209 [============================>.] - ETA: 0sFIT PARAMS : 
    {'nb_filters_1': 32, 'nb_filters_2': 64, 'filter_size_2': 3, 'pool_size_1': 4, 'pool_size_2': 4, 'filter_size_1': 3}
    Train on 10988 samples, validate on 1221 samples
    Epoch 1/20
    5s - loss: 2.4706 - acc: 0.2953 - val_loss: 2.3384 - val_acc: 0.3645
    Epoch 2/20
    4s - loss: 2.2620 - acc: 0.3591 - val_loss: 2.1974 - val_acc: 0.3898
    Epoch 3/20
    4s - loss: 2.1803 - acc: 0.3684 - val_loss: 2.1665 - val_acc: 0.3792
    Epoch 4/20
    4s - loss: 2.1076 - acc: 0.3778 - val_loss: 2.0438 - val_acc: 0.4144
    Epoch 5/20
    4s - loss: 2.0043 - acc: 0.4035 - val_loss: 1.9664 - val_acc: 0.4390
    Epoch 6/20
    4s - loss: 1.9144 - acc: 0.4275 - val_loss: 1.9113 - val_acc: 0.4455
    Epoch 7/20
    4s - loss: 1.8465 - acc: 0.4407 - val_loss: 1.8591 - val_acc: 0.4513
    Epoch 8/20
    4s - loss: 1.7982 - acc: 0.4542 - val_loss: 1.8211 - val_acc: 0.4717
    Epoch 9/20
    4s - loss: 1.7551 - acc: 0.4657 - val_loss: 1.7742 - val_acc: 0.4832
    Epoch 10/20
    4s - loss: 1.7209 - acc: 0.4739 - val_loss: 1.7633 - val_acc: 0.4840
    Epoch 11/20
    4s - loss: 1.6899 - acc: 0.4837 - val_loss: 1.7540 - val_acc: 0.4947
    Epoch 12/20
    4s - loss: 1.6594 - acc: 0.4930 - val_loss: 1.7713 - val_acc: 0.4717
    Epoch 13/20
    4s - loss: 1.6319 - acc: 0.5004 - val_loss: 1.7237 - val_acc: 0.5004
    Epoch 14/20
    4s - loss: 1.6051 - acc: 0.5111 - val_loss: 1.6774 - val_acc: 0.5070
    Epoch 15/20
    4s - loss: 1.5778 - acc: 0.5156 - val_loss: 1.7036 - val_acc: 0.4922
    Epoch 16/20
    4s - loss: 1.5549 - acc: 0.5234 - val_loss: 1.6741 - val_acc: 0.5094
    Epoch 17/20
    4s - loss: 1.5311 - acc: 0.5297 - val_loss: 1.6615 - val_acc: 0.5094
    Epoch 18/20
    4s - loss: 1.5083 - acc: 0.5375 - val_loss: 1.6564 - val_acc: 0.5053
    Epoch 19/20
    4s - loss: 1.4854 - acc: 0.5450 - val_loss: 1.6677 - val_acc: 0.5086
    Epoch 20/20
    4s - loss: 1.4615 - acc: 0.5477 - val_loss: 1.5993 - val_acc: 0.5364
    PREDICT
    6104/6104 [==============================] - 2s     
    PREDICT
    12192/12209 [============================>.] - ETA: 0sFIT PARAMS : 
    {'nb_filters_1': 64, 'nb_filters_2': 32, 'filter_size_2': 3, 'pool_size_1': 2, 'pool_size_2': 2, 'filter_size_1': 3}
    Train on 10987 samples, validate on 1221 samples
    Epoch 1/20
    10s - loss: 2.4178 - acc: 0.3127 - val_loss: 2.2487 - val_acc: 0.3628
    Epoch 2/20
    8s - loss: 2.1301 - acc: 0.3787 - val_loss: 2.0484 - val_acc: 0.4021
    Epoch 3/20
    8s - loss: 1.9228 - acc: 0.4232 - val_loss: 1.8998 - val_acc: 0.4676
    Epoch 4/20
    8s - loss: 1.7860 - acc: 0.4649 - val_loss: 1.8835 - val_acc: 0.4447
    Epoch 5/20
    8s - loss: 1.6866 - acc: 0.4921 - val_loss: 1.7883 - val_acc: 0.4775
    Epoch 6/20
    8s - loss: 1.6125 - acc: 0.5119 - val_loss: 1.7599 - val_acc: 0.4783
    Epoch 7/20
    8s - loss: 1.5439 - acc: 0.5333 - val_loss: 1.7076 - val_acc: 0.4955
    Epoch 8/20
    8s - loss: 1.4824 - acc: 0.5480 - val_loss: 1.8691 - val_acc: 0.4627
    Epoch 9/20
    Epoch 00008: early stopping
    8s - loss: 1.4215 - acc: 0.5641 - val_loss: 1.7335 - val_acc: 0.5004
    PREDICT
    6105/6105 [==============================] - 2s     
    PREDICT
    12208/12208 [==============================] - 3s     
    FIT PARAMS : 
    {'nb_filters_1': 64, 'nb_filters_2': 32, 'filter_size_2': 3, 'pool_size_1': 2, 'pool_size_2': 2, 'filter_size_1': 3}
    Train on 10988 samples, validate on 1221 samples
    Epoch 1/20
    9s - loss: 2.4395 - acc: 0.3014 - val_loss: 2.2355 - val_acc: 0.3563
    Epoch 2/20
    8s - loss: 2.1770 - acc: 0.3680 - val_loss: 2.1201 - val_acc: 0.3890
    Epoch 3/20
    8s - loss: 2.0013 - acc: 0.4034 - val_loss: 1.9337 - val_acc: 0.4373
    Epoch 4/20
    8s - loss: 1.8632 - acc: 0.4395 - val_loss: 1.8706 - val_acc: 0.4676
    Epoch 5/20
    8s - loss: 1.7637 - acc: 0.4637 - val_loss: 1.8293 - val_acc: 0.4636
    Epoch 6/20
    8s - loss: 1.6884 - acc: 0.4889 - val_loss: 1.7690 - val_acc: 0.4824
    Epoch 7/20
    8s - loss: 1.6193 - acc: 0.5080 - val_loss: 1.8190 - val_acc: 0.4627
    Epoch 8/20
    8s - loss: 1.5554 - acc: 0.5218 - val_loss: 1.7065 - val_acc: 0.4955
    Epoch 9/20
    8s - loss: 1.4961 - acc: 0.5433 - val_loss: 1.6830 - val_acc: 0.5004
    Epoch 10/20
    8s - loss: 1.4376 - acc: 0.5593 - val_loss: 1.6607 - val_acc: 0.5192
    Epoch 11/20
    8s - loss: 1.3811 - acc: 0.5772 - val_loss: 1.6767 - val_acc: 0.5045
    Epoch 12/20
    Epoch 00011: early stopping
    8s - loss: 1.3270 - acc: 0.5907 - val_loss: 1.6775 - val_acc: 0.5061
    PREDICT
    6104/6104 [==============================] - 2s     
    PREDICT
    12209/12209 [==============================] - 4s     
    FIT PARAMS : 
    {'nb_filters_1': 64, 'nb_filters_2': 32, 'filter_size_2': 3, 'pool_size_1': 2, 'pool_size_2': 2, 'filter_size_1': 3}
    Train on 10988 samples, validate on 1221 samples
    Epoch 1/20
    9s - loss: 2.4080 - acc: 0.3172 - val_loss: 2.2229 - val_acc: 0.3898
    Epoch 2/20
    8s - loss: 2.1573 - acc: 0.3737 - val_loss: 2.3794 - val_acc: 0.2907
    Epoch 3/20
    8s - loss: 1.9790 - acc: 0.4094 - val_loss: 2.0160 - val_acc: 0.4070
    Epoch 4/20
    8s - loss: 1.8623 - acc: 0.4434 - val_loss: 1.8716 - val_acc: 0.4488
    Epoch 5/20
    8s - loss: 1.7693 - acc: 0.4672 - val_loss: 1.8326 - val_acc: 0.4595
    Epoch 6/20
    8s - loss: 1.6893 - acc: 0.4890 - val_loss: 1.8199 - val_acc: 0.4693
    Epoch 7/20
    8s - loss: 1.6178 - acc: 0.5121 - val_loss: 1.7572 - val_acc: 0.4930
    Epoch 8/20
    8s - loss: 1.5535 - acc: 0.5235 - val_loss: 1.7545 - val_acc: 0.4840
    Epoch 9/20
    8s - loss: 1.4985 - acc: 0.5372 - val_loss: 1.7557 - val_acc: 0.4955
    Epoch 10/20
    Epoch 00009: early stopping
    8s - loss: 1.4392 - acc: 0.5581 - val_loss: 1.7765 - val_acc: 0.4881
    PREDICT
    6080/6104 [============================>.] - ETA: 0sPREDICT
    12192/12209 [============================>.] - ETA: 0sFIT PARAMS : 
    {'nb_filters_1': 64, 'nb_filters_2': 32, 'filter_size_2': 3, 'pool_size_1': 2, 'pool_size_2': 4, 'filter_size_1': 3}
    Train on 10987 samples, validate on 1221 samples
    Epoch 1/20
    9s - loss: 2.4725 - acc: 0.2863 - val_loss: 2.4179 - val_acc: 0.2965
    Epoch 2/20
    8s - loss: 2.2399 - acc: 0.3652 - val_loss: 2.1429 - val_acc: 0.3735
    Epoch 3/20
    8s - loss: 2.0751 - acc: 0.3909 - val_loss: 2.0444 - val_acc: 0.4005
    Epoch 4/20
    8s - loss: 1.9562 - acc: 0.4098 - val_loss: 1.9346 - val_acc: 0.4390
    Epoch 5/20
    8s - loss: 1.8733 - acc: 0.4294 - val_loss: 1.9071 - val_acc: 0.4398
    Epoch 6/20
    8s - loss: 1.8056 - acc: 0.4529 - val_loss: 1.8581 - val_acc: 0.4505
    Epoch 7/20
    8s - loss: 1.7472 - acc: 0.4724 - val_loss: 1.8279 - val_acc: 0.4603
    Epoch 8/20
    8s - loss: 1.6956 - acc: 0.4860 - val_loss: 1.8004 - val_acc: 0.4595
    Epoch 9/20
    8s - loss: 1.6505 - acc: 0.5049 - val_loss: 1.7360 - val_acc: 0.4848
    Epoch 10/20
    8s - loss: 1.6084 - acc: 0.5132 - val_loss: 1.7473 - val_acc: 0.4824
    Epoch 11/20
    8s - loss: 1.5636 - acc: 0.5277 - val_loss: 1.6896 - val_acc: 0.4996
    Epoch 12/20
    8s - loss: 1.5285 - acc: 0.5369 - val_loss: 1.6711 - val_acc: 0.5094
    Epoch 13/20
    8s - loss: 1.4912 - acc: 0.5474 - val_loss: 1.7068 - val_acc: 0.4971
    Epoch 14/20
    Epoch 00013: early stopping
    8s - loss: 1.4610 - acc: 0.5548 - val_loss: 1.6899 - val_acc: 0.5119
    PREDICT
    6080/6105 [============================>.] - ETA: 0sPREDICT
    12208/12208 [==============================] - 3s     
    FIT PARAMS : 
    {'nb_filters_1': 64, 'nb_filters_2': 32, 'filter_size_2': 3, 'pool_size_1': 2, 'pool_size_2': 4, 'filter_size_1': 3}
    Train on 10988 samples, validate on 1221 samples
    Epoch 1/20
    9s - loss: 2.4260 - acc: 0.3102 - val_loss: 2.2210 - val_acc: 0.3686
    Epoch 2/20
    8s - loss: 2.1699 - acc: 0.3725 - val_loss: 2.1058 - val_acc: 0.3759
    Epoch 3/20
    8s - loss: 2.0452 - acc: 0.3952 - val_loss: 2.0236 - val_acc: 0.4005
    Epoch 4/20
    8s - loss: 1.9578 - acc: 0.4123 - val_loss: 1.9311 - val_acc: 0.4292
    Epoch 5/20
    8s - loss: 1.8897 - acc: 0.4296 - val_loss: 1.9260 - val_acc: 0.4292
    Epoch 6/20
    8s - loss: 1.8383 - acc: 0.4432 - val_loss: 1.8836 - val_acc: 0.4480
    Epoch 7/20
    8s - loss: 1.7912 - acc: 0.4539 - val_loss: 1.8466 - val_acc: 0.4521
    Epoch 8/20
    8s - loss: 1.7509 - acc: 0.4687 - val_loss: 1.7878 - val_acc: 0.4676
    Epoch 9/20
    8s - loss: 1.7084 - acc: 0.4800 - val_loss: 1.7755 - val_acc: 0.4750
    Epoch 10/20
    8s - loss: 1.6706 - acc: 0.4894 - val_loss: 1.8274 - val_acc: 0.4652
    Epoch 11/20
    8s - loss: 1.6370 - acc: 0.4990 - val_loss: 1.7042 - val_acc: 0.4980
    Epoch 12/20
    8s - loss: 1.5929 - acc: 0.5120 - val_loss: 1.6923 - val_acc: 0.4955
    Epoch 13/20
    8s - loss: 1.5595 - acc: 0.5266 - val_loss: 1.7310 - val_acc: 0.4930
    Epoch 14/20
    8s - loss: 1.5251 - acc: 0.5338 - val_loss: 1.6401 - val_acc: 0.5225
    Epoch 15/20
    8s - loss: 1.4900 - acc: 0.5454 - val_loss: 1.6659 - val_acc: 0.5078
    Epoch 16/20
    Epoch 00015: early stopping
    8s - loss: 1.4576 - acc: 0.5503 - val_loss: 1.6456 - val_acc: 0.5086
    PREDICT
    6104/6104 [==============================] - 2s     
    PREDICT
    12209/12209 [==============================] - 3s     
    FIT PARAMS : 
    {'nb_filters_1': 64, 'nb_filters_2': 32, 'filter_size_2': 3, 'pool_size_1': 2, 'pool_size_2': 4, 'filter_size_1': 3}
    Train on 10988 samples, validate on 1221 samples
    Epoch 1/20
    9s - loss: 2.4586 - acc: 0.2974 - val_loss: 2.2936 - val_acc: 0.3718
    Epoch 2/20
    8s - loss: 2.2157 - acc: 0.3643 - val_loss: 2.1813 - val_acc: 0.3833
    Epoch 3/20
    8s - loss: 2.1084 - acc: 0.3780 - val_loss: 2.0658 - val_acc: 0.4062
    Epoch 4/20
    8s - loss: 2.0013 - acc: 0.4001 - val_loss: 1.9638 - val_acc: 0.4242
    Epoch 5/20
    8s - loss: 1.9071 - acc: 0.4270 - val_loss: 1.9105 - val_acc: 0.4464
    Epoch 6/20
    8s - loss: 1.8353 - acc: 0.4408 - val_loss: 1.8542 - val_acc: 0.4529
    Epoch 7/20
    8s - loss: 1.7760 - acc: 0.4608 - val_loss: 1.8381 - val_acc: 0.4603
    Epoch 8/20
    8s - loss: 1.7269 - acc: 0.4778 - val_loss: 1.9039 - val_acc: 0.4242
    Epoch 9/20
    8s - loss: 1.6777 - acc: 0.4934 - val_loss: 1.7404 - val_acc: 0.5037
    Epoch 10/20
    8s - loss: 1.6338 - acc: 0.5009 - val_loss: 1.7497 - val_acc: 0.5061
    Epoch 11/20
    8s - loss: 1.5958 - acc: 0.5147 - val_loss: 1.7284 - val_acc: 0.4996
    Epoch 12/20
    8s - loss: 1.5569 - acc: 0.5223 - val_loss: 1.7187 - val_acc: 0.4709
    Epoch 13/20
    8s - loss: 1.5265 - acc: 0.5354 - val_loss: 1.6475 - val_acc: 0.5356
    Epoch 14/20
    8s - loss: 1.4908 - acc: 0.5441 - val_loss: 1.6432 - val_acc: 0.5168
    Epoch 15/20
    8s - loss: 1.4580 - acc: 0.5531 - val_loss: 1.6420 - val_acc: 0.5250
    Epoch 16/20
    8s - loss: 1.4293 - acc: 0.5647 - val_loss: 1.6229 - val_acc: 0.5266
    Epoch 17/20
    8s - loss: 1.3989 - acc: 0.5716 - val_loss: 1.6731 - val_acc: 0.5143
    Epoch 18/20
    Epoch 00017: early stopping
    8s - loss: 1.3693 - acc: 0.5817 - val_loss: 1.6406 - val_acc: 0.5135
    PREDICT
    6080/6104 [============================>.] - ETA: 0sPREDICT
    12209/12209 [==============================] - 3s     
    FIT PARAMS : 
    {'nb_filters_1': 64, 'nb_filters_2': 32, 'filter_size_2': 3, 'pool_size_1': 4, 'pool_size_2': 2, 'filter_size_1': 3}
    Train on 10987 samples, validate on 1221 samples
    Epoch 1/20
    7s - loss: 2.4556 - acc: 0.2978 - val_loss: 2.2806 - val_acc: 0.3612
    Epoch 2/20
    6s - loss: 2.2186 - acc: 0.3727 - val_loss: 2.1928 - val_acc: 0.3710
    Epoch 3/20
    6s - loss: 2.0791 - acc: 0.3893 - val_loss: 2.0015 - val_acc: 0.4046
    Epoch 4/20
    6s - loss: 1.9527 - acc: 0.4211 - val_loss: 1.9234 - val_acc: 0.4423
    Epoch 5/20
    6s - loss: 1.8587 - acc: 0.4436 - val_loss: 1.8340 - val_acc: 0.4545
    Epoch 6/20
    6s - loss: 1.7917 - acc: 0.4695 - val_loss: 1.8114 - val_acc: 0.4816
    Epoch 7/20
    6s - loss: 1.7380 - acc: 0.4816 - val_loss: 1.7852 - val_acc: 0.4660
    Epoch 8/20
    6s - loss: 1.6937 - acc: 0.4883 - val_loss: 1.7737 - val_acc: 0.4685
    Epoch 9/20
    6s - loss: 1.6527 - acc: 0.5041 - val_loss: 1.7282 - val_acc: 0.5078
    Epoch 10/20
    6s - loss: 1.6204 - acc: 0.5104 - val_loss: 1.6921 - val_acc: 0.5045
    Epoch 11/20
    6s - loss: 1.5920 - acc: 0.5229 - val_loss: 1.6736 - val_acc: 0.5160
    Epoch 12/20
    6s - loss: 1.5558 - acc: 0.5306 - val_loss: 1.6856 - val_acc: 0.5209
    Epoch 13/20
    6s - loss: 1.5301 - acc: 0.5399 - val_loss: 1.6536 - val_acc: 0.5070
    Epoch 14/20
    6s - loss: 1.5067 - acc: 0.5423 - val_loss: 1.6576 - val_acc: 0.5102
    Epoch 15/20
    6s - loss: 1.4764 - acc: 0.5499 - val_loss: 1.6392 - val_acc: 0.5258
    Epoch 16/20
    6s - loss: 1.4522 - acc: 0.5557 - val_loss: 1.6311 - val_acc: 0.5168
    Epoch 17/20
    6s - loss: 1.4266 - acc: 0.5619 - val_loss: 1.5957 - val_acc: 0.5315
    Epoch 18/20
    6s - loss: 1.4025 - acc: 0.5715 - val_loss: 1.6136 - val_acc: 0.5274
    Epoch 19/20
    Epoch 00018: early stopping
    6s - loss: 1.3792 - acc: 0.5822 - val_loss: 1.6408 - val_acc: 0.5348
    PREDICT
    6105/6105 [==============================] - 2s     
    PREDICT
    12192/12208 [============================>.] - ETA: 0sFIT PARAMS : 
    {'nb_filters_1': 64, 'nb_filters_2': 32, 'filter_size_2': 3, 'pool_size_1': 4, 'pool_size_2': 2, 'filter_size_1': 3}
    Train on 10988 samples, validate on 1221 samples
    Epoch 1/20
    7s - loss: 2.4458 - acc: 0.3068 - val_loss: 2.2495 - val_acc: 0.3636
    Epoch 2/20
    6s - loss: 2.2109 - acc: 0.3654 - val_loss: 2.1533 - val_acc: 0.3759
    Epoch 3/20
    6s - loss: 2.0915 - acc: 0.3791 - val_loss: 2.0411 - val_acc: 0.3972
    Epoch 4/20
    6s - loss: 1.9588 - acc: 0.4126 - val_loss: 1.9131 - val_acc: 0.4275
    Epoch 5/20
    6s - loss: 1.8577 - acc: 0.4393 - val_loss: 1.8424 - val_acc: 0.4693
    Epoch 6/20
    6s - loss: 1.7865 - acc: 0.4588 - val_loss: 1.8918 - val_acc: 0.4619
    Epoch 7/20
    6s - loss: 1.7305 - acc: 0.4809 - val_loss: 1.7566 - val_acc: 0.4963
    Epoch 8/20
    6s - loss: 1.6867 - acc: 0.4870 - val_loss: 1.7304 - val_acc: 0.4848
    Epoch 9/20
    6s - loss: 1.6491 - acc: 0.4986 - val_loss: 1.6943 - val_acc: 0.5127
    Epoch 10/20
    6s - loss: 1.6150 - acc: 0.5101 - val_loss: 1.8140 - val_acc: 0.4529
    Epoch 11/20
    6s - loss: 1.5789 - acc: 0.5176 - val_loss: 1.6873 - val_acc: 0.4881
    Epoch 12/20
    6s - loss: 1.5518 - acc: 0.5277 - val_loss: 1.6937 - val_acc: 0.4808
    Epoch 13/20
    6s - loss: 1.5258 - acc: 0.5324 - val_loss: 1.6230 - val_acc: 0.5061
    Epoch 14/20
    6s - loss: 1.4977 - acc: 0.5377 - val_loss: 1.6180 - val_acc: 0.5152
    Epoch 15/20
    6s - loss: 1.4715 - acc: 0.5478 - val_loss: 1.6060 - val_acc: 0.5364
    Epoch 16/20
    6s - loss: 1.4448 - acc: 0.5513 - val_loss: 1.5553 - val_acc: 0.5307
    Epoch 17/20
    6s - loss: 1.4198 - acc: 0.5652 - val_loss: 1.6601 - val_acc: 0.4848
    Epoch 18/20
    Epoch 00017: early stopping
    6s - loss: 1.3961 - acc: 0.5708 - val_loss: 1.5900 - val_acc: 0.5143
    PREDICT
    6104/6104 [==============================] - 2s     
    PREDICT
    12192/12209 [============================>.] - ETA: 0sFIT PARAMS : 
    {'nb_filters_1': 64, 'nb_filters_2': 32, 'filter_size_2': 3, 'pool_size_1': 4, 'pool_size_2': 2, 'filter_size_1': 3}
    Train on 10988 samples, validate on 1221 samples
    Epoch 1/20
    7s - loss: 2.4318 - acc: 0.3085 - val_loss: 2.2604 - val_acc: 0.3800
    Epoch 2/20
    6s - loss: 2.2044 - acc: 0.3686 - val_loss: 2.1132 - val_acc: 0.4029
    Epoch 3/20
    6s - loss: 2.0581 - acc: 0.3916 - val_loss: 1.9893 - val_acc: 0.4283
    Epoch 4/20
    6s - loss: 1.9412 - acc: 0.4166 - val_loss: 1.9016 - val_acc: 0.4488
    Epoch 5/20
    6s - loss: 1.8544 - acc: 0.4457 - val_loss: 1.8550 - val_acc: 0.4586
    Epoch 6/20
    6s - loss: 1.7936 - acc: 0.4611 - val_loss: 1.8219 - val_acc: 0.4767
    Epoch 7/20
    6s - loss: 1.7465 - acc: 0.4752 - val_loss: 1.7868 - val_acc: 0.5045
    Epoch 8/20
    6s - loss: 1.7022 - acc: 0.4878 - val_loss: 1.7584 - val_acc: 0.4922
    Epoch 9/20
    6s - loss: 1.6617 - acc: 0.4987 - val_loss: 1.7438 - val_acc: 0.5111
    Epoch 10/20
    6s - loss: 1.6296 - acc: 0.5086 - val_loss: 1.7533 - val_acc: 0.5135
    Epoch 11/20
    Epoch 00010: early stopping
    6s - loss: 1.5984 - acc: 0.5173 - val_loss: 1.7474 - val_acc: 0.4758
    PREDICT
    6104/6104 [==============================] - 2s     
    PREDICT
    12192/12209 [============================>.] - ETA: 0sFIT PARAMS : 
    {'nb_filters_1': 64, 'nb_filters_2': 32, 'filter_size_2': 3, 'pool_size_1': 4, 'pool_size_2': 4, 'filter_size_1': 3}
    Train on 10987 samples, validate on 1221 samples
    Epoch 1/20
    7s - loss: 2.4728 - acc: 0.2880 - val_loss: 2.4220 - val_acc: 0.2965
    Epoch 2/20
    6s - loss: 2.3415 - acc: 0.3312 - val_loss: 2.2089 - val_acc: 0.3686
    Epoch 3/20
    6s - loss: 2.1876 - acc: 0.3712 - val_loss: 2.1230 - val_acc: 0.3792
    Epoch 4/20
    6s - loss: 2.1016 - acc: 0.3842 - val_loss: 2.0447 - val_acc: 0.3997
    Epoch 5/20
    6s - loss: 2.0156 - acc: 0.4067 - val_loss: 1.9715 - val_acc: 0.4242
    Epoch 6/20
    6s - loss: 1.9377 - acc: 0.4250 - val_loss: 1.9137 - val_acc: 0.4226
    Epoch 7/20
    6s - loss: 1.8718 - acc: 0.4405 - val_loss: 1.8522 - val_acc: 0.4505
    Epoch 8/20
    6s - loss: 1.8226 - acc: 0.4518 - val_loss: 1.8394 - val_acc: 0.4627
    Epoch 9/20
    6s - loss: 1.7807 - acc: 0.4655 - val_loss: 1.8410 - val_acc: 0.4529
    Epoch 10/20
    6s - loss: 1.7438 - acc: 0.4779 - val_loss: 1.7833 - val_acc: 0.4709
    Epoch 11/20
    6s - loss: 1.7121 - acc: 0.4866 - val_loss: 1.7486 - val_acc: 0.4709
    Epoch 12/20
    6s - loss: 1.6796 - acc: 0.4914 - val_loss: 1.7205 - val_acc: 0.4857
    Epoch 13/20
    6s - loss: 1.6510 - acc: 0.5039 - val_loss: 1.7752 - val_acc: 0.4709
    Epoch 14/20
    6s - loss: 1.6249 - acc: 0.5129 - val_loss: 1.6662 - val_acc: 0.5029
    Epoch 15/20
    6s - loss: 1.5998 - acc: 0.5182 - val_loss: 1.7056 - val_acc: 0.4889
    Epoch 16/20
    6s - loss: 1.5726 - acc: 0.5265 - val_loss: 1.6472 - val_acc: 0.5160
    Epoch 17/20
    6s - loss: 1.5495 - acc: 0.5317 - val_loss: 1.6273 - val_acc: 0.5127
    Epoch 18/20
    6s - loss: 1.5279 - acc: 0.5391 - val_loss: 1.6262 - val_acc: 0.5111
    Epoch 19/20
    6s - loss: 1.5051 - acc: 0.5407 - val_loss: 1.6018 - val_acc: 0.5283
    Epoch 20/20
    6s - loss: 1.4860 - acc: 0.5532 - val_loss: 1.5841 - val_acc: 0.5364
    PREDICT
    6105/6105 [==============================] - 2s     
    PREDICT
    12192/12208 [============================>.] - ETA: 0sFIT PARAMS : 
    {'nb_filters_1': 64, 'nb_filters_2': 32, 'filter_size_2': 3, 'pool_size_1': 4, 'pool_size_2': 4, 'filter_size_1': 3}
    Train on 10988 samples, validate on 1221 samples
    Epoch 1/20
    7s - loss: 2.4777 - acc: 0.2864 - val_loss: 2.4634 - val_acc: 0.2957
    Epoch 2/20
    6s - loss: 2.3704 - acc: 0.3157 - val_loss: 2.2450 - val_acc: 0.3579
    Epoch 3/20
    6s - loss: 2.1918 - acc: 0.3660 - val_loss: 2.1040 - val_acc: 0.3800
    Epoch 4/20
    6s - loss: 2.0893 - acc: 0.3849 - val_loss: 2.0414 - val_acc: 0.3956
    Epoch 5/20
    6s - loss: 2.0191 - acc: 0.3963 - val_loss: 1.9701 - val_acc: 0.4054
    Epoch 6/20
    6s - loss: 1.9623 - acc: 0.4080 - val_loss: 1.9334 - val_acc: 0.4169
    Epoch 7/20
    6s - loss: 1.9099 - acc: 0.4222 - val_loss: 1.8902 - val_acc: 0.4333
    Epoch 8/20
    6s - loss: 1.8653 - acc: 0.4341 - val_loss: 1.8438 - val_acc: 0.4545
    Epoch 9/20
    6s - loss: 1.8243 - acc: 0.4433 - val_loss: 1.8084 - val_acc: 0.4644
    Epoch 10/20
    6s - loss: 1.7842 - acc: 0.4591 - val_loss: 1.8266 - val_acc: 0.4595
    Epoch 11/20
    6s - loss: 1.7501 - acc: 0.4671 - val_loss: 1.7811 - val_acc: 0.4734
    Epoch 12/20
    6s - loss: 1.7174 - acc: 0.4751 - val_loss: 1.7513 - val_acc: 0.4775
    Epoch 13/20
    6s - loss: 1.6885 - acc: 0.4829 - val_loss: 1.7091 - val_acc: 0.4914
    Epoch 14/20
    6s - loss: 1.6620 - acc: 0.4994 - val_loss: 1.6859 - val_acc: 0.4848
    Epoch 15/20
    6s - loss: 1.6362 - acc: 0.5022 - val_loss: 1.7079 - val_acc: 0.4775
    Epoch 16/20
    6s - loss: 1.6142 - acc: 0.5066 - val_loss: 1.6636 - val_acc: 0.5053
    Epoch 17/20
    6s - loss: 1.5893 - acc: 0.5161 - val_loss: 1.7389 - val_acc: 0.4799
    Epoch 18/20
    6s - loss: 1.5657 - acc: 0.5223 - val_loss: 1.6487 - val_acc: 0.5012
    Epoch 19/20
    6s - loss: 1.5455 - acc: 0.5288 - val_loss: 1.6347 - val_acc: 0.5094
    Epoch 20/20
    6s - loss: 1.5253 - acc: 0.5316 - val_loss: 1.6090 - val_acc: 0.5119
    PREDICT
    6104/6104 [==============================] - 2s     
    PREDICT
    12192/12209 [============================>.] - ETA: 0sFIT PARAMS : 
    {'nb_filters_1': 64, 'nb_filters_2': 32, 'filter_size_2': 3, 'pool_size_1': 4, 'pool_size_2': 4, 'filter_size_1': 3}
    Train on 10988 samples, validate on 1221 samples
    Epoch 1/20
    7s - loss: 2.4741 - acc: 0.2868 - val_loss: 2.3589 - val_acc: 0.3432
    Epoch 2/20
    6s - loss: 2.2836 - acc: 0.3590 - val_loss: 2.2187 - val_acc: 0.3857
    Epoch 3/20
    6s - loss: 2.2008 - acc: 0.3686 - val_loss: 2.1795 - val_acc: 0.3948
    Epoch 4/20
    6s - loss: 2.1520 - acc: 0.3707 - val_loss: 2.1176 - val_acc: 0.3939
    Epoch 5/20
    6s - loss: 2.0948 - acc: 0.3836 - val_loss: 2.1405 - val_acc: 0.3923
    Epoch 6/20
    6s - loss: 2.0208 - acc: 0.4003 - val_loss: 1.9911 - val_acc: 0.4144
    Epoch 7/20
    6s - loss: 1.9458 - acc: 0.4223 - val_loss: 1.9427 - val_acc: 0.4242
    Epoch 8/20
    6s - loss: 1.8825 - acc: 0.4354 - val_loss: 1.8891 - val_acc: 0.4390
    Epoch 9/20
    6s - loss: 1.8333 - acc: 0.4482 - val_loss: 1.8325 - val_acc: 0.4603
    Epoch 10/20
    6s - loss: 1.7918 - acc: 0.4614 - val_loss: 1.8007 - val_acc: 0.4644
    Epoch 11/20
    6s - loss: 1.7572 - acc: 0.4680 - val_loss: 1.8018 - val_acc: 0.4578
    Epoch 12/20
    Epoch 00011: early stopping
    6s - loss: 1.7253 - acc: 0.4759 - val_loss: 1.8129 - val_acc: 0.4529
    PREDICT
    6104/6104 [==============================] - 2s     
    PREDICT
    12192/12209 [============================>.] - ETA: 0sFIT PARAMS : 
    {'nb_filters_1': 64, 'nb_filters_2': 64, 'filter_size_2': 3, 'pool_size_1': 2, 'pool_size_2': 2, 'filter_size_1': 3}
    Train on 10987 samples, validate on 1221 samples
    Epoch 1/20
    12s - loss: 2.3836 - acc: 0.3238 - val_loss: 2.2381 - val_acc: 0.3726
    Epoch 2/20
    11s - loss: 2.0790 - acc: 0.3901 - val_loss: 2.0650 - val_acc: 0.3898
    Epoch 3/20
    11s - loss: 1.8614 - acc: 0.4447 - val_loss: 1.9757 - val_acc: 0.4333
    Epoch 4/20
    11s - loss: 1.7272 - acc: 0.4777 - val_loss: 1.8729 - val_acc: 0.4619
    Epoch 5/20
    11s - loss: 1.6270 - acc: 0.5111 - val_loss: 1.7664 - val_acc: 0.4922
    Epoch 6/20
    11s - loss: 1.5478 - acc: 0.5304 - val_loss: 1.7682 - val_acc: 0.4791
    Epoch 7/20
    Epoch 00006: early stopping
    11s - loss: 1.4629 - acc: 0.5538 - val_loss: 1.8421 - val_acc: 0.4750
    PREDICT
    6105/6105 [==============================] - 3s     
    PREDICT
    12208/12208 [==============================] - 4s     
    FIT PARAMS : 
    {'nb_filters_1': 64, 'nb_filters_2': 64, 'filter_size_2': 3, 'pool_size_1': 2, 'pool_size_2': 2, 'filter_size_1': 3}
    Train on 10988 samples, validate on 1221 samples
    Epoch 1/20
    12s - loss: 2.3900 - acc: 0.3214 - val_loss: 2.1634 - val_acc: 0.3645
    Epoch 2/20
    11s - loss: 2.0707 - acc: 0.3897 - val_loss: 2.0075 - val_acc: 0.3890
    Epoch 3/20
    11s - loss: 1.8718 - acc: 0.4366 - val_loss: 1.8617 - val_acc: 0.4545
    Epoch 4/20
    11s - loss: 1.7424 - acc: 0.4695 - val_loss: 1.8154 - val_acc: 0.4644
    Epoch 5/20
    11s - loss: 1.6438 - acc: 0.5016 - val_loss: 1.7249 - val_acc: 0.4980
    Epoch 6/20
    11s - loss: 1.5568 - acc: 0.5257 - val_loss: 1.6978 - val_acc: 0.5111
    Epoch 7/20
    11s - loss: 1.4746 - acc: 0.5444 - val_loss: 1.7421 - val_acc: 0.4840
    Epoch 8/20
    11s - loss: 1.4030 - acc: 0.5664 - val_loss: 1.6712 - val_acc: 0.5135
    Epoch 9/20
    11s - loss: 1.3248 - acc: 0.5877 - val_loss: 1.6992 - val_acc: 0.4930
    Epoch 10/20
    Epoch 00009: early stopping
    11s - loss: 1.2463 - acc: 0.6183 - val_loss: 1.7372 - val_acc: 0.4930
    PREDICT
    6104/6104 [==============================] - 3s     
    PREDICT
    12209/12209 [==============================] - 4s     
    FIT PARAMS : 
    {'nb_filters_1': 64, 'nb_filters_2': 64, 'filter_size_2': 3, 'pool_size_1': 2, 'pool_size_2': 2, 'filter_size_1': 3}
    Train on 10988 samples, validate on 1221 samples
    Epoch 1/20
    12s - loss: 2.3784 - acc: 0.3292 - val_loss: 2.2057 - val_acc: 0.3808
    Epoch 2/20
    11s - loss: 2.0784 - acc: 0.3893 - val_loss: 2.1113 - val_acc: 0.3882
    Epoch 3/20
    11s - loss: 1.8601 - acc: 0.4433 - val_loss: 1.8889 - val_acc: 0.4398
    Epoch 4/20
    11s - loss: 1.7281 - acc: 0.4772 - val_loss: 1.7929 - val_acc: 0.4816
    Epoch 5/20
    11s - loss: 1.6326 - acc: 0.5041 - val_loss: 1.8037 - val_acc: 0.4562
    Epoch 6/20
    11s - loss: 1.5528 - acc: 0.5278 - val_loss: 1.7037 - val_acc: 0.4971
    Epoch 7/20
    11s - loss: 1.4730 - acc: 0.5492 - val_loss: 1.6994 - val_acc: 0.5053
    Epoch 8/20
    11s - loss: 1.3975 - acc: 0.5718 - val_loss: 1.8878 - val_acc: 0.4333
    Epoch 9/20
    Epoch 00008: early stopping
    11s - loss: 1.3169 - acc: 0.5958 - val_loss: 1.7771 - val_acc: 0.4914
    PREDICT
    6104/6104 [==============================] - 3s     
    PREDICT
    12192/12209 [============================>.] - ETA: 0sFIT PARAMS : 
    {'nb_filters_1': 64, 'nb_filters_2': 64, 'filter_size_2': 3, 'pool_size_1': 2, 'pool_size_2': 4, 'filter_size_1': 3}
    Train on 10987 samples, validate on 1221 samples
    Epoch 1/20
    12s - loss: 2.4448 - acc: 0.3019 - val_loss: 2.2761 - val_acc: 0.3620
    Epoch 2/20
    10s - loss: 2.2084 - acc: 0.3705 - val_loss: 2.1462 - val_acc: 0.3735
    Epoch 3/20
    10s - loss: 2.0940 - acc: 0.3818 - val_loss: 2.0355 - val_acc: 0.3997
    Epoch 4/20
    10s - loss: 1.9461 - acc: 0.4193 - val_loss: 1.9986 - val_acc: 0.4431
    Epoch 5/20
    10s - loss: 1.8182 - acc: 0.4573 - val_loss: 1.8220 - val_acc: 0.4668
    Epoch 6/20
    10s - loss: 1.7263 - acc: 0.4836 - val_loss: 1.7882 - val_acc: 0.4832
    Epoch 7/20
    10s - loss: 1.6576 - acc: 0.5046 - val_loss: 1.7613 - val_acc: 0.4742
    Epoch 8/20
    10s - loss: 1.6020 - acc: 0.5183 - val_loss: 1.7248 - val_acc: 0.4914
    Epoch 9/20
    10s - loss: 1.5523 - acc: 0.5310 - val_loss: 1.6906 - val_acc: 0.4971
    Epoch 10/20
    10s - loss: 1.5071 - acc: 0.5449 - val_loss: 1.6735 - val_acc: 0.5029
    Epoch 11/20
    10s - loss: 1.4600 - acc: 0.5591 - val_loss: 1.6409 - val_acc: 0.5192
    Epoch 12/20
    10s - loss: 1.4175 - acc: 0.5715 - val_loss: 1.6475 - val_acc: 0.5127
    Epoch 13/20
    10s - loss: 1.3810 - acc: 0.5766 - val_loss: 1.5955 - val_acc: 0.5233
    Epoch 14/20
    10s - loss: 1.3387 - acc: 0.5898 - val_loss: 1.6317 - val_acc: 0.5176
    Epoch 15/20
    Epoch 00014: early stopping
    10s - loss: 1.3020 - acc: 0.6011 - val_loss: 1.6030 - val_acc: 0.5364
    PREDICT
    6080/6105 [============================>.] - ETA: 0sPREDICT
    12192/12208 [============================>.] - ETA: 0sFIT PARAMS : 
    {'nb_filters_1': 64, 'nb_filters_2': 64, 'filter_size_2': 3, 'pool_size_1': 2, 'pool_size_2': 4, 'filter_size_1': 3}
    Train on 10988 samples, validate on 1221 samples
    Epoch 1/20
    12s - loss: 2.4215 - acc: 0.3087 - val_loss: 2.1908 - val_acc: 0.3686
    Epoch 2/20
    10s - loss: 2.1302 - acc: 0.3781 - val_loss: 2.0308 - val_acc: 0.3980
    Epoch 3/20
    10s - loss: 1.9892 - acc: 0.4039 - val_loss: 1.9638 - val_acc: 0.4251
    Epoch 4/20
    10s - loss: 1.8957 - acc: 0.4284 - val_loss: 1.8812 - val_acc: 0.4439
    Epoch 5/20
    10s - loss: 1.8206 - acc: 0.4464 - val_loss: 1.8392 - val_acc: 0.4595
    Epoch 6/20
    10s - loss: 1.7494 - acc: 0.4705 - val_loss: 1.7832 - val_acc: 0.4717
    Epoch 7/20
    10s - loss: 1.6889 - acc: 0.4877 - val_loss: 1.7739 - val_acc: 0.4840
    Epoch 8/20
    10s - loss: 1.6325 - acc: 0.5030 - val_loss: 1.7199 - val_acc: 0.4914
    Epoch 9/20
    10s - loss: 1.5800 - acc: 0.5158 - val_loss: 1.6747 - val_acc: 0.5201
    Epoch 10/20
    10s - loss: 1.5318 - acc: 0.5318 - val_loss: 1.6615 - val_acc: 0.5037
    Epoch 11/20
    10s - loss: 1.4814 - acc: 0.5461 - val_loss: 1.6609 - val_acc: 0.5061
    Epoch 12/20
    10s - loss: 1.4433 - acc: 0.5594 - val_loss: 1.6030 - val_acc: 0.5348
    Epoch 13/20
    10s - loss: 1.3964 - acc: 0.5720 - val_loss: 1.5872 - val_acc: 0.5258
    Epoch 14/20
    10s - loss: 1.3535 - acc: 0.5873 - val_loss: 1.5585 - val_acc: 0.5438
    Epoch 15/20
    10s - loss: 1.3182 - acc: 0.5945 - val_loss: 1.5834 - val_acc: 0.5291
    Epoch 16/20
    Epoch 00015: early stopping
    10s - loss: 1.2747 - acc: 0.6098 - val_loss: 1.5673 - val_acc: 0.5430
    PREDICT
    6080/6104 [============================>.] - ETA: 0sPREDICT
    12192/12209 [============================>.] - ETA: 0sFIT PARAMS : 
    {'nb_filters_1': 64, 'nb_filters_2': 64, 'filter_size_2': 3, 'pool_size_1': 2, 'pool_size_2': 4, 'filter_size_1': 3}
    Train on 10988 samples, validate on 1221 samples
    Epoch 1/20
    12s - loss: 2.4178 - acc: 0.3121 - val_loss: 2.2616 - val_acc: 0.3702
    Epoch 2/20
    10s - loss: 2.1408 - acc: 0.3798 - val_loss: 2.0439 - val_acc: 0.4177
    Epoch 3/20
    10s - loss: 1.9955 - acc: 0.4038 - val_loss: 1.9396 - val_acc: 0.4316
    Epoch 4/20
    10s - loss: 1.8880 - acc: 0.4319 - val_loss: 1.8821 - val_acc: 0.4423
    Epoch 5/20
    10s - loss: 1.7930 - acc: 0.4599 - val_loss: 1.8698 - val_acc: 0.4390
    Epoch 6/20
    10s - loss: 1.7188 - acc: 0.4848 - val_loss: 1.7744 - val_acc: 0.4717
    Epoch 7/20
    10s - loss: 1.6570 - acc: 0.4986 - val_loss: 1.7790 - val_acc: 0.4701
    Epoch 8/20
    10s - loss: 1.6021 - acc: 0.5143 - val_loss: 1.6791 - val_acc: 0.5094
    Epoch 9/20
    10s - loss: 1.5549 - acc: 0.5291 - val_loss: 1.6384 - val_acc: 0.5438
    Epoch 10/20
    10s - loss: 1.5065 - acc: 0.5400 - val_loss: 1.6307 - val_acc: 0.5364
    Epoch 11/20
    10s - loss: 1.4604 - acc: 0.5536 - val_loss: 1.6238 - val_acc: 0.5127
    Epoch 12/20
    10s - loss: 1.4176 - acc: 0.5673 - val_loss: 1.6434 - val_acc: 0.5250
    Epoch 13/20
    Epoch 00012: early stopping
    10s - loss: 1.3763 - acc: 0.5808 - val_loss: 1.6268 - val_acc: 0.5373
    PREDICT
    6080/6104 [============================>.] - ETA: 0sPREDICT
    12192/12209 [============================>.] - ETA: 0sFIT PARAMS : 
    {'nb_filters_1': 64, 'nb_filters_2': 64, 'filter_size_2': 3, 'pool_size_1': 4, 'pool_size_2': 2, 'filter_size_1': 3}
    Train on 10987 samples, validate on 1221 samples
    Epoch 1/20
    8s - loss: 2.4076 - acc: 0.3174 - val_loss: 2.3347 - val_acc: 0.3366
    Epoch 2/20
    6s - loss: 2.1751 - acc: 0.3763 - val_loss: 2.1407 - val_acc: 0.3735
    Epoch 3/20
    6s - loss: 2.0138 - acc: 0.4046 - val_loss: 1.9962 - val_acc: 0.4283
    Epoch 4/20
    6s - loss: 1.8820 - acc: 0.4348 - val_loss: 1.8792 - val_acc: 0.4513
    Epoch 5/20
    6s - loss: 1.7909 - acc: 0.4676 - val_loss: 1.8008 - val_acc: 0.4676
    Epoch 6/20
    6s - loss: 1.7305 - acc: 0.4816 - val_loss: 1.7960 - val_acc: 0.4857
    Epoch 7/20
    6s - loss: 1.6736 - acc: 0.4965 - val_loss: 1.7839 - val_acc: 0.4824
    Epoch 8/20
    6s - loss: 1.6254 - acc: 0.5085 - val_loss: 1.7113 - val_acc: 0.4922
    Epoch 9/20
    6s - loss: 1.5809 - acc: 0.5238 - val_loss: 1.6733 - val_acc: 0.5176
    Epoch 10/20
    6s - loss: 1.5381 - acc: 0.5361 - val_loss: 1.6495 - val_acc: 0.5242
    Epoch 11/20
    6s - loss: 1.4962 - acc: 0.5467 - val_loss: 1.7545 - val_acc: 0.4734
    Epoch 12/20
    6s - loss: 1.4550 - acc: 0.5543 - val_loss: 1.6191 - val_acc: 0.5315
    Epoch 13/20
    6s - loss: 1.4170 - acc: 0.5681 - val_loss: 1.6089 - val_acc: 0.5274
    Epoch 14/20
    6s - loss: 1.3757 - acc: 0.5807 - val_loss: 1.5820 - val_acc: 0.5405
    Epoch 15/20
    6s - loss: 1.3367 - acc: 0.5941 - val_loss: 1.5990 - val_acc: 0.5373
    Epoch 16/20
    6s - loss: 1.3026 - acc: 0.5992 - val_loss: 1.5316 - val_acc: 0.5397
    Epoch 17/20
    6s - loss: 1.2667 - acc: 0.6105 - val_loss: 1.5584 - val_acc: 0.5356
    Epoch 18/20
    Epoch 00017: early stopping
    6s - loss: 1.2352 - acc: 0.6215 - val_loss: 1.5364 - val_acc: 0.5487
    PREDICT
    6105/6105 [==============================] - 2s     
    PREDICT
    12208/12208 [==============================] - 3s     
    FIT PARAMS : 
    {'nb_filters_1': 64, 'nb_filters_2': 64, 'filter_size_2': 3, 'pool_size_1': 4, 'pool_size_2': 2, 'filter_size_1': 3}
    Train on 10988 samples, validate on 1221 samples
    Epoch 1/20
    8s - loss: 2.4435 - acc: 0.3024 - val_loss: 2.2893 - val_acc: 0.3530
    Epoch 2/20
    6s - loss: 2.1560 - acc: 0.3734 - val_loss: 2.0765 - val_acc: 0.3964
    Epoch 3/20
    6s - loss: 1.9961 - acc: 0.4040 - val_loss: 1.9524 - val_acc: 0.4267
    Epoch 4/20
    6s - loss: 1.8872 - acc: 0.4288 - val_loss: 1.9085 - val_acc: 0.4357
    Epoch 5/20
    6s - loss: 1.7976 - acc: 0.4574 - val_loss: 1.8633 - val_acc: 0.4406
    Epoch 6/20
    6s - loss: 1.7327 - acc: 0.4714 - val_loss: 1.7516 - val_acc: 0.4750
    Epoch 7/20
    6s - loss: 1.6815 - acc: 0.4870 - val_loss: 1.7016 - val_acc: 0.4914
    Epoch 8/20
    6s - loss: 1.6421 - acc: 0.5002 - val_loss: 1.6610 - val_acc: 0.5119
    Epoch 9/20
    6s - loss: 1.5978 - acc: 0.5133 - val_loss: 1.8103 - val_acc: 0.4709
    Epoch 10/20
    Epoch 00009: early stopping
    6s - loss: 1.5612 - acc: 0.5223 - val_loss: 1.6620 - val_acc: 0.5070
    PREDICT
    6104/6104 [==============================] - 2s     
    PREDICT
    12209/12209 [==============================] - 3s     
    FIT PARAMS : 
    {'nb_filters_1': 64, 'nb_filters_2': 64, 'filter_size_2': 3, 'pool_size_1': 4, 'pool_size_2': 2, 'filter_size_1': 3}
    Train on 10988 samples, validate on 1221 samples
    Epoch 1/20
    8s - loss: 2.4370 - acc: 0.3066 - val_loss: 2.2966 - val_acc: 0.3825
    Epoch 2/20
    6s - loss: 2.1724 - acc: 0.3732 - val_loss: 2.1436 - val_acc: 0.3776
    Epoch 3/20
    6s - loss: 2.0131 - acc: 0.4013 - val_loss: 1.9690 - val_acc: 0.4275
    Epoch 4/20
    6s - loss: 1.8877 - acc: 0.4366 - val_loss: 1.9363 - val_acc: 0.4292
    Epoch 5/20
    6s - loss: 1.8007 - acc: 0.4619 - val_loss: 1.8556 - val_acc: 0.4545
    Epoch 6/20
    6s - loss: 1.7420 - acc: 0.4788 - val_loss: 1.7661 - val_acc: 0.4914
    Epoch 7/20
    6s - loss: 1.6914 - acc: 0.4921 - val_loss: 1.7909 - val_acc: 0.4939
    Epoch 8/20
    6s - loss: 1.6445 - acc: 0.5049 - val_loss: 1.7097 - val_acc: 0.5102
    Epoch 9/20
    6s - loss: 1.6008 - acc: 0.5149 - val_loss: 1.6574 - val_acc: 0.5094
    Epoch 10/20
    6s - loss: 1.5601 - acc: 0.5244 - val_loss: 1.6763 - val_acc: 0.5160
    Epoch 11/20
    6s - loss: 1.5220 - acc: 0.5353 - val_loss: 1.6480 - val_acc: 0.5160
    Epoch 12/20
    6s - loss: 1.4838 - acc: 0.5488 - val_loss: 1.6009 - val_acc: 0.5340
    Epoch 13/20
    6s - loss: 1.4465 - acc: 0.5575 - val_loss: 1.5885 - val_acc: 0.5373
    Epoch 14/20
    6s - loss: 1.4118 - acc: 0.5685 - val_loss: 1.5757 - val_acc: 0.5356
    Epoch 15/20
    6s - loss: 1.3745 - acc: 0.5763 - val_loss: 1.6167 - val_acc: 0.5266
    Epoch 16/20
    6s - loss: 1.3391 - acc: 0.5889 - val_loss: 1.5551 - val_acc: 0.5455
    Epoch 17/20
    6s - loss: 1.3103 - acc: 0.5967 - val_loss: 1.6490 - val_acc: 0.5152
    Epoch 18/20
    6s - loss: 1.2789 - acc: 0.6033 - val_loss: 1.5459 - val_acc: 0.5545
    Epoch 19/20
    6s - loss: 1.2461 - acc: 0.6144 - val_loss: 1.5595 - val_acc: 0.5438
    Epoch 20/20
    6s - loss: 1.2124 - acc: 0.6286 - val_loss: 1.5249 - val_acc: 0.5594
    PREDICT
    6104/6104 [==============================] - 2s     
    PREDICT
    12192/12209 [============================>.] - ETA: 0sFIT PARAMS : 
    {'nb_filters_1': 64, 'nb_filters_2': 64, 'filter_size_2': 3, 'pool_size_1': 4, 'pool_size_2': 4, 'filter_size_1': 3}
    Train on 10987 samples, validate on 1221 samples
    Epoch 1/20
    8s - loss: 2.4721 - acc: 0.2870 - val_loss: 2.3904 - val_acc: 0.3014
    Epoch 2/20
    6s - loss: 2.2845 - acc: 0.3554 - val_loss: 2.2126 - val_acc: 0.3636
    Epoch 3/20
    6s - loss: 2.1568 - acc: 0.3751 - val_loss: 2.1096 - val_acc: 0.3800
    Epoch 4/20
    6s - loss: 2.0595 - acc: 0.3924 - val_loss: 1.9978 - val_acc: 0.3989
    Epoch 5/20
    6s - loss: 1.9619 - acc: 0.4158 - val_loss: 1.9164 - val_acc: 0.4226
    Epoch 6/20
    6s - loss: 1.8788 - acc: 0.4354 - val_loss: 1.8710 - val_acc: 0.4545
    Epoch 7/20
    6s - loss: 1.8142 - acc: 0.4562 - val_loss: 1.8379 - val_acc: 0.4505
    Epoch 8/20
    6s - loss: 1.7611 - acc: 0.4733 - val_loss: 1.7946 - val_acc: 0.4652
    Epoch 9/20
    6s - loss: 1.7141 - acc: 0.4835 - val_loss: 1.7506 - val_acc: 0.4636
    Epoch 10/20
    6s - loss: 1.6747 - acc: 0.4952 - val_loss: 1.7980 - val_acc: 0.4636
    Epoch 11/20
    6s - loss: 1.6355 - acc: 0.5022 - val_loss: 1.6677 - val_acc: 0.4939
    Epoch 12/20
    6s - loss: 1.6016 - acc: 0.5168 - val_loss: 1.6788 - val_acc: 0.5045
    Epoch 13/20
    6s - loss: 1.5708 - acc: 0.5234 - val_loss: 1.6593 - val_acc: 0.5004
    Epoch 14/20
    6s - loss: 1.5395 - acc: 0.5329 - val_loss: 1.6840 - val_acc: 0.4947
    Epoch 15/20
    6s - loss: 1.5134 - acc: 0.5419 - val_loss: 1.6238 - val_acc: 0.5078
    Epoch 16/20
    6s - loss: 1.4845 - acc: 0.5527 - val_loss: 1.6029 - val_acc: 0.5127
    Epoch 17/20
    6s - loss: 1.4605 - acc: 0.5564 - val_loss: 1.5650 - val_acc: 0.5266
    Epoch 18/20
    6s - loss: 1.4351 - acc: 0.5645 - val_loss: 1.5614 - val_acc: 0.5266
    Epoch 19/20
    6s - loss: 1.4129 - acc: 0.5697 - val_loss: 1.5391 - val_acc: 0.5414
    Epoch 20/20
    6s - loss: 1.3900 - acc: 0.5790 - val_loss: 1.5553 - val_acc: 0.5324
    PREDICT
    6105/6105 [==============================] - 2s     
    PREDICT
    12192/12208 [============================>.] - ETA: 0sFIT PARAMS : 
    {'nb_filters_1': 64, 'nb_filters_2': 64, 'filter_size_2': 3, 'pool_size_1': 4, 'pool_size_2': 4, 'filter_size_1': 3}
    Train on 10988 samples, validate on 1221 samples
    Epoch 1/20
    8s - loss: 2.4759 - acc: 0.2879 - val_loss: 2.4337 - val_acc: 0.2957
    Epoch 2/20
    6s - loss: 2.2542 - acc: 0.3566 - val_loss: 2.1239 - val_acc: 0.3923
    Epoch 3/20
    6s - loss: 2.0828 - acc: 0.3871 - val_loss: 2.0261 - val_acc: 0.3997
    Epoch 4/20
    6s - loss: 2.0124 - acc: 0.3994 - val_loss: 1.9863 - val_acc: 0.4161
    Epoch 5/20
    6s - loss: 1.9557 - acc: 0.4099 - val_loss: 1.9297 - val_acc: 0.4365
    Epoch 6/20
    6s - loss: 1.9001 - acc: 0.4261 - val_loss: 1.8778 - val_acc: 0.4472
    Epoch 7/20
    6s - loss: 1.8523 - acc: 0.4386 - val_loss: 1.8221 - val_acc: 0.4513
    Epoch 8/20
    6s - loss: 1.8078 - acc: 0.4511 - val_loss: 1.8109 - val_acc: 0.4611
    Epoch 9/20
    6s - loss: 1.7635 - acc: 0.4625 - val_loss: 1.7907 - val_acc: 0.4726
    Epoch 10/20
    6s - loss: 1.7255 - acc: 0.4775 - val_loss: 1.7759 - val_acc: 0.4545
    Epoch 11/20
    6s - loss: 1.6878 - acc: 0.4869 - val_loss: 1.7266 - val_acc: 0.4767
    Epoch 12/20
    6s - loss: 1.6525 - acc: 0.4963 - val_loss: 1.6983 - val_acc: 0.4947
    Epoch 13/20
    6s - loss: 1.6198 - acc: 0.5082 - val_loss: 1.6548 - val_acc: 0.5029
    Epoch 14/20
    6s - loss: 1.5891 - acc: 0.5122 - val_loss: 1.6320 - val_acc: 0.5061
    Epoch 15/20
    6s - loss: 1.5578 - acc: 0.5243 - val_loss: 1.6485 - val_acc: 0.5070
    Epoch 16/20
    Epoch 00015: early stopping
    6s - loss: 1.5292 - acc: 0.5337 - val_loss: 1.6463 - val_acc: 0.5127
    PREDICT
    6104/6104 [==============================] - 2s     
    PREDICT
    12192/12209 [============================>.] - ETA: 0sFIT PARAMS : 
    {'nb_filters_1': 64, 'nb_filters_2': 64, 'filter_size_2': 3, 'pool_size_1': 4, 'pool_size_2': 4, 'filter_size_1': 3}
    Train on 10988 samples, validate on 1221 samples
    Epoch 1/20
    8s - loss: 2.4746 - acc: 0.2909 - val_loss: 2.3725 - val_acc: 0.3481
    Epoch 2/20
    6s - loss: 2.2679 - acc: 0.3612 - val_loss: 2.1827 - val_acc: 0.3849
    Epoch 3/20
    6s - loss: 2.1534 - acc: 0.3740 - val_loss: 2.0954 - val_acc: 0.3923
    Epoch 4/20
    6s - loss: 2.0413 - acc: 0.3936 - val_loss: 1.9870 - val_acc: 0.4234
    Epoch 5/20
    6s - loss: 1.9374 - acc: 0.4185 - val_loss: 1.8971 - val_acc: 0.4505
    Epoch 6/20
    6s - loss: 1.8588 - acc: 0.4391 - val_loss: 1.8489 - val_acc: 0.4496
    Epoch 7/20
    6s - loss: 1.7969 - acc: 0.4566 - val_loss: 1.8278 - val_acc: 0.4791
    Epoch 8/20
    6s - loss: 1.7452 - acc: 0.4723 - val_loss: 1.7779 - val_acc: 0.4734
    Epoch 9/20
    6s - loss: 1.7000 - acc: 0.4814 - val_loss: 1.7611 - val_acc: 0.4750
    Epoch 10/20
    6s - loss: 1.6609 - acc: 0.4973 - val_loss: 1.6981 - val_acc: 0.4963
    Epoch 11/20
    6s - loss: 1.6267 - acc: 0.5025 - val_loss: 1.7079 - val_acc: 0.5094
    Epoch 12/20
    6s - loss: 1.5920 - acc: 0.5153 - val_loss: 1.6423 - val_acc: 0.5192
    Epoch 13/20
    6s - loss: 1.5634 - acc: 0.5246 - val_loss: 1.6187 - val_acc: 0.5152
    Epoch 14/20
    6s - loss: 1.5304 - acc: 0.5323 - val_loss: 1.6102 - val_acc: 0.5258
    Epoch 15/20
    6s - loss: 1.5021 - acc: 0.5440 - val_loss: 1.6807 - val_acc: 0.4873
    Epoch 16/20
    6s - loss: 1.4738 - acc: 0.5515 - val_loss: 1.5651 - val_acc: 0.5332
    Epoch 17/20
    6s - loss: 1.4470 - acc: 0.5594 - val_loss: 1.5902 - val_acc: 0.5250
    Epoch 18/20
    6s - loss: 1.4192 - acc: 0.5635 - val_loss: 1.5298 - val_acc: 0.5414
    Epoch 19/20
    6s - loss: 1.3924 - acc: 0.5727 - val_loss: 1.5361 - val_acc: 0.5397
    Epoch 20/20
    Epoch 00019: early stopping
    6s - loss: 1.3676 - acc: 0.5820 - val_loss: 1.5874 - val_acc: 0.5381
    PREDICT
    6104/6104 [==============================] - 2s     
    PREDICT
    12192/12209 [============================>.] - ETA: 0sFIT PARAMS : 
    {'nb_filters_1': 32, 'nb_filters_2': 32, 'filter_size_2': 6, 'pool_size_1': 2, 'pool_size_2': 2, 'filter_size_1': 3}
    Train on 10987 samples, validate on 1221 samples
    Epoch 1/20
    10s - loss: 2.3886 - acc: 0.3230 - val_loss: 2.2404 - val_acc: 0.3407
    Epoch 2/20
    8s - loss: 2.1042 - acc: 0.3838 - val_loss: 2.2518 - val_acc: 0.3055
    Epoch 3/20
    8s - loss: 1.9154 - acc: 0.4240 - val_loss: 1.9437 - val_acc: 0.4488
    Epoch 4/20
    8s - loss: 1.7815 - acc: 0.4672 - val_loss: 1.7911 - val_acc: 0.4734
    Epoch 5/20
    8s - loss: 1.6801 - acc: 0.4927 - val_loss: 1.8282 - val_acc: 0.4783
    Epoch 6/20
    8s - loss: 1.6045 - acc: 0.5170 - val_loss: 1.7084 - val_acc: 0.5004
    Epoch 7/20
    8s - loss: 1.5315 - acc: 0.5366 - val_loss: 1.7121 - val_acc: 0.5070
    Epoch 8/20
    8s - loss: 1.4649 - acc: 0.5469 - val_loss: 1.7042 - val_acc: 0.5053
    Epoch 9/20
    8s - loss: 1.3970 - acc: 0.5700 - val_loss: 1.6797 - val_acc: 0.5020
    Epoch 10/20
    8s - loss: 1.3400 - acc: 0.5862 - val_loss: 1.6776 - val_acc: 0.5070
    Epoch 11/20
    8s - loss: 1.2786 - acc: 0.6041 - val_loss: 1.8543 - val_acc: 0.4758
    Epoch 12/20
    8s - loss: 1.2185 - acc: 0.6240 - val_loss: 1.6535 - val_acc: 0.5102
    Epoch 13/20
    8s - loss: 1.1597 - acc: 0.6402 - val_loss: 1.6597 - val_acc: 0.5201
    Epoch 14/20
    Epoch 00013: early stopping
    8s - loss: 1.0966 - acc: 0.6611 - val_loss: 1.6891 - val_acc: 0.5061
    PREDICT
    6105/6105 [==============================] - 3s     
    PREDICT
    12208/12208 [==============================] - 3s     
    FIT PARAMS : 
    {'nb_filters_1': 32, 'nb_filters_2': 32, 'filter_size_2': 6, 'pool_size_1': 2, 'pool_size_2': 2, 'filter_size_1': 3}
    Train on 10988 samples, validate on 1221 samples
    Epoch 1/20
    9s - loss: 2.3712 - acc: 0.3249 - val_loss: 2.2310 - val_acc: 0.3653
    Epoch 2/20
    8s - loss: 2.0520 - acc: 0.3922 - val_loss: 1.9599 - val_acc: 0.4161
    Epoch 3/20
    8s - loss: 1.8917 - acc: 0.4327 - val_loss: 1.8596 - val_acc: 0.4382
    Epoch 4/20
    8s - loss: 1.7823 - acc: 0.4613 - val_loss: 1.8595 - val_acc: 0.4619
    Epoch 5/20
    8s - loss: 1.7115 - acc: 0.4811 - val_loss: 1.8092 - val_acc: 0.4586
    Epoch 6/20
    8s - loss: 1.6469 - acc: 0.4969 - val_loss: 1.7165 - val_acc: 0.4955
    Epoch 7/20
    8s - loss: 1.5948 - acc: 0.5134 - val_loss: 1.6976 - val_acc: 0.5020
    Epoch 8/20
    8s - loss: 1.5428 - acc: 0.5259 - val_loss: 1.6533 - val_acc: 0.5045
    Epoch 9/20
    8s - loss: 1.4866 - acc: 0.5403 - val_loss: 1.6448 - val_acc: 0.5135
    Epoch 10/20
    8s - loss: 1.4333 - acc: 0.5562 - val_loss: 1.8402 - val_acc: 0.4455
    Epoch 11/20
    Epoch 00010: early stopping
    8s - loss: 1.3760 - acc: 0.5748 - val_loss: 1.6514 - val_acc: 0.5094
    PREDICT
    6104/6104 [==============================] - 3s     
    PREDICT
    12209/12209 [==============================] - 3s     
    FIT PARAMS : 
    {'nb_filters_1': 32, 'nb_filters_2': 32, 'filter_size_2': 6, 'pool_size_1': 2, 'pool_size_2': 2, 'filter_size_1': 3}
    Train on 10988 samples, validate on 1221 samples
    Epoch 1/20
    10s - loss: 2.3983 - acc: 0.3238 - val_loss: 2.2275 - val_acc: 0.3776
    Epoch 2/20
    8s - loss: 2.1136 - acc: 0.3814 - val_loss: 2.0569 - val_acc: 0.4103
    Epoch 3/20
    8s - loss: 1.9350 - acc: 0.4225 - val_loss: 1.9232 - val_acc: 0.4398
    Epoch 4/20
    8s - loss: 1.8214 - acc: 0.4544 - val_loss: 1.8884 - val_acc: 0.4308
    Epoch 5/20
    8s - loss: 1.7335 - acc: 0.4762 - val_loss: 1.8535 - val_acc: 0.4627
    Epoch 6/20
    8s - loss: 1.6566 - acc: 0.4952 - val_loss: 1.7827 - val_acc: 0.4848
    Epoch 7/20
    8s - loss: 1.5893 - acc: 0.5172 - val_loss: 1.7521 - val_acc: 0.4848
    Epoch 8/20
    8s - loss: 1.5230 - acc: 0.5321 - val_loss: 1.7176 - val_acc: 0.4955
    Epoch 9/20
    8s - loss: 1.4568 - acc: 0.5482 - val_loss: 1.6870 - val_acc: 0.5111
    Epoch 10/20
    8s - loss: 1.3960 - acc: 0.5681 - val_loss: 1.7244 - val_acc: 0.4840
    Epoch 11/20
    Epoch 00010: early stopping
    8s - loss: 1.3332 - acc: 0.5890 - val_loss: 1.7028 - val_acc: 0.5037
    PREDICT
    6080/6104 [============================>.] - ETA: 0sPREDICT
    12209/12209 [==============================] - 3s     
    FIT PARAMS : 
    {'nb_filters_1': 32, 'nb_filters_2': 32, 'filter_size_2': 6, 'pool_size_1': 2, 'pool_size_2': 4, 'filter_size_1': 3}
    Train on 10987 samples, validate on 1221 samples
    Epoch 1/20
    9s - loss: 2.4053 - acc: 0.3160 - val_loss: 2.2824 - val_acc: 0.3481
    Epoch 2/20
    8s - loss: 2.1400 - acc: 0.3815 - val_loss: 2.0613 - val_acc: 0.4120
    Epoch 3/20
    8s - loss: 1.9820 - acc: 0.4128 - val_loss: 1.9375 - val_acc: 0.4275
    Epoch 4/20
    8s - loss: 1.8775 - acc: 0.4385 - val_loss: 2.0341 - val_acc: 0.4046
    Epoch 5/20
    8s - loss: 1.8050 - acc: 0.4586 - val_loss: 1.8708 - val_acc: 0.4717
    Epoch 6/20
    8s - loss: 1.7459 - acc: 0.4785 - val_loss: 1.7773 - val_acc: 0.4840
    Epoch 7/20
    8s - loss: 1.6912 - acc: 0.4922 - val_loss: 1.8344 - val_acc: 0.4726
    Epoch 8/20
    8s - loss: 1.6467 - acc: 0.5031 - val_loss: 1.7158 - val_acc: 0.4971
    Epoch 9/20
    8s - loss: 1.6011 - acc: 0.5160 - val_loss: 1.7497 - val_acc: 0.4906
    Epoch 10/20
    Epoch 00009: early stopping
    8s - loss: 1.5563 - acc: 0.5295 - val_loss: 1.7558 - val_acc: 0.4799
    PREDICT
    6105/6105 [==============================] - 3s     
    PREDICT
    12192/12208 [============================>.] - ETA: 0sFIT PARAMS : 
    {'nb_filters_1': 32, 'nb_filters_2': 32, 'filter_size_2': 6, 'pool_size_1': 2, 'pool_size_2': 4, 'filter_size_1': 3}
    Train on 10988 samples, validate on 1221 samples
    Epoch 1/20
    9s - loss: 2.4096 - acc: 0.3208 - val_loss: 2.2567 - val_acc: 0.3669
    Epoch 2/20
    8s - loss: 2.1639 - acc: 0.3697 - val_loss: 2.0895 - val_acc: 0.4038
    Epoch 3/20
    8s - loss: 2.0019 - acc: 0.3981 - val_loss: 1.9605 - val_acc: 0.4128
    Epoch 4/20
    8s - loss: 1.8846 - acc: 0.4307 - val_loss: 1.8487 - val_acc: 0.4521
    Epoch 5/20
    8s - loss: 1.7963 - acc: 0.4514 - val_loss: 1.8128 - val_acc: 0.4775
    Epoch 6/20
    8s - loss: 1.7279 - acc: 0.4752 - val_loss: 1.7605 - val_acc: 0.4742
    Epoch 7/20
    8s - loss: 1.6671 - acc: 0.4893 - val_loss: 1.7194 - val_acc: 0.4816
    Epoch 8/20
    8s - loss: 1.6234 - acc: 0.5003 - val_loss: 1.6759 - val_acc: 0.5004
    Epoch 9/20
    8s - loss: 1.5749 - acc: 0.5158 - val_loss: 1.6859 - val_acc: 0.5020
    Epoch 10/20
    8s - loss: 1.5386 - acc: 0.5239 - val_loss: 1.6430 - val_acc: 0.5192
    Epoch 11/20
    8s - loss: 1.5004 - acc: 0.5391 - val_loss: 1.7017 - val_acc: 0.4857
    Epoch 12/20
    8s - loss: 1.4672 - acc: 0.5465 - val_loss: 1.6005 - val_acc: 0.5217
    Epoch 13/20
    8s - loss: 1.4260 - acc: 0.5578 - val_loss: 1.6160 - val_acc: 0.5111
    Epoch 14/20
    Epoch 00013: early stopping
    8s - loss: 1.3948 - acc: 0.5669 - val_loss: 1.7098 - val_acc: 0.4832
    PREDICT
    6080/6104 [============================>.] - ETA: 0sPREDICT
    12209/12209 [==============================] - 3s     
    FIT PARAMS : 
    {'nb_filters_1': 32, 'nb_filters_2': 32, 'filter_size_2': 6, 'pool_size_1': 2, 'pool_size_2': 4, 'filter_size_1': 3}
    Train on 10988 samples, validate on 1221 samples
    Epoch 1/20
    9s - loss: 2.4341 - acc: 0.3050 - val_loss: 2.2853 - val_acc: 0.3776
    Epoch 2/20
    8s - loss: 2.1727 - acc: 0.3694 - val_loss: 2.0955 - val_acc: 0.3923
    Epoch 3/20
    8s - loss: 2.0016 - acc: 0.4024 - val_loss: 2.0288 - val_acc: 0.4087
    Epoch 4/20
    8s - loss: 1.8960 - acc: 0.4311 - val_loss: 1.9004 - val_acc: 0.4373
    Epoch 5/20
    8s - loss: 1.8112 - acc: 0.4530 - val_loss: 1.8731 - val_acc: 0.4398
    Epoch 6/20
    8s - loss: 1.7385 - acc: 0.4769 - val_loss: 1.7817 - val_acc: 0.4832
    Epoch 7/20
    8s - loss: 1.6834 - acc: 0.4900 - val_loss: 1.7493 - val_acc: 0.4898
    Epoch 8/20
    8s - loss: 1.6354 - acc: 0.5039 - val_loss: 1.7687 - val_acc: 0.4824
    Epoch 9/20
    8s - loss: 1.5908 - acc: 0.5147 - val_loss: 1.6866 - val_acc: 0.5135
    Epoch 10/20
    8s - loss: 1.5516 - acc: 0.5254 - val_loss: 1.7002 - val_acc: 0.5094
    Epoch 11/20
    Epoch 00010: early stopping
    8s - loss: 1.5157 - acc: 0.5382 - val_loss: 1.7108 - val_acc: 0.4947
    PREDICT
    6104/6104 [==============================] - 3s     
    PREDICT
    12209/12209 [==============================] - 3s     
    FIT PARAMS : 
    {'nb_filters_1': 32, 'nb_filters_2': 32, 'filter_size_2': 6, 'pool_size_1': 4, 'pool_size_2': 2, 'filter_size_1': 3}
    Train on 10987 samples, validate on 1221 samples
    Epoch 1/20
    6s - loss: 2.4331 - acc: 0.3050 - val_loss: 2.4240 - val_acc: 0.3407
    Epoch 2/20
    4s - loss: 2.2111 - acc: 0.3685 - val_loss: 2.1953 - val_acc: 0.3735
    Epoch 3/20
    4s - loss: 2.0858 - acc: 0.3856 - val_loss: 2.0218 - val_acc: 0.3972
    Epoch 4/20
    4s - loss: 1.9693 - acc: 0.4139 - val_loss: 1.9525 - val_acc: 0.4292
    Epoch 5/20
    4s - loss: 1.8556 - acc: 0.4447 - val_loss: 1.8955 - val_acc: 0.4234
    Epoch 6/20
    4s - loss: 1.7797 - acc: 0.4647 - val_loss: 1.8282 - val_acc: 0.4783
    Epoch 7/20
    4s - loss: 1.7229 - acc: 0.4782 - val_loss: 1.7933 - val_acc: 0.4595
    Epoch 8/20
    4s - loss: 1.6695 - acc: 0.4925 - val_loss: 1.7448 - val_acc: 0.4783
    Epoch 9/20
    4s - loss: 1.6264 - acc: 0.5040 - val_loss: 1.7708 - val_acc: 0.4758
    Epoch 10/20
    4s - loss: 1.5838 - acc: 0.5196 - val_loss: 1.7314 - val_acc: 0.4767
    Epoch 11/20
    4s - loss: 1.5451 - acc: 0.5255 - val_loss: 1.7271 - val_acc: 0.4971
    Epoch 12/20
    4s - loss: 1.5074 - acc: 0.5416 - val_loss: 1.6512 - val_acc: 0.5127
    Epoch 13/20
    4s - loss: 1.4684 - acc: 0.5511 - val_loss: 1.6767 - val_acc: 0.4963
    Epoch 14/20
    4s - loss: 1.4355 - acc: 0.5575 - val_loss: 1.6415 - val_acc: 0.5004
    Epoch 15/20
    4s - loss: 1.4050 - acc: 0.5690 - val_loss: 1.6282 - val_acc: 0.5135
    Epoch 16/20
    4s - loss: 1.3714 - acc: 0.5791 - val_loss: 1.6462 - val_acc: 0.5176
    Epoch 17/20
    4s - loss: 1.3430 - acc: 0.5871 - val_loss: 1.6047 - val_acc: 0.5283
    Epoch 18/20
    4s - loss: 1.3099 - acc: 0.5978 - val_loss: 1.6183 - val_acc: 0.5061
    Epoch 19/20
    Epoch 00018: early stopping
    4s - loss: 1.2804 - acc: 0.6074 - val_loss: 1.6552 - val_acc: 0.5192
    PREDICT
    6105/6105 [==============================] - 2s     
    PREDICT
    12192/12208 [============================>.] - ETA: 0sFIT PARAMS : 
    {'nb_filters_1': 32, 'nb_filters_2': 32, 'filter_size_2': 6, 'pool_size_1': 4, 'pool_size_2': 2, 'filter_size_1': 3}
    Train on 10988 samples, validate on 1221 samples
    Epoch 1/20
    6s - loss: 2.4602 - acc: 0.2893 - val_loss: 2.3296 - val_acc: 0.3227
    Epoch 2/20
    4s - loss: 2.2190 - acc: 0.3614 - val_loss: 2.1381 - val_acc: 0.3767
    Epoch 3/20
    4s - loss: 2.0574 - acc: 0.3891 - val_loss: 2.0724 - val_acc: 0.3882
    Epoch 4/20
    4s - loss: 1.9484 - acc: 0.4107 - val_loss: 1.9869 - val_acc: 0.4111
    Epoch 5/20
    4s - loss: 1.8675 - acc: 0.4299 - val_loss: 1.8612 - val_acc: 0.4349
    Epoch 6/20
    4s - loss: 1.8044 - acc: 0.4519 - val_loss: 1.8521 - val_acc: 0.4529
    Epoch 7/20
    4s - loss: 1.7460 - acc: 0.4647 - val_loss: 1.7780 - val_acc: 0.4726
    Epoch 8/20
    4s - loss: 1.6982 - acc: 0.4830 - val_loss: 1.7755 - val_acc: 0.4775
    Epoch 9/20
    4s - loss: 1.6552 - acc: 0.4906 - val_loss: 1.7591 - val_acc: 0.4775
    Epoch 10/20
    4s - loss: 1.6181 - acc: 0.5092 - val_loss: 1.7207 - val_acc: 0.4971
    Epoch 11/20
    4s - loss: 1.5816 - acc: 0.5177 - val_loss: 1.7118 - val_acc: 0.4988
    Epoch 12/20
    4s - loss: 1.5448 - acc: 0.5257 - val_loss: 1.6920 - val_acc: 0.5004
    Epoch 13/20
    4s - loss: 1.5125 - acc: 0.5378 - val_loss: 1.6908 - val_acc: 0.4996
    Epoch 14/20
    4s - loss: 1.4784 - acc: 0.5421 - val_loss: 1.6277 - val_acc: 0.5217
    Epoch 15/20
    4s - loss: 1.4447 - acc: 0.5549 - val_loss: 1.7291 - val_acc: 0.4930
    Epoch 16/20
    4s - loss: 1.4131 - acc: 0.5668 - val_loss: 1.5952 - val_acc: 0.5102
    Epoch 17/20
    4s - loss: 1.3872 - acc: 0.5716 - val_loss: 1.6327 - val_acc: 0.5135
    Epoch 18/20
    4s - loss: 1.3565 - acc: 0.5791 - val_loss: 1.5916 - val_acc: 0.5283
    Epoch 19/20
    4s - loss: 1.3267 - acc: 0.5922 - val_loss: 1.5744 - val_acc: 0.5373
    Epoch 20/20
    4s - loss: 1.2992 - acc: 0.5968 - val_loss: 1.5855 - val_acc: 0.5266
    PREDICT
    6104/6104 [==============================] - 2s     
    PREDICT
    12192/12209 [============================>.] - ETA: 0sFIT PARAMS : 
    {'nb_filters_1': 32, 'nb_filters_2': 32, 'filter_size_2': 6, 'pool_size_1': 4, 'pool_size_2': 2, 'filter_size_1': 3}
    Train on 10988 samples, validate on 1221 samples
    Epoch 1/20
    6s - loss: 2.4586 - acc: 0.2918 - val_loss: 2.3608 - val_acc: 0.3497
    Epoch 2/20
    4s - loss: 2.2452 - acc: 0.3525 - val_loss: 2.1271 - val_acc: 0.3882
    Epoch 3/20
    4s - loss: 2.0742 - acc: 0.3896 - val_loss: 2.0581 - val_acc: 0.4275
    Epoch 4/20
    4s - loss: 1.9408 - acc: 0.4204 - val_loss: 1.8768 - val_acc: 0.4488
    Epoch 5/20
    4s - loss: 1.8488 - acc: 0.4431 - val_loss: 1.8837 - val_acc: 0.4562
    Epoch 6/20
    4s - loss: 1.7840 - acc: 0.4632 - val_loss: 1.7820 - val_acc: 0.4848
    Epoch 7/20
    4s - loss: 1.7285 - acc: 0.4774 - val_loss: 1.7723 - val_acc: 0.4832
    Epoch 8/20
    4s - loss: 1.6898 - acc: 0.4914 - val_loss: 1.7706 - val_acc: 0.4906
    Epoch 9/20
    4s - loss: 1.6470 - acc: 0.4970 - val_loss: 1.7345 - val_acc: 0.5045
    Epoch 10/20
    4s - loss: 1.6115 - acc: 0.5111 - val_loss: 1.7119 - val_acc: 0.4873
    Epoch 11/20
    4s - loss: 1.5749 - acc: 0.5218 - val_loss: 1.6584 - val_acc: 0.5258
    Epoch 12/20
    4s - loss: 1.5407 - acc: 0.5302 - val_loss: 1.7429 - val_acc: 0.4889
    Epoch 13/20
    4s - loss: 1.5081 - acc: 0.5391 - val_loss: 1.6355 - val_acc: 0.5217
    Epoch 14/20
    4s - loss: 1.4776 - acc: 0.5482 - val_loss: 1.6477 - val_acc: 0.5160
    Epoch 15/20
    4s - loss: 1.4426 - acc: 0.5594 - val_loss: 1.6182 - val_acc: 0.5242
    Epoch 16/20
    4s - loss: 1.4116 - acc: 0.5651 - val_loss: 1.6128 - val_acc: 0.5274
    Epoch 17/20
    4s - loss: 1.3810 - acc: 0.5737 - val_loss: 1.5744 - val_acc: 0.5373
    Epoch 18/20
    4s - loss: 1.3486 - acc: 0.5863 - val_loss: 1.5857 - val_acc: 0.5340
    Epoch 19/20
    Epoch 00018: early stopping
    4s - loss: 1.3186 - acc: 0.5933 - val_loss: 1.6272 - val_acc: 0.5274
    PREDICT
    6104/6104 [==============================] - 2s     
    PREDICT
    12192/12209 [============================>.] - ETA: 0sFIT PARAMS : 
    {'nb_filters_1': 32, 'nb_filters_2': 32, 'filter_size_2': 6, 'pool_size_1': 4, 'pool_size_2': 4, 'filter_size_1': 3}
    Train on 10987 samples, validate on 1221 samples
    Epoch 1/20
    6s - loss: 2.4373 - acc: 0.2978 - val_loss: 2.3243 - val_acc: 0.3292
    Epoch 2/20
    4s - loss: 2.2397 - acc: 0.3613 - val_loss: 2.1484 - val_acc: 0.3726
    Epoch 3/20
    4s - loss: 2.1141 - acc: 0.3837 - val_loss: 2.0586 - val_acc: 0.3964
    Epoch 4/20
    4s - loss: 2.0051 - acc: 0.4068 - val_loss: 1.9593 - val_acc: 0.4054
    Epoch 5/20
    4s - loss: 1.9155 - acc: 0.4327 - val_loss: 1.8866 - val_acc: 0.4300
    Epoch 6/20
    4s - loss: 1.8402 - acc: 0.4525 - val_loss: 1.8922 - val_acc: 0.4300
    Epoch 7/20
    4s - loss: 1.7766 - acc: 0.4703 - val_loss: 1.7932 - val_acc: 0.4775
    Epoch 8/20
    4s - loss: 1.7313 - acc: 0.4865 - val_loss: 1.7738 - val_acc: 0.4717
    Epoch 9/20
    4s - loss: 1.6866 - acc: 0.4947 - val_loss: 1.7726 - val_acc: 0.4939
    Epoch 10/20
    4s - loss: 1.6454 - acc: 0.5131 - val_loss: 1.7584 - val_acc: 0.4857
    Epoch 11/20
    4s - loss: 1.6093 - acc: 0.5211 - val_loss: 1.7326 - val_acc: 0.5004
    Epoch 12/20
    4s - loss: 1.5753 - acc: 0.5285 - val_loss: 1.7502 - val_acc: 0.4767
    Epoch 13/20
    4s - loss: 1.5477 - acc: 0.5376 - val_loss: 1.7042 - val_acc: 0.5020
    Epoch 14/20
    4s - loss: 1.5152 - acc: 0.5433 - val_loss: 1.6651 - val_acc: 0.5152
    Epoch 15/20
    4s - loss: 1.4836 - acc: 0.5529 - val_loss: 1.6728 - val_acc: 0.4816
    Epoch 16/20
    4s - loss: 1.4579 - acc: 0.5601 - val_loss: 1.6333 - val_acc: 0.5217
    Epoch 17/20
    4s - loss: 1.4323 - acc: 0.5682 - val_loss: 1.6104 - val_acc: 0.5143
    Epoch 18/20
    4s - loss: 1.4058 - acc: 0.5771 - val_loss: 1.5947 - val_acc: 0.5135
    Epoch 19/20
    4s - loss: 1.3805 - acc: 0.5816 - val_loss: 1.5973 - val_acc: 0.5225
    Epoch 20/20
    4s - loss: 1.3563 - acc: 0.5898 - val_loss: 1.5630 - val_acc: 0.5348
    PREDICT
    6105/6105 [==============================] - 2s     
    PREDICT
    12192/12208 [============================>.] - ETA: 0sFIT PARAMS : 
    {'nb_filters_1': 32, 'nb_filters_2': 32, 'filter_size_2': 6, 'pool_size_1': 4, 'pool_size_2': 4, 'filter_size_1': 3}
    Train on 10988 samples, validate on 1221 samples
    Epoch 1/20
    6s - loss: 2.4360 - acc: 0.3011 - val_loss: 2.2882 - val_acc: 0.3571
    Epoch 2/20
    4s - loss: 2.2370 - acc: 0.3634 - val_loss: 2.1536 - val_acc: 0.3735
    Epoch 3/20
    4s - loss: 2.1344 - acc: 0.3756 - val_loss: 2.0757 - val_acc: 0.3841
    Epoch 4/20
    4s - loss: 2.0427 - acc: 0.3928 - val_loss: 1.9922 - val_acc: 0.4046
    Epoch 5/20
    4s - loss: 1.9485 - acc: 0.4157 - val_loss: 1.9204 - val_acc: 0.4333
    Epoch 6/20
    4s - loss: 1.8729 - acc: 0.4351 - val_loss: 1.9028 - val_acc: 0.4275
    Epoch 7/20
    4s - loss: 1.8122 - acc: 0.4490 - val_loss: 1.8533 - val_acc: 0.4537
    Epoch 8/20
    4s - loss: 1.7645 - acc: 0.4636 - val_loss: 1.7828 - val_acc: 0.4676
    Epoch 9/20
    4s - loss: 1.7234 - acc: 0.4783 - val_loss: 1.7668 - val_acc: 0.4758
    Epoch 10/20
    4s - loss: 1.6837 - acc: 0.4882 - val_loss: 1.7513 - val_acc: 0.4865
    Epoch 11/20
    4s - loss: 1.6495 - acc: 0.4954 - val_loss: 1.7447 - val_acc: 0.4832
    Epoch 12/20
    4s - loss: 1.6172 - acc: 0.5062 - val_loss: 1.7611 - val_acc: 0.4701
    Epoch 13/20
    4s - loss: 1.5909 - acc: 0.5142 - val_loss: 1.6702 - val_acc: 0.5078
    Epoch 14/20
    4s - loss: 1.5617 - acc: 0.5207 - val_loss: 1.6865 - val_acc: 0.4980
    Epoch 15/20
    Epoch 00014: early stopping
    4s - loss: 1.5366 - acc: 0.5317 - val_loss: 1.7034 - val_acc: 0.5176
    PREDICT
    6104/6104 [==============================] - 2s     
    PREDICT
    12192/12209 [============================>.] - ETA: 0sFIT PARAMS : 
    {'nb_filters_1': 32, 'nb_filters_2': 32, 'filter_size_2': 6, 'pool_size_1': 4, 'pool_size_2': 4, 'filter_size_1': 3}
    Train on 10988 samples, validate on 1221 samples
    Epoch 1/20
    6s - loss: 2.4548 - acc: 0.3000 - val_loss: 2.3112 - val_acc: 0.3669
    Epoch 2/20
    4s - loss: 2.2540 - acc: 0.3620 - val_loss: 2.2082 - val_acc: 0.3833
    Epoch 3/20
    4s - loss: 2.1628 - acc: 0.3711 - val_loss: 2.1172 - val_acc: 0.3948
    Epoch 4/20
    4s - loss: 2.0802 - acc: 0.3876 - val_loss: 2.0469 - val_acc: 0.3980
    Epoch 5/20
    4s - loss: 2.0002 - acc: 0.4033 - val_loss: 1.9998 - val_acc: 0.4111
    Epoch 6/20
    4s - loss: 1.9272 - acc: 0.4228 - val_loss: 1.9463 - val_acc: 0.4300
    Epoch 7/20
    4s - loss: 1.8590 - acc: 0.4388 - val_loss: 1.8662 - val_acc: 0.4611
    Epoch 8/20
    4s - loss: 1.8038 - acc: 0.4547 - val_loss: 1.8763 - val_acc: 0.4480
    Epoch 9/20
    4s - loss: 1.7566 - acc: 0.4671 - val_loss: 1.8124 - val_acc: 0.4832
    Epoch 10/20
    4s - loss: 1.7133 - acc: 0.4805 - val_loss: 1.7826 - val_acc: 0.4717
    Epoch 11/20
    4s - loss: 1.6765 - acc: 0.4889 - val_loss: 1.7397 - val_acc: 0.4971
    Epoch 12/20
    4s - loss: 1.6405 - acc: 0.5023 - val_loss: 1.7307 - val_acc: 0.4996
    Epoch 13/20
    4s - loss: 1.6098 - acc: 0.5133 - val_loss: 1.7096 - val_acc: 0.5094
    Epoch 14/20
    4s - loss: 1.5795 - acc: 0.5193 - val_loss: 1.8282 - val_acc: 0.4676
    Epoch 15/20
    4s - loss: 1.5517 - acc: 0.5258 - val_loss: 1.6997 - val_acc: 0.5053
    Epoch 16/20
    4s - loss: 1.5280 - acc: 0.5300 - val_loss: 1.6419 - val_acc: 0.5225
    Epoch 17/20
    4s - loss: 1.4997 - acc: 0.5401 - val_loss: 1.7618 - val_acc: 0.4881
    Epoch 18/20
    4s - loss: 1.4777 - acc: 0.5463 - val_loss: 1.6410 - val_acc: 0.5258
    Epoch 19/20
    4s - loss: 1.4545 - acc: 0.5542 - val_loss: 1.6904 - val_acc: 0.5102
    Epoch 20/20
    4s - loss: 1.4293 - acc: 0.5645 - val_loss: 1.6271 - val_acc: 0.5242
    PREDICT
    6104/6104 [==============================] - 2s     
    PREDICT
    12192/12209 [============================>.] - ETA: 0sFIT PARAMS : 
    {'nb_filters_1': 32, 'nb_filters_2': 64, 'filter_size_2': 6, 'pool_size_1': 2, 'pool_size_2': 2, 'filter_size_1': 3}
    Train on 10987 samples, validate on 1221 samples
    Epoch 1/20
    13s - loss: 2.3813 - acc: 0.3249 - val_loss: 2.1680 - val_acc: 0.3948
    Epoch 2/20
    11s - loss: 2.0582 - acc: 0.3976 - val_loss: 2.0368 - val_acc: 0.4234
    Epoch 3/20
    11s - loss: 1.8624 - acc: 0.4426 - val_loss: 1.9122 - val_acc: 0.4406
    Epoch 4/20
    11s - loss: 1.7230 - acc: 0.4828 - val_loss: 1.8024 - val_acc: 0.4775
    Epoch 5/20
    11s - loss: 1.6130 - acc: 0.5144 - val_loss: 1.8595 - val_acc: 0.4578
    Epoch 6/20
    Epoch 00005: early stopping
    11s - loss: 1.5127 - acc: 0.5395 - val_loss: 1.8192 - val_acc: 0.4717
    PREDICT
    6105/6105 [==============================] - 3s     
    PREDICT
    12208/12208 [==============================] - 4s     
    FIT PARAMS : 
    {'nb_filters_1': 32, 'nb_filters_2': 64, 'filter_size_2': 6, 'pool_size_1': 2, 'pool_size_2': 2, 'filter_size_1': 3}
    Train on 10988 samples, validate on 1221 samples
    Epoch 1/20
    13s - loss: 2.3706 - acc: 0.3260 - val_loss: 2.2795 - val_acc: 0.2981
    Epoch 2/20
    11s - loss: 2.0505 - acc: 0.3921 - val_loss: 1.9904 - val_acc: 0.4021
    Epoch 3/20
    11s - loss: 1.8668 - acc: 0.4384 - val_loss: 1.8719 - val_acc: 0.4341
    Epoch 4/20
    11s - loss: 1.7335 - acc: 0.4745 - val_loss: 1.7721 - val_acc: 0.4914
    Epoch 5/20
    11s - loss: 1.6319 - acc: 0.4986 - val_loss: 1.7001 - val_acc: 0.4873
    Epoch 6/20
    11s - loss: 1.5392 - acc: 0.5245 - val_loss: 1.6862 - val_acc: 0.4840
    Epoch 7/20
    11s - loss: 1.4447 - acc: 0.5518 - val_loss: 1.6530 - val_acc: 0.5201
    Epoch 8/20
    11s - loss: 1.3587 - acc: 0.5822 - val_loss: 1.6096 - val_acc: 0.5364
    Epoch 9/20
    11s - loss: 1.2674 - acc: 0.6095 - val_loss: 1.6725 - val_acc: 0.5111
    Epoch 10/20
    Epoch 00009: early stopping
    11s - loss: 1.1789 - acc: 0.6329 - val_loss: 1.6287 - val_acc: 0.5160
    PREDICT
    6104/6104 [==============================] - 4s     
    PREDICT
    12209/12209 [==============================] - 4s     
    FIT PARAMS : 
    {'nb_filters_1': 32, 'nb_filters_2': 64, 'filter_size_2': 6, 'pool_size_1': 2, 'pool_size_2': 2, 'filter_size_1': 3}
    Train on 10988 samples, validate on 1221 samples
    Epoch 1/20
    13s - loss: 2.3950 - acc: 0.3204 - val_loss: 2.3195 - val_acc: 0.3358
    Epoch 2/20
    11s - loss: 2.0889 - acc: 0.3876 - val_loss: 2.0605 - val_acc: 0.4070
    Epoch 3/20
    11s - loss: 1.9017 - acc: 0.4319 - val_loss: 1.8662 - val_acc: 0.4505
    Epoch 4/20
    11s - loss: 1.7701 - acc: 0.4678 - val_loss: 1.7743 - val_acc: 0.4799
    Epoch 5/20
    11s - loss: 1.6680 - acc: 0.4966 - val_loss: 1.7809 - val_acc: 0.4922
    Epoch 6/20
    11s - loss: 1.5729 - acc: 0.5246 - val_loss: 1.7022 - val_acc: 0.4947
    Epoch 7/20
    11s - loss: 1.4857 - acc: 0.5468 - val_loss: 1.6428 - val_acc: 0.5258
    Epoch 8/20
    11s - loss: 1.4080 - acc: 0.5692 - val_loss: 1.6483 - val_acc: 0.5299
    Epoch 9/20
    Epoch 00008: early stopping
    11s - loss: 1.3331 - acc: 0.5873 - val_loss: 1.6829 - val_acc: 0.5168
    PREDICT
    6104/6104 [==============================] - 4s     
    PREDICT
    12192/12209 [============================>.] - ETA: 0sFIT PARAMS : 
    {'nb_filters_1': 32, 'nb_filters_2': 64, 'filter_size_2': 6, 'pool_size_1': 2, 'pool_size_2': 4, 'filter_size_1': 3}
    Train on 10987 samples, validate on 1221 samples
    Epoch 1/20
    12s - loss: 2.3890 - acc: 0.3225 - val_loss: 2.1571 - val_acc: 0.3743
    Epoch 2/20
    10s - loss: 2.0884 - acc: 0.3899 - val_loss: 1.9871 - val_acc: 0.4070
    Epoch 3/20
    10s - loss: 1.9250 - acc: 0.4261 - val_loss: 1.8838 - val_acc: 0.4431
    Epoch 4/20
    10s - loss: 1.8057 - acc: 0.4587 - val_loss: 1.8102 - val_acc: 0.4701
    Epoch 5/20
    10s - loss: 1.7149 - acc: 0.4860 - val_loss: 1.8042 - val_acc: 0.4799
    Epoch 6/20
    10s - loss: 1.6421 - acc: 0.5089 - val_loss: 1.8096 - val_acc: 0.4816
    Epoch 7/20
    10s - loss: 1.5789 - acc: 0.5268 - val_loss: 1.6640 - val_acc: 0.4939
    Epoch 8/20
    10s - loss: 1.5203 - acc: 0.5394 - val_loss: 1.6635 - val_acc: 0.5037
    Epoch 9/20
    10s - loss: 1.4671 - acc: 0.5535 - val_loss: 1.6004 - val_acc: 0.5348
    Epoch 10/20
    10s - loss: 1.4095 - acc: 0.5693 - val_loss: 1.6825 - val_acc: 0.4947
    Epoch 11/20
    10s - loss: 1.3588 - acc: 0.5844 - val_loss: 1.5668 - val_acc: 0.5455
    Epoch 12/20
    10s - loss: 1.3072 - acc: 0.5977 - val_loss: 1.5412 - val_acc: 0.5430
    Epoch 13/20
    10s - loss: 1.2580 - acc: 0.6127 - val_loss: 1.5946 - val_acc: 0.5168
    Epoch 14/20
    Epoch 00013: early stopping
    10s - loss: 1.2110 - acc: 0.6264 - val_loss: 1.6344 - val_acc: 0.5111
    PREDICT
    6080/6105 [============================>.] - ETA: 0sPREDICT
    12192/12208 [============================>.] - ETA: 0sFIT PARAMS : 
    {'nb_filters_1': 32, 'nb_filters_2': 64, 'filter_size_2': 6, 'pool_size_1': 2, 'pool_size_2': 4, 'filter_size_1': 3}
    Train on 10988 samples, validate on 1221 samples
    Epoch 1/20
    12s - loss: 2.4080 - acc: 0.3168 - val_loss: 2.2349 - val_acc: 0.3726
    Epoch 2/20
    10s - loss: 2.1709 - acc: 0.3688 - val_loss: 2.1512 - val_acc: 0.3653
    Epoch 3/20
    10s - loss: 2.0107 - acc: 0.3945 - val_loss: 1.9566 - val_acc: 0.4226
    Epoch 4/20
    10s - loss: 1.8809 - acc: 0.4265 - val_loss: 1.8275 - val_acc: 0.4513
    Epoch 5/20
    10s - loss: 1.7801 - acc: 0.4550 - val_loss: 1.7924 - val_acc: 0.4767
    Epoch 6/20
    10s - loss: 1.6922 - acc: 0.4843 - val_loss: 1.7337 - val_acc: 0.4767
    Epoch 7/20
    10s - loss: 1.6169 - acc: 0.5061 - val_loss: 1.7540 - val_acc: 0.4775
    Epoch 8/20
    10s - loss: 1.5520 - acc: 0.5259 - val_loss: 1.6728 - val_acc: 0.4906
    Epoch 9/20
    10s - loss: 1.4938 - acc: 0.5452 - val_loss: 1.6534 - val_acc: 0.5184
    Epoch 10/20
    10s - loss: 1.4406 - acc: 0.5583 - val_loss: 1.6153 - val_acc: 0.5143
    Epoch 11/20
    10s - loss: 1.3847 - acc: 0.5733 - val_loss: 1.6639 - val_acc: 0.5029
    Epoch 12/20
    10s - loss: 1.3340 - acc: 0.5903 - val_loss: 1.5955 - val_acc: 0.5176
    Epoch 13/20
    10s - loss: 1.2866 - acc: 0.6040 - val_loss: 1.6543 - val_acc: 0.5029
    Epoch 14/20
    10s - loss: 1.2406 - acc: 0.6158 - val_loss: 1.5565 - val_acc: 0.5348
    Epoch 15/20
    10s - loss: 1.1922 - acc: 0.6316 - val_loss: 1.5710 - val_acc: 0.5160
    Epoch 16/20
    Epoch 00015: early stopping
    10s - loss: 1.1481 - acc: 0.6436 - val_loss: 1.6332 - val_acc: 0.5168
    PREDICT
    6080/6104 [============================>.] - ETA: 0sPREDICT
    12192/12209 [============================>.] - ETA: 0sFIT PARAMS : 
    {'nb_filters_1': 32, 'nb_filters_2': 64, 'filter_size_2': 6, 'pool_size_1': 2, 'pool_size_2': 4, 'filter_size_1': 3}
    Train on 10988 samples, validate on 1221 samples
    Epoch 1/20
    13s - loss: 2.4084 - acc: 0.3135 - val_loss: 2.2841 - val_acc: 0.3776
    Epoch 2/20
    10s - loss: 2.1819 - acc: 0.3675 - val_loss: 2.0955 - val_acc: 0.4046
    Epoch 3/20
    10s - loss: 2.0042 - acc: 0.4012 - val_loss: 1.9733 - val_acc: 0.4373
    Epoch 4/20
    10s - loss: 1.8518 - acc: 0.4390 - val_loss: 1.9067 - val_acc: 0.4406
    Epoch 5/20
    10s - loss: 1.7474 - acc: 0.4727 - val_loss: 1.7776 - val_acc: 0.4668
    Epoch 6/20
    10s - loss: 1.6676 - acc: 0.4924 - val_loss: 1.7348 - val_acc: 0.4914
    Epoch 7/20
    10s - loss: 1.6027 - acc: 0.5137 - val_loss: 1.7762 - val_acc: 0.4808
    Epoch 8/20
    10s - loss: 1.5498 - acc: 0.5313 - val_loss: 1.6506 - val_acc: 0.5250
    Epoch 9/20
    10s - loss: 1.4985 - acc: 0.5438 - val_loss: 1.6559 - val_acc: 0.5233
    Epoch 10/20
    10s - loss: 1.4532 - acc: 0.5574 - val_loss: 1.6025 - val_acc: 0.5307
    Epoch 11/20
    10s - loss: 1.4038 - acc: 0.5666 - val_loss: 1.5985 - val_acc: 0.5356
    Epoch 12/20
    10s - loss: 1.3535 - acc: 0.5814 - val_loss: 1.6369 - val_acc: 0.5225
    Epoch 13/20
    10s - loss: 1.3115 - acc: 0.5980 - val_loss: 1.5817 - val_acc: 0.5446
    Epoch 14/20
    10s - loss: 1.2718 - acc: 0.6101 - val_loss: 1.5930 - val_acc: 0.5258
    Epoch 15/20
    10s - loss: 1.2242 - acc: 0.6249 - val_loss: 1.5512 - val_acc: 0.5340
    Epoch 16/20
    10s - loss: 1.1810 - acc: 0.6355 - val_loss: 1.5473 - val_acc: 0.5487
    Epoch 17/20
    10s - loss: 1.1384 - acc: 0.6515 - val_loss: 1.6168 - val_acc: 0.5299
    Epoch 18/20
    Epoch 00017: early stopping
    10s - loss: 1.0980 - acc: 0.6624 - val_loss: 1.5557 - val_acc: 0.5479
    PREDICT
    6080/6104 [============================>.] - ETA: 0sPREDICT
    12192/12209 [============================>.] - ETA: 0sFIT PARAMS : 
    {'nb_filters_1': 32, 'nb_filters_2': 64, 'filter_size_2': 6, 'pool_size_1': 4, 'pool_size_2': 2, 'filter_size_1': 3}
    Train on 10987 samples, validate on 1221 samples
    Epoch 1/20
    7s - loss: 2.4067 - acc: 0.3199 - val_loss: 2.2301 - val_acc: 0.3694
    Epoch 2/20
    5s - loss: 2.1781 - acc: 0.3752 - val_loss: 2.1688 - val_acc: 0.3792
    Epoch 3/20
    5s - loss: 2.0157 - acc: 0.3997 - val_loss: 1.9590 - val_acc: 0.4259
    Epoch 4/20
    5s - loss: 1.8629 - acc: 0.4430 - val_loss: 1.8475 - val_acc: 0.4513
    Epoch 5/20
    5s - loss: 1.7613 - acc: 0.4715 - val_loss: 1.7777 - val_acc: 0.4693
    Epoch 6/20
    5s - loss: 1.6917 - acc: 0.4901 - val_loss: 1.8733 - val_acc: 0.4423
    Epoch 7/20
    5s - loss: 1.6270 - acc: 0.5071 - val_loss: 1.7025 - val_acc: 0.4939
    Epoch 8/20
    5s - loss: 1.5781 - acc: 0.5228 - val_loss: 1.7340 - val_acc: 0.4963
    Epoch 9/20
    5s - loss: 1.5246 - acc: 0.5371 - val_loss: 1.6884 - val_acc: 0.5020
    Epoch 10/20
    5s - loss: 1.4794 - acc: 0.5525 - val_loss: 1.6191 - val_acc: 0.5127
    Epoch 11/20
    5s - loss: 1.4338 - acc: 0.5618 - val_loss: 1.6399 - val_acc: 0.5168
    Epoch 12/20
    5s - loss: 1.3855 - acc: 0.5771 - val_loss: 1.5786 - val_acc: 0.5348
    Epoch 13/20
    5s - loss: 1.3453 - acc: 0.5873 - val_loss: 1.5878 - val_acc: 0.5299
    Epoch 14/20
    5s - loss: 1.3016 - acc: 0.5958 - val_loss: 1.5529 - val_acc: 0.5422
    Epoch 15/20
    5s - loss: 1.2551 - acc: 0.6110 - val_loss: 1.5451 - val_acc: 0.5422
    Epoch 16/20
    5s - loss: 1.2115 - acc: 0.6325 - val_loss: 1.5705 - val_acc: 0.5225
    Epoch 17/20
    Epoch 00016: early stopping
    5s - loss: 1.1686 - acc: 0.6434 - val_loss: 1.6042 - val_acc: 0.5192
    PREDICT
    6105/6105 [==============================] - 3s     
    PREDICT
    12208/12208 [==============================] - 2s     
    FIT PARAMS : 
    {'nb_filters_1': 32, 'nb_filters_2': 64, 'filter_size_2': 6, 'pool_size_1': 4, 'pool_size_2': 2, 'filter_size_1': 3}
    Train on 10988 samples, validate on 1221 samples
    Epoch 1/20
    7s - loss: 2.4398 - acc: 0.2993 - val_loss: 2.2261 - val_acc: 0.3710
    Epoch 2/20
    5s - loss: 2.1306 - acc: 0.3763 - val_loss: 2.0403 - val_acc: 0.4169
    Epoch 3/20
    5s - loss: 1.9448 - acc: 0.4169 - val_loss: 1.8864 - val_acc: 0.4275
    Epoch 4/20
    5s - loss: 1.8199 - acc: 0.4497 - val_loss: 1.8038 - val_acc: 0.4488
    Epoch 5/20
    5s - loss: 1.7404 - acc: 0.4708 - val_loss: 1.7923 - val_acc: 0.4685
    Epoch 6/20
    5s - loss: 1.6761 - acc: 0.4934 - val_loss: 1.7249 - val_acc: 0.4742
    Epoch 7/20
    5s - loss: 1.6148 - acc: 0.5075 - val_loss: 1.7573 - val_acc: 0.4881
    Epoch 8/20
    Epoch 00007: early stopping
    5s - loss: 1.5639 - acc: 0.5228 - val_loss: 1.7517 - val_acc: 0.4742
    PREDICT
    6104/6104 [==============================] - 3s     
    PREDICT
    12192/12209 [============================>.] - ETA: 0sFIT PARAMS : 
    {'nb_filters_1': 32, 'nb_filters_2': 64, 'filter_size_2': 6, 'pool_size_1': 4, 'pool_size_2': 2, 'filter_size_1': 3}
    Train on 10988 samples, validate on 1221 samples
    Epoch 1/20
    7s - loss: 2.4033 - acc: 0.3186 - val_loss: 2.2067 - val_acc: 0.3800
    Epoch 2/20
    5s - loss: 2.1535 - acc: 0.3728 - val_loss: 2.0919 - val_acc: 0.3874
    Epoch 3/20
    5s - loss: 1.9612 - acc: 0.4175 - val_loss: 1.9627 - val_acc: 0.4201
    Epoch 4/20
    5s - loss: 1.8364 - acc: 0.4476 - val_loss: 1.8289 - val_acc: 0.4570
    Epoch 5/20
    5s - loss: 1.7446 - acc: 0.4723 - val_loss: 1.8057 - val_acc: 0.4939
    Epoch 6/20
    5s - loss: 1.6733 - acc: 0.4916 - val_loss: 1.8006 - val_acc: 0.4758
    Epoch 7/20
    5s - loss: 1.6089 - acc: 0.5078 - val_loss: 1.7481 - val_acc: 0.4767
    Epoch 8/20
    5s - loss: 1.5554 - acc: 0.5254 - val_loss: 1.6528 - val_acc: 0.5135
    Epoch 9/20
    5s - loss: 1.5008 - acc: 0.5402 - val_loss: 1.6191 - val_acc: 0.5315
    Epoch 10/20
    5s - loss: 1.4566 - acc: 0.5553 - val_loss: 1.6356 - val_acc: 0.5094
    Epoch 11/20
    5s - loss: 1.4093 - acc: 0.5663 - val_loss: 1.5834 - val_acc: 0.5332
    Epoch 12/20
    5s - loss: 1.3663 - acc: 0.5815 - val_loss: 1.5424 - val_acc: 0.5455
    Epoch 13/20
    5s - loss: 1.3229 - acc: 0.5956 - val_loss: 1.5561 - val_acc: 0.5430
    Epoch 14/20
    Epoch 00013: early stopping
    5s - loss: 1.2809 - acc: 0.6068 - val_loss: 1.5590 - val_acc: 0.5487
    PREDICT
    6104/6104 [==============================] - 3s     
    PREDICT
    12192/12209 [============================>.] - ETA: 0sFIT PARAMS : 
    {'nb_filters_1': 32, 'nb_filters_2': 64, 'filter_size_2': 6, 'pool_size_1': 4, 'pool_size_2': 4, 'filter_size_1': 3}
    Train on 10987 samples, validate on 1221 samples
    Epoch 1/20
    7s - loss: 2.4205 - acc: 0.3089 - val_loss: 2.2601 - val_acc: 0.3546
    Epoch 2/20
    5s - loss: 2.1918 - acc: 0.3714 - val_loss: 2.0964 - val_acc: 0.3800
    Epoch 3/20
    5s - loss: 2.0550 - acc: 0.3965 - val_loss: 2.0058 - val_acc: 0.3956
    Epoch 4/20
    5s - loss: 1.9472 - acc: 0.4229 - val_loss: 1.9431 - val_acc: 0.4406
    Epoch 5/20
    5s - loss: 1.8643 - acc: 0.4413 - val_loss: 1.8685 - val_acc: 0.4595
    Epoch 6/20
    5s - loss: 1.7947 - acc: 0.4655 - val_loss: 1.8284 - val_acc: 0.4505
    Epoch 7/20
    5s - loss: 1.7400 - acc: 0.4818 - val_loss: 1.7903 - val_acc: 0.4824
    Epoch 8/20
    5s - loss: 1.6881 - acc: 0.4947 - val_loss: 1.7439 - val_acc: 0.4955
    Epoch 9/20
    5s - loss: 1.6374 - acc: 0.5086 - val_loss: 1.7410 - val_acc: 0.4898
    Epoch 10/20
    5s - loss: 1.5969 - acc: 0.5207 - val_loss: 1.6905 - val_acc: 0.5004
    Epoch 11/20
    5s - loss: 1.5525 - acc: 0.5324 - val_loss: 1.6579 - val_acc: 0.5143
    Epoch 12/20
    5s - loss: 1.5156 - acc: 0.5407 - val_loss: 1.6254 - val_acc: 0.5209
    Epoch 13/20
    5s - loss: 1.4814 - acc: 0.5484 - val_loss: 1.6530 - val_acc: 0.5135
    Epoch 14/20
    Epoch 00013: early stopping
    5s - loss: 1.4425 - acc: 0.5594 - val_loss: 1.6947 - val_acc: 0.4816
    PREDICT
    6105/6105 [==============================] - 3s     
    PREDICT
    12192/12208 [============================>.] - ETA: 0sFIT PARAMS : 
    {'nb_filters_1': 32, 'nb_filters_2': 64, 'filter_size_2': 6, 'pool_size_1': 4, 'pool_size_2': 4, 'filter_size_1': 3}
    Train on 10988 samples, validate on 1221 samples
    Epoch 1/20
    7s - loss: 2.4288 - acc: 0.3112 - val_loss: 2.2458 - val_acc: 0.3595
    Epoch 2/20
    5s - loss: 2.2237 - acc: 0.3648 - val_loss: 2.1417 - val_acc: 0.3759
    Epoch 3/20
    5s - loss: 2.1067 - acc: 0.3793 - val_loss: 2.1108 - val_acc: 0.3923
    Epoch 4/20
    5s - loss: 1.9994 - acc: 0.4006 - val_loss: 1.9768 - val_acc: 0.4095
    Epoch 5/20
    5s - loss: 1.9175 - acc: 0.4227 - val_loss: 1.9236 - val_acc: 0.4283
    Epoch 6/20
    5s - loss: 1.8468 - acc: 0.4421 - val_loss: 1.9450 - val_acc: 0.4169
    Epoch 7/20
    5s - loss: 1.7928 - acc: 0.4581 - val_loss: 1.8258 - val_acc: 0.4513
    Epoch 8/20
    5s - loss: 1.7348 - acc: 0.4767 - val_loss: 1.8050 - val_acc: 0.4529
    Epoch 9/20
    5s - loss: 1.6886 - acc: 0.4919 - val_loss: 1.8285 - val_acc: 0.4652
    Epoch 10/20
    5s - loss: 1.6443 - acc: 0.5001 - val_loss: 1.7323 - val_acc: 0.4816
    Epoch 11/20
    5s - loss: 1.6006 - acc: 0.5150 - val_loss: 1.6988 - val_acc: 0.5061
    Epoch 12/20
    5s - loss: 1.5606 - acc: 0.5268 - val_loss: 1.6877 - val_acc: 0.4971
    Epoch 13/20
    5s - loss: 1.5242 - acc: 0.5357 - val_loss: 1.6951 - val_acc: 0.4971
    Epoch 14/20
    Epoch 00013: early stopping
    5s - loss: 1.4894 - acc: 0.5450 - val_loss: 1.7083 - val_acc: 0.4660
    PREDICT
    6104/6104 [==============================] - 3s     
    PREDICT
    12192/12209 [============================>.] - ETA: 0sFIT PARAMS : 
    {'nb_filters_1': 32, 'nb_filters_2': 64, 'filter_size_2': 6, 'pool_size_1': 4, 'pool_size_2': 4, 'filter_size_1': 3}
    Train on 10988 samples, validate on 1221 samples
    Epoch 1/20
    7s - loss: 2.4499 - acc: 0.2990 - val_loss: 2.3163 - val_acc: 0.3620
    Epoch 2/20
    5s - loss: 2.2388 - acc: 0.3575 - val_loss: 2.1322 - val_acc: 0.3890
    Epoch 3/20
    5s - loss: 2.1013 - acc: 0.3810 - val_loss: 2.0321 - val_acc: 0.4169
    Epoch 4/20
    5s - loss: 1.9787 - acc: 0.4097 - val_loss: 1.9882 - val_acc: 0.4439
    Epoch 5/20
    5s - loss: 1.8809 - acc: 0.4368 - val_loss: 1.8758 - val_acc: 0.4365
    Epoch 6/20
    5s - loss: 1.8022 - acc: 0.4588 - val_loss: 1.8121 - val_acc: 0.4693
    Epoch 7/20
    5s - loss: 1.7367 - acc: 0.4750 - val_loss: 1.7701 - val_acc: 0.4734
    Epoch 8/20
    5s - loss: 1.6793 - acc: 0.4904 - val_loss: 1.7249 - val_acc: 0.4898
    Epoch 9/20
    5s - loss: 1.6275 - acc: 0.5104 - val_loss: 1.7406 - val_acc: 0.4767
    Epoch 10/20
    5s - loss: 1.5818 - acc: 0.5207 - val_loss: 1.7108 - val_acc: 0.4988
    Epoch 11/20
    5s - loss: 1.5390 - acc: 0.5341 - val_loss: 1.6235 - val_acc: 0.5201
    Epoch 12/20
    5s - loss: 1.4961 - acc: 0.5459 - val_loss: 1.6311 - val_acc: 0.5168
    Epoch 13/20
    5s - loss: 1.4581 - acc: 0.5558 - val_loss: 1.6115 - val_acc: 0.5233
    Epoch 14/20
    5s - loss: 1.4159 - acc: 0.5701 - val_loss: 1.5886 - val_acc: 0.5299
    Epoch 15/20
    5s - loss: 1.3828 - acc: 0.5783 - val_loss: 1.6466 - val_acc: 0.5078
    Epoch 16/20
    5s - loss: 1.3414 - acc: 0.5928 - val_loss: 1.5372 - val_acc: 0.5455
    Epoch 17/20
    5s - loss: 1.3129 - acc: 0.6015 - val_loss: 1.5274 - val_acc: 0.5389
    Epoch 18/20
    5s - loss: 1.2760 - acc: 0.6091 - val_loss: 1.6005 - val_acc: 0.5274
    Epoch 19/20
    Epoch 00018: early stopping
    5s - loss: 1.2441 - acc: 0.6192 - val_loss: 1.5283 - val_acc: 0.5602
    PREDICT
    6104/6104 [==============================] - 3s     
    PREDICT
    12192/12209 [============================>.] - ETA: 0sFIT PARAMS : 
    {'nb_filters_1': 64, 'nb_filters_2': 32, 'filter_size_2': 6, 'pool_size_1': 2, 'pool_size_2': 2, 'filter_size_1': 3}
    Train on 10987 samples, validate on 1221 samples
    Epoch 1/20
    15s - loss: 2.4035 - acc: 0.3186 - val_loss: 2.2169 - val_acc: 0.3702
    Epoch 2/20
    12s - loss: 2.0822 - acc: 0.3928 - val_loss: 2.0106 - val_acc: 0.4021
    Epoch 3/20
    12s - loss: 1.8807 - acc: 0.4428 - val_loss: 1.9533 - val_acc: 0.4316
    Epoch 4/20
    12s - loss: 1.7506 - acc: 0.4739 - val_loss: 1.7959 - val_acc: 0.4595
    Epoch 5/20
    12s - loss: 1.6586 - acc: 0.5033 - val_loss: 1.7725 - val_acc: 0.4947
    Epoch 6/20
    12s - loss: 1.5859 - acc: 0.5182 - val_loss: 1.7145 - val_acc: 0.4996
    Epoch 7/20
    12s - loss: 1.5192 - acc: 0.5347 - val_loss: 1.6346 - val_acc: 0.5201
    Epoch 8/20
    12s - loss: 1.4588 - acc: 0.5561 - val_loss: 1.7423 - val_acc: 0.4914
    Epoch 9/20
    Epoch 00008: early stopping
    12s - loss: 1.3942 - acc: 0.5770 - val_loss: 1.7357 - val_acc: 0.4980
    PREDICT
    6105/6105 [==============================] - 4s     
    PREDICT
    12208/12208 [==============================] - 5s     
    FIT PARAMS : 
    {'nb_filters_1': 64, 'nb_filters_2': 32, 'filter_size_2': 6, 'pool_size_1': 2, 'pool_size_2': 2, 'filter_size_1': 3}
    Train on 10988 samples, validate on 1221 samples
    Epoch 1/20
    14s - loss: 2.3965 - acc: 0.3205 - val_loss: 2.2592 - val_acc: 0.3792
    Epoch 2/20
    12s - loss: 2.1003 - acc: 0.3844 - val_loss: 2.0367 - val_acc: 0.3997
    Epoch 3/20
    12s - loss: 1.8959 - acc: 0.4256 - val_loss: 1.8827 - val_acc: 0.4488
    Epoch 4/20
    12s - loss: 1.7672 - acc: 0.4675 - val_loss: 1.8243 - val_acc: 0.4521
    Epoch 5/20
    12s - loss: 1.6723 - acc: 0.4902 - val_loss: 1.7154 - val_acc: 0.5029
    Epoch 6/20
    12s - loss: 1.5976 - acc: 0.5118 - val_loss: 1.6946 - val_acc: 0.4963
    Epoch 7/20
    12s - loss: 1.5207 - acc: 0.5335 - val_loss: 1.6233 - val_acc: 0.5201
    Epoch 8/20
    12s - loss: 1.4579 - acc: 0.5471 - val_loss: 1.6428 - val_acc: 0.5111
    Epoch 9/20
    12s - loss: 1.3978 - acc: 0.5713 - val_loss: 1.6069 - val_acc: 0.5217
    Epoch 10/20
    12s - loss: 1.3319 - acc: 0.5875 - val_loss: 1.6027 - val_acc: 0.5045
    Epoch 11/20
    12s - loss: 1.2678 - acc: 0.6061 - val_loss: 1.6070 - val_acc: 0.5094
    Epoch 12/20
    Epoch 00011: early stopping
    12s - loss: 1.2084 - acc: 0.6246 - val_loss: 1.6403 - val_acc: 0.5135
    PREDICT
    6104/6104 [==============================] - 4s     
    PREDICT
    12209/12209 [==============================] - 5s     
    FIT PARAMS : 
    {'nb_filters_1': 64, 'nb_filters_2': 32, 'filter_size_2': 6, 'pool_size_1': 2, 'pool_size_2': 2, 'filter_size_1': 3}
    Train on 10988 samples, validate on 1221 samples
    Epoch 1/20
    14s - loss: 2.4095 - acc: 0.3164 - val_loss: 2.2109 - val_acc: 0.3931
    Epoch 2/20
    12s - loss: 2.0937 - acc: 0.3849 - val_loss: 2.0075 - val_acc: 0.4120
    Epoch 3/20
    12s - loss: 1.9039 - acc: 0.4294 - val_loss: 1.9428 - val_acc: 0.4349
    Epoch 4/20
    12s - loss: 1.7885 - acc: 0.4618 - val_loss: 1.8377 - val_acc: 0.4701
    Epoch 5/20
    12s - loss: 1.7013 - acc: 0.4854 - val_loss: 1.8051 - val_acc: 0.4668
    Epoch 6/20
    12s - loss: 1.6242 - acc: 0.5046 - val_loss: 1.8160 - val_acc: 0.4709
    Epoch 7/20
    12s - loss: 1.5541 - acc: 0.5278 - val_loss: 1.7326 - val_acc: 0.4865
    Epoch 8/20
    12s - loss: 1.4785 - acc: 0.5467 - val_loss: 1.6426 - val_acc: 0.5184
    Epoch 9/20
    12s - loss: 1.4104 - acc: 0.5669 - val_loss: 1.7058 - val_acc: 0.4889
    Epoch 10/20
    Epoch 00009: early stopping
    12s - loss: 1.3453 - acc: 0.5833 - val_loss: 1.7096 - val_acc: 0.5086
    PREDICT
    6104/6104 [==============================] - 4s     
    PREDICT
    12192/12209 [============================>.] - ETA: 0sFIT PARAMS : 
    {'nb_filters_1': 64, 'nb_filters_2': 32, 'filter_size_2': 6, 'pool_size_1': 2, 'pool_size_2': 4, 'filter_size_1': 3}
    Train on 10987 samples, validate on 1221 samples
    Epoch 1/20
    15s - loss: 2.3951 - acc: 0.3202 - val_loss: 2.3240 - val_acc: 0.3473
    Epoch 2/20
    12s - loss: 2.1827 - acc: 0.3747 - val_loss: 2.1559 - val_acc: 0.3726
    Epoch 3/20
    12s - loss: 2.0381 - acc: 0.3975 - val_loss: 2.0168 - val_acc: 0.4095
    Epoch 4/20
    12s - loss: 1.9051 - acc: 0.4261 - val_loss: 1.9397 - val_acc: 0.4349
    Epoch 5/20
    12s - loss: 1.8051 - acc: 0.4562 - val_loss: 1.8047 - val_acc: 0.4840
    Epoch 6/20
    12s - loss: 1.7261 - acc: 0.4813 - val_loss: 1.8517 - val_acc: 0.4480
    Epoch 7/20
    12s - loss: 1.6616 - acc: 0.5000 - val_loss: 1.7372 - val_acc: 0.5061
    Epoch 8/20
    12s - loss: 1.6042 - acc: 0.5129 - val_loss: 1.6845 - val_acc: 0.5102
    Epoch 9/20
    12s - loss: 1.5561 - acc: 0.5304 - val_loss: 1.7047 - val_acc: 0.5012
    Epoch 10/20
    Epoch 00009: early stopping
    12s - loss: 1.5087 - acc: 0.5427 - val_loss: 1.8483 - val_acc: 0.4701
    PREDICT
    6105/6105 [==============================] - 4s     
    PREDICT
    12192/12208 [============================>.] - ETA: 0sFIT PARAMS : 
    {'nb_filters_1': 64, 'nb_filters_2': 32, 'filter_size_2': 6, 'pool_size_1': 2, 'pool_size_2': 4, 'filter_size_1': 3}
    Train on 10988 samples, validate on 1221 samples
    Epoch 1/20
    14s - loss: 2.3950 - acc: 0.3210 - val_loss: 2.2629 - val_acc: 0.3677
    Epoch 2/20
    12s - loss: 2.1503 - acc: 0.3718 - val_loss: 2.0924 - val_acc: 0.3997
    Epoch 3/20
    12s - loss: 2.0201 - acc: 0.3972 - val_loss: 1.9730 - val_acc: 0.4242
    Epoch 4/20
    12s - loss: 1.9110 - acc: 0.4256 - val_loss: 1.9136 - val_acc: 0.4308
    Epoch 5/20
    12s - loss: 1.8138 - acc: 0.4517 - val_loss: 1.8561 - val_acc: 0.4701
    Epoch 6/20
    12s - loss: 1.7396 - acc: 0.4698 - val_loss: 1.7692 - val_acc: 0.4758
    Epoch 7/20
    12s - loss: 1.6760 - acc: 0.4890 - val_loss: 1.7432 - val_acc: 0.4734
    Epoch 8/20
    12s - loss: 1.6215 - acc: 0.5038 - val_loss: 1.6990 - val_acc: 0.5020
    Epoch 9/20
    12s - loss: 1.5760 - acc: 0.5177 - val_loss: 1.6876 - val_acc: 0.5037
    Epoch 10/20
    12s - loss: 1.5296 - acc: 0.5269 - val_loss: 1.6234 - val_acc: 0.5266
    Epoch 11/20
    12s - loss: 1.4841 - acc: 0.5439 - val_loss: 1.6616 - val_acc: 0.5061
    Epoch 12/20
    Epoch 00011: early stopping
    12s - loss: 1.4472 - acc: 0.5548 - val_loss: 1.7016 - val_acc: 0.5078
    PREDICT
    6104/6104 [==============================] - 4s     
    PREDICT
    12192/12209 [============================>.] - ETA: 0sFIT PARAMS : 
    {'nb_filters_1': 64, 'nb_filters_2': 32, 'filter_size_2': 6, 'pool_size_1': 2, 'pool_size_2': 4, 'filter_size_1': 3}
    Train on 10988 samples, validate on 1221 samples
    Epoch 1/20
    14s - loss: 2.4672 - acc: 0.2935 - val_loss: 2.3089 - val_acc: 0.3358
    Epoch 2/20
    12s - loss: 2.1733 - acc: 0.3703 - val_loss: 2.0384 - val_acc: 0.4087
    Epoch 3/20
    12s - loss: 2.0086 - acc: 0.4018 - val_loss: 1.9857 - val_acc: 0.4251
    Epoch 4/20
    12s - loss: 1.9009 - acc: 0.4311 - val_loss: 1.8927 - val_acc: 0.4406
    Epoch 5/20
    12s - loss: 1.8111 - acc: 0.4552 - val_loss: 1.8275 - val_acc: 0.4529
    Epoch 6/20
    12s - loss: 1.7347 - acc: 0.4798 - val_loss: 1.7492 - val_acc: 0.4930
    Epoch 7/20
    12s - loss: 1.6713 - acc: 0.4938 - val_loss: 1.7517 - val_acc: 0.4898
    Epoch 8/20
    12s - loss: 1.6134 - acc: 0.5106 - val_loss: 1.6891 - val_acc: 0.5111
    Epoch 9/20
    12s - loss: 1.5684 - acc: 0.5257 - val_loss: 1.6632 - val_acc: 0.5209
    Epoch 10/20
    12s - loss: 1.5150 - acc: 0.5398 - val_loss: 1.6183 - val_acc: 0.5160
    Epoch 11/20
    12s - loss: 1.4723 - acc: 0.5528 - val_loss: 1.6727 - val_acc: 0.5094
    Epoch 12/20
    Epoch 00011: early stopping
    12s - loss: 1.4303 - acc: 0.5648 - val_loss: 1.6577 - val_acc: 0.5176
    PREDICT
    6104/6104 [==============================] - 4s     
    PREDICT
    12192/12209 [============================>.] - ETA: 0sFIT PARAMS : 
    {'nb_filters_1': 64, 'nb_filters_2': 32, 'filter_size_2': 6, 'pool_size_1': 4, 'pool_size_2': 2, 'filter_size_1': 3}
    Train on 10987 samples, validate on 1221 samples
    Epoch 1/20
    9s - loss: 2.4010 - acc: 0.3211 - val_loss: 2.2541 - val_acc: 0.3636
    Epoch 2/20
    6s - loss: 2.1700 - acc: 0.3726 - val_loss: 2.0888 - val_acc: 0.3849
    Epoch 3/20
    6s - loss: 2.0178 - acc: 0.4053 - val_loss: 2.0062 - val_acc: 0.4021
    Epoch 4/20
    6s - loss: 1.8927 - acc: 0.4372 - val_loss: 1.9264 - val_acc: 0.4529
    Epoch 5/20
    6s - loss: 1.7961 - acc: 0.4682 - val_loss: 1.8105 - val_acc: 0.4758
    Epoch 6/20
    6s - loss: 1.7316 - acc: 0.4798 - val_loss: 1.8035 - val_acc: 0.4840
    Epoch 7/20
    6s - loss: 1.6732 - acc: 0.4939 - val_loss: 1.7195 - val_acc: 0.5061
    Epoch 8/20
    6s - loss: 1.6215 - acc: 0.5103 - val_loss: 1.7058 - val_acc: 0.5078
    Epoch 9/20
    6s - loss: 1.5777 - acc: 0.5186 - val_loss: 1.7618 - val_acc: 0.4808
    Epoch 10/20
    Epoch 00009: early stopping
    6s - loss: 1.5297 - acc: 0.5353 - val_loss: 1.7239 - val_acc: 0.5086
    PREDICT
    6105/6105 [==============================] - 3s     
    PREDICT
    12208/12208 [==============================] - 3s     
    FIT PARAMS : 
    {'nb_filters_1': 64, 'nb_filters_2': 32, 'filter_size_2': 6, 'pool_size_1': 4, 'pool_size_2': 2, 'filter_size_1': 3}
    Train on 10988 samples, validate on 1221 samples
    Epoch 1/20
    9s - loss: 2.4317 - acc: 0.3089 - val_loss: 2.2592 - val_acc: 0.3530
    Epoch 2/20
    6s - loss: 2.1916 - acc: 0.3673 - val_loss: 2.1304 - val_acc: 0.3817
    Epoch 3/20
    6s - loss: 2.0352 - acc: 0.3984 - val_loss: 1.9744 - val_acc: 0.4103
    Epoch 4/20
    6s - loss: 1.9148 - acc: 0.4232 - val_loss: 1.9004 - val_acc: 0.4373
    Epoch 5/20
    6s - loss: 1.8177 - acc: 0.4440 - val_loss: 1.8354 - val_acc: 0.4636
    Epoch 6/20
    6s - loss: 1.7493 - acc: 0.4691 - val_loss: 1.7645 - val_acc: 0.4816
    Epoch 7/20
    6s - loss: 1.6895 - acc: 0.4863 - val_loss: 1.7398 - val_acc: 0.4922
    Epoch 8/20
    6s - loss: 1.6437 - acc: 0.4988 - val_loss: 1.7099 - val_acc: 0.4848
    Epoch 9/20
    6s - loss: 1.5970 - acc: 0.5144 - val_loss: 1.7151 - val_acc: 0.4865
    Epoch 10/20
    6s - loss: 1.5529 - acc: 0.5275 - val_loss: 1.6366 - val_acc: 0.5119
    Epoch 11/20
    6s - loss: 1.5143 - acc: 0.5321 - val_loss: 1.6459 - val_acc: 0.5053
    Epoch 12/20
    Epoch 00011: early stopping
    6s - loss: 1.4721 - acc: 0.5471 - val_loss: 1.6541 - val_acc: 0.5078
    PREDICT
    6104/6104 [==============================] - 3s     
    PREDICT
    12209/12209 [==============================] - 3s     
    FIT PARAMS : 
    {'nb_filters_1': 64, 'nb_filters_2': 32, 'filter_size_2': 6, 'pool_size_1': 4, 'pool_size_2': 2, 'filter_size_1': 3}
    Train on 10988 samples, validate on 1221 samples
    Epoch 1/20
    9s - loss: 2.4187 - acc: 0.3161 - val_loss: 2.2450 - val_acc: 0.3710
    Epoch 2/20
    6s - loss: 2.1713 - acc: 0.3719 - val_loss: 2.0789 - val_acc: 0.4136
    Epoch 3/20
    6s - loss: 1.9900 - acc: 0.4117 - val_loss: 1.9510 - val_acc: 0.4333
    Epoch 4/20
    6s - loss: 1.8608 - acc: 0.4439 - val_loss: 1.9288 - val_acc: 0.4234
    Epoch 5/20
    6s - loss: 1.7756 - acc: 0.4655 - val_loss: 1.8244 - val_acc: 0.4808
    Epoch 6/20
    6s - loss: 1.7080 - acc: 0.4817 - val_loss: 1.7305 - val_acc: 0.4799
    Epoch 7/20
    6s - loss: 1.6508 - acc: 0.4994 - val_loss: 1.7251 - val_acc: 0.4996
    Epoch 8/20
    6s - loss: 1.5957 - acc: 0.5144 - val_loss: 1.6954 - val_acc: 0.5061
    Epoch 9/20
    6s - loss: 1.5476 - acc: 0.5300 - val_loss: 1.6411 - val_acc: 0.5381
    Epoch 10/20
    6s - loss: 1.5007 - acc: 0.5400 - val_loss: 1.6704 - val_acc: 0.5070
    Epoch 11/20
    6s - loss: 1.4549 - acc: 0.5582 - val_loss: 1.5932 - val_acc: 0.5266
    Epoch 12/20
    6s - loss: 1.4163 - acc: 0.5653 - val_loss: 1.7928 - val_acc: 0.4570
    Epoch 13/20
    6s - loss: 1.3734 - acc: 0.5777 - val_loss: 1.5536 - val_acc: 0.5438
    Epoch 14/20
    6s - loss: 1.3347 - acc: 0.5875 - val_loss: 1.5606 - val_acc: 0.5455
    Epoch 15/20
    Epoch 00014: early stopping
    6s - loss: 1.2965 - acc: 0.6059 - val_loss: 1.5912 - val_acc: 0.5233
    PREDICT
    6104/6104 [==============================] - 3s     
    PREDICT
    12192/12209 [============================>.] - ETA: 0sFIT PARAMS : 
    {'nb_filters_1': 64, 'nb_filters_2': 32, 'filter_size_2': 6, 'pool_size_1': 4, 'pool_size_2': 4, 'filter_size_1': 3}
    Train on 10987 samples, validate on 1221 samples
    Epoch 1/20
    9s - loss: 2.4556 - acc: 0.2945 - val_loss: 2.3655 - val_acc: 0.3514
    Epoch 2/20
    6s - loss: 2.2677 - acc: 0.3633 - val_loss: 2.2052 - val_acc: 0.3628
    Epoch 3/20
    6s - loss: 2.1416 - acc: 0.3803 - val_loss: 2.0752 - val_acc: 0.3997
    Epoch 4/20
    6s - loss: 2.0362 - acc: 0.4006 - val_loss: 1.9878 - val_acc: 0.4128
    Epoch 5/20
    6s - loss: 1.9442 - acc: 0.4210 - val_loss: 1.9055 - val_acc: 0.4242
    Epoch 6/20
    6s - loss: 1.8689 - acc: 0.4392 - val_loss: 1.8814 - val_acc: 0.4570
    Epoch 7/20
    6s - loss: 1.8057 - acc: 0.4599 - val_loss: 1.8731 - val_acc: 0.4676
    Epoch 8/20
    6s - loss: 1.7542 - acc: 0.4762 - val_loss: 1.8145 - val_acc: 0.4726
    Epoch 9/20
    6s - loss: 1.7087 - acc: 0.4875 - val_loss: 1.7512 - val_acc: 0.4881
    Epoch 10/20
    6s - loss: 1.6668 - acc: 0.4992 - val_loss: 1.7335 - val_acc: 0.5029
    Epoch 11/20
    6s - loss: 1.6308 - acc: 0.5090 - val_loss: 1.7200 - val_acc: 0.5111
    Epoch 12/20
    6s - loss: 1.5914 - acc: 0.5195 - val_loss: 1.7055 - val_acc: 0.4865
    Epoch 13/20
    6s - loss: 1.5577 - acc: 0.5294 - val_loss: 1.6956 - val_acc: 0.4914
    Epoch 14/20
    6s - loss: 1.5266 - acc: 0.5368 - val_loss: 1.6352 - val_acc: 0.5307
    Epoch 15/20
    6s - loss: 1.4975 - acc: 0.5456 - val_loss: 1.6164 - val_acc: 0.5127
    Epoch 16/20
    6s - loss: 1.4664 - acc: 0.5544 - val_loss: 1.6213 - val_acc: 0.5250
    Epoch 17/20
    Epoch 00016: early stopping
    6s - loss: 1.4399 - acc: 0.5654 - val_loss: 1.6487 - val_acc: 0.5037
    PREDICT
    6105/6105 [==============================] - 3s     
    PREDICT
    12192/12208 [============================>.] - ETA: 0sFIT PARAMS : 
    {'nb_filters_1': 64, 'nb_filters_2': 32, 'filter_size_2': 6, 'pool_size_1': 4, 'pool_size_2': 4, 'filter_size_1': 3}
    Train on 10988 samples, validate on 1221 samples
    Epoch 1/20
    9s - loss: 2.4350 - acc: 0.3080 - val_loss: 2.2706 - val_acc: 0.3628
    Epoch 2/20
    6s - loss: 2.2153 - acc: 0.3659 - val_loss: 2.1407 - val_acc: 0.3800
    Epoch 3/20
    6s - loss: 2.0819 - acc: 0.3876 - val_loss: 2.0174 - val_acc: 0.3964
    Epoch 4/20
    6s - loss: 1.9726 - acc: 0.4141 - val_loss: 1.9682 - val_acc: 0.3948
    Epoch 5/20
    6s - loss: 1.8868 - acc: 0.4299 - val_loss: 1.9312 - val_acc: 0.4259
    Epoch 6/20
    6s - loss: 1.8174 - acc: 0.4480 - val_loss: 1.8377 - val_acc: 0.4660
    Epoch 7/20
    6s - loss: 1.7550 - acc: 0.4670 - val_loss: 1.8180 - val_acc: 0.4619
    Epoch 8/20
    6s - loss: 1.6990 - acc: 0.4845 - val_loss: 1.8027 - val_acc: 0.4595
    Epoch 9/20
    6s - loss: 1.6578 - acc: 0.4957 - val_loss: 1.9749 - val_acc: 0.4308
    Epoch 10/20
    6s - loss: 1.6147 - acc: 0.5066 - val_loss: 1.7035 - val_acc: 0.4922
    Epoch 11/20
    6s - loss: 1.5785 - acc: 0.5172 - val_loss: 1.6859 - val_acc: 0.4922
    Epoch 12/20
    6s - loss: 1.5414 - acc: 0.5287 - val_loss: 1.6876 - val_acc: 0.4988
    Epoch 13/20
    6s - loss: 1.5111 - acc: 0.5356 - val_loss: 1.6551 - val_acc: 0.5053
    Epoch 14/20
    6s - loss: 1.4761 - acc: 0.5457 - val_loss: 1.6587 - val_acc: 0.4947
    Epoch 15/20
    6s - loss: 1.4438 - acc: 0.5577 - val_loss: 1.6300 - val_acc: 0.5201
    Epoch 16/20
    6s - loss: 1.4127 - acc: 0.5678 - val_loss: 1.6420 - val_acc: 0.5143
    Epoch 17/20
    6s - loss: 1.3846 - acc: 0.5726 - val_loss: 1.6045 - val_acc: 0.5283
    Epoch 18/20
    6s - loss: 1.3584 - acc: 0.5853 - val_loss: 1.5827 - val_acc: 0.5291
    Epoch 19/20
    6s - loss: 1.3263 - acc: 0.5899 - val_loss: 1.5967 - val_acc: 0.5332
    Epoch 20/20
    Epoch 00019: early stopping
    6s - loss: 1.3026 - acc: 0.5968 - val_loss: 1.6046 - val_acc: 0.5250
    PREDICT
    6104/6104 [==============================] - 3s     
    PREDICT
    12192/12209 [============================>.] - ETA: 0sFIT PARAMS : 
    {'nb_filters_1': 64, 'nb_filters_2': 32, 'filter_size_2': 6, 'pool_size_1': 4, 'pool_size_2': 4, 'filter_size_1': 3}
    Train on 10988 samples, validate on 1221 samples
    Epoch 1/20
    9s - loss: 2.4775 - acc: 0.2826 - val_loss: 2.4103 - val_acc: 0.3104
    Epoch 2/20
    6s - loss: 2.3571 - acc: 0.3123 - val_loss: 2.2265 - val_acc: 0.3661
    Epoch 3/20
    6s - loss: 2.1841 - acc: 0.3644 - val_loss: 2.1363 - val_acc: 0.3857
    Epoch 4/20
    6s - loss: 2.0651 - acc: 0.3906 - val_loss: 2.0038 - val_acc: 0.4128
    Epoch 5/20
    6s - loss: 1.9703 - acc: 0.4126 - val_loss: 1.9325 - val_acc: 0.4373
    Epoch 6/20
    6s - loss: 1.8923 - acc: 0.4343 - val_loss: 1.9002 - val_acc: 0.4414
    Epoch 7/20
    6s - loss: 1.8290 - acc: 0.4514 - val_loss: 1.8852 - val_acc: 0.4423
    Epoch 8/20
    6s - loss: 1.7785 - acc: 0.4677 - val_loss: 1.7854 - val_acc: 0.4775
    Epoch 9/20
    6s - loss: 1.7347 - acc: 0.4735 - val_loss: 1.7698 - val_acc: 0.4783
    Epoch 10/20
    6s - loss: 1.6920 - acc: 0.4919 - val_loss: 1.7628 - val_acc: 0.4816
    Epoch 11/20
    6s - loss: 1.6500 - acc: 0.5016 - val_loss: 1.7067 - val_acc: 0.4996
    Epoch 12/20
    6s - loss: 1.6173 - acc: 0.5117 - val_loss: 1.6833 - val_acc: 0.4881
    Epoch 13/20
    6s - loss: 1.5813 - acc: 0.5213 - val_loss: 1.6853 - val_acc: 0.4963
    Epoch 14/20
    Epoch 00013: early stopping
    6s - loss: 1.5472 - acc: 0.5331 - val_loss: 1.7824 - val_acc: 0.4767
    PREDICT
    6104/6104 [==============================] - 3s     
    PREDICT
    12192/12209 [============================>.] - ETA: 0sFIT PARAMS : 
    {'nb_filters_1': 64, 'nb_filters_2': 64, 'filter_size_2': 6, 'pool_size_1': 2, 'pool_size_2': 2, 'filter_size_1': 3}
    Train on 10987 samples, validate on 1221 samples
    Epoch 1/20
    20s - loss: 2.3644 - acc: 0.3316 - val_loss: 2.1292 - val_acc: 0.3825
    Epoch 2/20
    17s - loss: 2.0239 - acc: 0.4043 - val_loss: 1.9742 - val_acc: 0.4300
    Epoch 3/20
    17s - loss: 1.8104 - acc: 0.4603 - val_loss: 1.8962 - val_acc: 0.4521
    Epoch 4/20
    17s - loss: 1.6818 - acc: 0.4963 - val_loss: 1.7070 - val_acc: 0.4971
    Epoch 5/20
    17s - loss: 1.5750 - acc: 0.5213 - val_loss: 1.7323 - val_acc: 0.4898
    Epoch 6/20
    17s - loss: 1.4891 - acc: 0.5461 - val_loss: 1.6446 - val_acc: 0.5135
    Epoch 7/20
    17s - loss: 1.4008 - acc: 0.5694 - val_loss: 1.6105 - val_acc: 0.5225
    Epoch 8/20
    17s - loss: 1.3180 - acc: 0.5925 - val_loss: 1.6103 - val_acc: 0.5242
    Epoch 9/20
    17s - loss: 1.2420 - acc: 0.6153 - val_loss: 1.6099 - val_acc: 0.5201
    Epoch 10/20
    17s - loss: 1.1611 - acc: 0.6378 - val_loss: 1.6390 - val_acc: 0.5111
    Epoch 11/20
    Epoch 00010: early stopping
    17s - loss: 1.0747 - acc: 0.6660 - val_loss: 1.6968 - val_acc: 0.5348
    PREDICT
    6105/6105 [==============================] - 5s     
    PREDICT
    12208/12208 [==============================] - 6s     
    FIT PARAMS : 
    {'nb_filters_1': 64, 'nb_filters_2': 64, 'filter_size_2': 6, 'pool_size_1': 2, 'pool_size_2': 2, 'filter_size_1': 3}
    Train on 10988 samples, validate on 1221 samples
    Epoch 1/20
    20s - loss: 2.3809 - acc: 0.3233 - val_loss: 2.1586 - val_acc: 0.3759
    Epoch 2/20
    17s - loss: 2.0632 - acc: 0.3894 - val_loss: 2.0379 - val_acc: 0.4120
    Epoch 3/20
    17s - loss: 1.8774 - acc: 0.4359 - val_loss: 1.8460 - val_acc: 0.4513
    Epoch 4/20
    17s - loss: 1.7297 - acc: 0.4738 - val_loss: 1.7411 - val_acc: 0.4726
    Epoch 5/20
    17s - loss: 1.6147 - acc: 0.5049 - val_loss: 1.7958 - val_acc: 0.4840
    Epoch 6/20
    17s - loss: 1.5221 - acc: 0.5305 - val_loss: 1.6276 - val_acc: 0.5176
    Epoch 7/20
    17s - loss: 1.4450 - acc: 0.5543 - val_loss: 1.5974 - val_acc: 0.5250
    Epoch 8/20
    17s - loss: 1.3672 - acc: 0.5735 - val_loss: 1.6113 - val_acc: 0.5078
    Epoch 9/20
    17s - loss: 1.2911 - acc: 0.5989 - val_loss: 1.5720 - val_acc: 0.5152
    Epoch 10/20
    17s - loss: 1.2191 - acc: 0.6184 - val_loss: 1.7508 - val_acc: 0.4963
    Epoch 11/20
    Epoch 00010: early stopping
    17s - loss: 1.1476 - acc: 0.6429 - val_loss: 1.6601 - val_acc: 0.4914
    PREDICT
    6104/6104 [==============================] - 5s     
    PREDICT
    12209/12209 [==============================] - 6s     
    FIT PARAMS : 
    {'nb_filters_1': 64, 'nb_filters_2': 64, 'filter_size_2': 6, 'pool_size_1': 2, 'pool_size_2': 2, 'filter_size_1': 3}
    Train on 10988 samples, validate on 1221 samples
    Epoch 1/20
    20s - loss: 2.3892 - acc: 0.3254 - val_loss: 2.2507 - val_acc: 0.3956
    Epoch 2/20
    17s - loss: 2.0501 - acc: 0.4024 - val_loss: 2.0304 - val_acc: 0.4070
    Epoch 3/20
    17s - loss: 1.8360 - acc: 0.4455 - val_loss: 1.8046 - val_acc: 0.4963
    Epoch 4/20
    17s - loss: 1.6925 - acc: 0.4846 - val_loss: 1.7755 - val_acc: 0.4791
    Epoch 5/20
    17s - loss: 1.5783 - acc: 0.5164 - val_loss: 1.7122 - val_acc: 0.4988
    Epoch 6/20
    17s - loss: 1.4859 - acc: 0.5421 - val_loss: 1.6295 - val_acc: 0.5070
    Epoch 7/20
    17s - loss: 1.3962 - acc: 0.5683 - val_loss: 1.7024 - val_acc: 0.4808
    Epoch 8/20
    Epoch 00007: early stopping
    17s - loss: 1.3096 - acc: 0.5967 - val_loss: 1.7586 - val_acc: 0.4726
    PREDICT
    6104/6104 [==============================] - 5s     
    PREDICT
    12209/12209 [==============================] - 6s     
    FIT PARAMS : 
    {'nb_filters_1': 64, 'nb_filters_2': 64, 'filter_size_2': 6, 'pool_size_1': 2, 'pool_size_2': 4, 'filter_size_1': 3}
    Train on 10987 samples, validate on 1221 samples
    Epoch 1/20
    19s - loss: 2.4104 - acc: 0.3153 - val_loss: 2.3722 - val_acc: 0.3571
    Epoch 2/20
    16s - loss: 2.1624 - acc: 0.3739 - val_loss: 2.0888 - val_acc: 0.3882
    Epoch 3/20
    16s - loss: 1.9503 - acc: 0.4179 - val_loss: 1.8997 - val_acc: 0.4652
    Epoch 4/20
    16s - loss: 1.7847 - acc: 0.4657 - val_loss: 1.8859 - val_acc: 0.4455
    Epoch 5/20
    16s - loss: 1.6816 - acc: 0.4955 - val_loss: 1.7569 - val_acc: 0.5061
    Epoch 6/20
    17s - loss: 1.6048 - acc: 0.5145 - val_loss: 1.7289 - val_acc: 0.4865
    Epoch 7/20
    17s - loss: 1.5371 - acc: 0.5381 - val_loss: 1.7621 - val_acc: 0.4971
    Epoch 8/20
    17s - loss: 1.4759 - acc: 0.5535 - val_loss: 1.6551 - val_acc: 0.5094
    Epoch 9/20
    17s - loss: 1.4213 - acc: 0.5666 - val_loss: 1.7571 - val_acc: 0.4881
    Epoch 10/20
    17s - loss: 1.3655 - acc: 0.5819 - val_loss: 1.6159 - val_acc: 0.5192
    Epoch 11/20
    17s - loss: 1.3132 - acc: 0.5993 - val_loss: 1.6035 - val_acc: 0.5373
    Epoch 12/20
    17s - loss: 1.2686 - acc: 0.6078 - val_loss: 1.6286 - val_acc: 0.5250
    Epoch 13/20
    17s - loss: 1.2141 - acc: 0.6256 - val_loss: 1.5631 - val_acc: 0.5332
    Epoch 14/20
    17s - loss: 1.1686 - acc: 0.6374 - val_loss: 1.6975 - val_acc: 0.4971
    Epoch 15/20
    17s - loss: 1.1207 - acc: 0.6519 - val_loss: 1.5501 - val_acc: 0.5487
    Epoch 16/20
    17s - loss: 1.0769 - acc: 0.6681 - val_loss: 1.5876 - val_acc: 0.5233
    Epoch 17/20
    Epoch 00016: early stopping
    17s - loss: 1.0312 - acc: 0.6807 - val_loss: 1.5666 - val_acc: 0.5430
    PREDICT
    6105/6105 [==============================] - 5s     
    PREDICT
    12192/12208 [============================>.] - ETA: 0sFIT PARAMS : 
    {'nb_filters_1': 64, 'nb_filters_2': 64, 'filter_size_2': 6, 'pool_size_1': 2, 'pool_size_2': 4, 'filter_size_1': 3}
    Train on 10988 samples, validate on 1221 samples
    Epoch 1/20
    19s - loss: 2.4091 - acc: 0.3160 - val_loss: 2.2141 - val_acc: 0.3653
    Epoch 2/20
    16s - loss: 2.1149 - acc: 0.3810 - val_loss: 2.0116 - val_acc: 0.4062
    Epoch 3/20
    16s - loss: 1.9535 - acc: 0.4129 - val_loss: 1.9212 - val_acc: 0.4226
    Epoch 4/20
    16s - loss: 1.8294 - acc: 0.4459 - val_loss: 1.8009 - val_acc: 0.4717
    Epoch 5/20
    16s - loss: 1.7320 - acc: 0.4780 - val_loss: 1.8454 - val_acc: 0.4373
    Epoch 6/20
    16s - loss: 1.6549 - acc: 0.4974 - val_loss: 1.6687 - val_acc: 0.5209
    Epoch 7/20
    16s - loss: 1.5824 - acc: 0.5139 - val_loss: 1.7050 - val_acc: 0.5078
    Epoch 8/20
    16s - loss: 1.5220 - acc: 0.5390 - val_loss: 1.6183 - val_acc: 0.5143
    Epoch 9/20
    16s - loss: 1.4632 - acc: 0.5493 - val_loss: 1.6699 - val_acc: 0.5152
    Epoch 10/20
    16s - loss: 1.4111 - acc: 0.5636 - val_loss: 1.5618 - val_acc: 0.5504
    Epoch 11/20
    16s - loss: 1.3605 - acc: 0.5775 - val_loss: 1.5901 - val_acc: 0.5405
    Epoch 12/20
    16s - loss: 1.3099 - acc: 0.5933 - val_loss: 1.5221 - val_acc: 0.5512
    Epoch 13/20
    16s - loss: 1.2615 - acc: 0.6123 - val_loss: 1.5493 - val_acc: 0.5315
    Epoch 14/20
    Epoch 00013: early stopping
    16s - loss: 1.2128 - acc: 0.6244 - val_loss: 1.5310 - val_acc: 0.5438
    PREDICT
    6104/6104 [==============================] - 5s     
    PREDICT
    12209/12209 [==============================] - 5s     
    FIT PARAMS : 
    {'nb_filters_1': 64, 'nb_filters_2': 64, 'filter_size_2': 6, 'pool_size_1': 2, 'pool_size_2': 4, 'filter_size_1': 3}
    Train on 10988 samples, validate on 1221 samples
    Epoch 1/20
    19s - loss: 2.4466 - acc: 0.3007 - val_loss: 2.3376 - val_acc: 0.3694
    Epoch 2/20
    16s - loss: 2.1569 - acc: 0.3722 - val_loss: 2.0585 - val_acc: 0.4201
    Epoch 3/20
    16s - loss: 1.9804 - acc: 0.4089 - val_loss: 1.9512 - val_acc: 0.4365
    Epoch 4/20
    16s - loss: 1.8557 - acc: 0.4393 - val_loss: 1.8988 - val_acc: 0.4488
    Epoch 5/20
    16s - loss: 1.7606 - acc: 0.4669 - val_loss: 1.7647 - val_acc: 0.4824
    Epoch 6/20
    16s - loss: 1.6815 - acc: 0.4843 - val_loss: 1.7526 - val_acc: 0.4988
    Epoch 7/20
    16s - loss: 1.6143 - acc: 0.5150 - val_loss: 1.6890 - val_acc: 0.5119
    Epoch 8/20
    16s - loss: 1.5549 - acc: 0.5247 - val_loss: 1.6772 - val_acc: 0.5168
    Epoch 9/20
    16s - loss: 1.4950 - acc: 0.5440 - val_loss: 1.7243 - val_acc: 0.4939
    Epoch 10/20
    16s - loss: 1.4402 - acc: 0.5574 - val_loss: 1.6419 - val_acc: 0.5029
    Epoch 11/20
    16s - loss: 1.3883 - acc: 0.5729 - val_loss: 1.6347 - val_acc: 0.5012
    Epoch 12/20
    16s - loss: 1.3432 - acc: 0.5869 - val_loss: 1.5817 - val_acc: 0.5340
    Epoch 13/20
    16s - loss: 1.2943 - acc: 0.6025 - val_loss: 1.5627 - val_acc: 0.5291
    Epoch 14/20
    16s - loss: 1.2506 - acc: 0.6156 - val_loss: 1.6047 - val_acc: 0.5274
    Epoch 15/20
    16s - loss: 1.2098 - acc: 0.6269 - val_loss: 1.5299 - val_acc: 0.5545
    Epoch 16/20
    16s - loss: 1.1643 - acc: 0.6391 - val_loss: 1.6185 - val_acc: 0.5209
    Epoch 17/20
    16s - loss: 1.1221 - acc: 0.6594 - val_loss: 1.5264 - val_acc: 0.5438
    Epoch 18/20
    16s - loss: 1.0794 - acc: 0.6610 - val_loss: 1.5501 - val_acc: 0.5233
    Epoch 19/20
    Epoch 00018: early stopping
    16s - loss: 1.0433 - acc: 0.6732 - val_loss: 1.5353 - val_acc: 0.5561
    PREDICT
    6104/6104 [==============================] - 5s     
    PREDICT
    12209/12209 [==============================] - 5s     
    FIT PARAMS : 
    {'nb_filters_1': 64, 'nb_filters_2': 64, 'filter_size_2': 6, 'pool_size_1': 4, 'pool_size_2': 2, 'filter_size_1': 3}
    Train on 10987 samples, validate on 1221 samples
    Epoch 1/20
    10s - loss: 2.4232 - acc: 0.3093 - val_loss: 2.2865 - val_acc: 0.3554
    Epoch 2/20
    7s - loss: 2.1804 - acc: 0.3734 - val_loss: 2.1316 - val_acc: 0.3956
    Epoch 3/20
    7s - loss: 2.0105 - acc: 0.3999 - val_loss: 1.9682 - val_acc: 0.4054
    Epoch 4/20
    7s - loss: 1.8531 - acc: 0.4420 - val_loss: 1.8310 - val_acc: 0.4537
    Epoch 5/20
    7s - loss: 1.7488 - acc: 0.4740 - val_loss: 1.7685 - val_acc: 0.4873
    Epoch 6/20
    7s - loss: 1.6694 - acc: 0.4947 - val_loss: 1.7436 - val_acc: 0.4824
    Epoch 7/20
    7s - loss: 1.6007 - acc: 0.5152 - val_loss: 1.6959 - val_acc: 0.5012
    Epoch 8/20
    7s - loss: 1.5430 - acc: 0.5323 - val_loss: 1.6369 - val_acc: 0.5242
    Epoch 9/20
    7s - loss: 1.4897 - acc: 0.5425 - val_loss: 1.6213 - val_acc: 0.5324
    Epoch 10/20
    7s - loss: 1.4335 - acc: 0.5613 - val_loss: 1.6223 - val_acc: 0.5143
    Epoch 11/20
    7s - loss: 1.3857 - acc: 0.5750 - val_loss: 1.5462 - val_acc: 0.5397
    Epoch 12/20
    7s - loss: 1.3433 - acc: 0.5914 - val_loss: 1.5230 - val_acc: 0.5422
    Epoch 13/20
    7s - loss: 1.2972 - acc: 0.6019 - val_loss: 1.5590 - val_acc: 0.5430
    Epoch 14/20
    7s - loss: 1.2485 - acc: 0.6153 - val_loss: 1.5003 - val_acc: 0.5708
    Epoch 15/20
    7s - loss: 1.2094 - acc: 0.6283 - val_loss: 1.8572 - val_acc: 0.4177
    Epoch 16/20
    Epoch 00015: early stopping
    7s - loss: 1.1685 - acc: 0.6390 - val_loss: 1.5050 - val_acc: 0.5553
    PREDICT
    6105/6105 [==============================] - 4s     
    PREDICT
    12208/12208 [==============================] - 3s     
    FIT PARAMS : 
    {'nb_filters_1': 64, 'nb_filters_2': 64, 'filter_size_2': 6, 'pool_size_1': 4, 'pool_size_2': 2, 'filter_size_1': 3}
    Train on 10988 samples, validate on 1221 samples
    Epoch 1/20
    10s - loss: 2.4150 - acc: 0.3135 - val_loss: 2.2400 - val_acc: 0.3669
    Epoch 2/20
    7s - loss: 2.1484 - acc: 0.3737 - val_loss: 2.0421 - val_acc: 0.3989
    Epoch 3/20
    7s - loss: 1.9742 - acc: 0.4088 - val_loss: 1.9789 - val_acc: 0.4324
    Epoch 4/20
    7s - loss: 1.8434 - acc: 0.4420 - val_loss: 1.8135 - val_acc: 0.4742
    Epoch 5/20
    7s - loss: 1.7484 - acc: 0.4681 - val_loss: 1.7507 - val_acc: 0.4824
    Epoch 6/20
    7s - loss: 1.6716 - acc: 0.4890 - val_loss: 1.7430 - val_acc: 0.4848
    Epoch 7/20
    7s - loss: 1.6061 - acc: 0.5095 - val_loss: 1.6718 - val_acc: 0.5119
    Epoch 8/20
    7s - loss: 1.5449 - acc: 0.5253 - val_loss: 1.6164 - val_acc: 0.5209
    Epoch 9/20
    7s - loss: 1.4851 - acc: 0.5464 - val_loss: 1.7101 - val_acc: 0.4906
    Epoch 10/20
    Epoch 00009: early stopping
    7s - loss: 1.4332 - acc: 0.5616 - val_loss: 1.6446 - val_acc: 0.5070
    PREDICT
    6104/6104 [==============================] - 4s     
    PREDICT
    12209/12209 [==============================] - 3s     
    FIT PARAMS : 
    {'nb_filters_1': 64, 'nb_filters_2': 64, 'filter_size_2': 6, 'pool_size_1': 4, 'pool_size_2': 2, 'filter_size_1': 3}
    Train on 10988 samples, validate on 1221 samples
    Epoch 1/20
    10s - loss: 2.4249 - acc: 0.3093 - val_loss: 2.3076 - val_acc: 0.3497
    Epoch 2/20
    7s - loss: 2.1530 - acc: 0.3751 - val_loss: 2.1008 - val_acc: 0.3989
    Epoch 3/20
    7s - loss: 1.9670 - acc: 0.4168 - val_loss: 1.8994 - val_acc: 0.4455
    Epoch 4/20
    7s - loss: 1.8288 - acc: 0.4526 - val_loss: 1.8525 - val_acc: 0.4488
    Epoch 5/20
    7s - loss: 1.7401 - acc: 0.4741 - val_loss: 1.8898 - val_acc: 0.4480
    Epoch 6/20
    7s - loss: 1.6666 - acc: 0.4954 - val_loss: 1.7112 - val_acc: 0.4898
    Epoch 7/20
    7s - loss: 1.5950 - acc: 0.5130 - val_loss: 1.6893 - val_acc: 0.5020
    Epoch 8/20
    7s - loss: 1.5357 - acc: 0.5301 - val_loss: 1.7453 - val_acc: 0.4832
    Epoch 9/20
    7s - loss: 1.4853 - acc: 0.5422 - val_loss: 1.6249 - val_acc: 0.5168
    Epoch 10/20
    7s - loss: 1.4292 - acc: 0.5582 - val_loss: 1.5899 - val_acc: 0.5258
    Epoch 11/20
    7s - loss: 1.3781 - acc: 0.5748 - val_loss: 1.5416 - val_acc: 0.5283
    Epoch 12/20
    7s - loss: 1.3295 - acc: 0.5913 - val_loss: 1.5261 - val_acc: 0.5528
    Epoch 13/20
    7s - loss: 1.2819 - acc: 0.6044 - val_loss: 1.4938 - val_acc: 0.5602
    Epoch 14/20
    7s - loss: 1.2347 - acc: 0.6159 - val_loss: 1.5311 - val_acc: 0.5536
    Epoch 15/20
    Epoch 00014: early stopping
    7s - loss: 1.1911 - acc: 0.6325 - val_loss: 1.5374 - val_acc: 0.5479
    PREDICT
    6080/6104 [============================>.] - ETA: 0sPREDICT
    12192/12209 [============================>.] - ETA: 0sFIT PARAMS : 
    {'nb_filters_1': 64, 'nb_filters_2': 64, 'filter_size_2': 6, 'pool_size_1': 4, 'pool_size_2': 4, 'filter_size_1': 3}
    Train on 10987 samples, validate on 1221 samples
    Epoch 1/20
    10s - loss: 2.4269 - acc: 0.3117 - val_loss: 2.2545 - val_acc: 0.3587
    Epoch 2/20
    7s - loss: 2.1913 - acc: 0.3703 - val_loss: 2.0728 - val_acc: 0.3956
    Epoch 3/20
    7s - loss: 2.0313 - acc: 0.4036 - val_loss: 2.0100 - val_acc: 0.3980
    Epoch 4/20
    7s - loss: 1.9070 - acc: 0.4311 - val_loss: 1.9326 - val_acc: 0.4300
    Epoch 5/20
    7s - loss: 1.8127 - acc: 0.4615 - val_loss: 1.8020 - val_acc: 0.4709
    Epoch 6/20
    7s - loss: 1.7317 - acc: 0.4822 - val_loss: 1.8095 - val_acc: 0.4799
    Epoch 7/20
    Epoch 00006: early stopping
    7s - loss: 1.6664 - acc: 0.5029 - val_loss: 1.8075 - val_acc: 0.4619
    PREDICT
    6105/6105 [==============================] - 4s     
    PREDICT
    12192/12208 [============================>.] - ETA: 0sFIT PARAMS : 
    {'nb_filters_1': 64, 'nb_filters_2': 64, 'filter_size_2': 6, 'pool_size_1': 4, 'pool_size_2': 4, 'filter_size_1': 3}
    Train on 10988 samples, validate on 1221 samples
    Epoch 1/20
    10s - loss: 2.4122 - acc: 0.3178 - val_loss: 2.2591 - val_acc: 0.3612
    Epoch 2/20
    7s - loss: 2.1732 - acc: 0.3706 - val_loss: 2.0735 - val_acc: 0.3915
    Epoch 3/20
    7s - loss: 2.0145 - acc: 0.4021 - val_loss: 2.0603 - val_acc: 0.4152
    Epoch 4/20
    7s - loss: 1.8912 - acc: 0.4307 - val_loss: 1.8564 - val_acc: 0.4455
    Epoch 5/20
    7s - loss: 1.8007 - acc: 0.4535 - val_loss: 1.7874 - val_acc: 0.4668
    Epoch 6/20
    7s - loss: 1.7253 - acc: 0.4758 - val_loss: 1.7640 - val_acc: 0.4791
    Epoch 7/20
    7s - loss: 1.6658 - acc: 0.4924 - val_loss: 1.6921 - val_acc: 0.4906
    Epoch 8/20
    7s - loss: 1.6132 - acc: 0.5109 - val_loss: 1.6663 - val_acc: 0.5094
    Epoch 9/20
    7s - loss: 1.5609 - acc: 0.5221 - val_loss: 1.6554 - val_acc: 0.5004
    Epoch 10/20
    7s - loss: 1.5188 - acc: 0.5385 - val_loss: 1.6142 - val_acc: 0.5192
    Epoch 11/20
    7s - loss: 1.4695 - acc: 0.5524 - val_loss: 1.6050 - val_acc: 0.5070
    Epoch 12/20
    7s - loss: 1.4268 - acc: 0.5610 - val_loss: 1.5941 - val_acc: 0.5192
    Epoch 13/20
    7s - loss: 1.3839 - acc: 0.5741 - val_loss: 1.5939 - val_acc: 0.5233
    Epoch 14/20
    7s - loss: 1.3412 - acc: 0.5902 - val_loss: 1.5486 - val_acc: 0.5348
    Epoch 15/20
    7s - loss: 1.3034 - acc: 0.6015 - val_loss: 1.5467 - val_acc: 0.5348
    Epoch 16/20
    7s - loss: 1.2625 - acc: 0.6114 - val_loss: 1.5940 - val_acc: 0.5201
    Epoch 17/20
    7s - loss: 1.2274 - acc: 0.6239 - val_loss: 1.5389 - val_acc: 0.5389
    Epoch 18/20
    7s - loss: 1.1889 - acc: 0.6333 - val_loss: 1.5797 - val_acc: 0.5242
    Epoch 19/20
    7s - loss: 1.1515 - acc: 0.6464 - val_loss: 1.5202 - val_acc: 0.5381
    Epoch 20/20
    7s - loss: 1.1163 - acc: 0.6559 - val_loss: 1.5945 - val_acc: 0.5332
    PREDICT
    6104/6104 [==============================] - 4s     
    PREDICT
    12192/12209 [============================>.] - ETA: 0sFIT PARAMS : 
    {'nb_filters_1': 64, 'nb_filters_2': 64, 'filter_size_2': 6, 'pool_size_1': 4, 'pool_size_2': 4, 'filter_size_1': 3}
    Train on 10988 samples, validate on 1221 samples
    Epoch 1/20
    10s - loss: 2.4469 - acc: 0.3025 - val_loss: 2.2784 - val_acc: 0.3694
    Epoch 2/20
    7s - loss: 2.2307 - acc: 0.3640 - val_loss: 2.1364 - val_acc: 0.3898
    Epoch 3/20
    7s - loss: 2.0926 - acc: 0.3870 - val_loss: 2.0118 - val_acc: 0.4079
    Epoch 4/20
    7s - loss: 1.9600 - acc: 0.4100 - val_loss: 1.9367 - val_acc: 0.4275
    Epoch 5/20
    7s - loss: 1.8531 - acc: 0.4430 - val_loss: 1.8313 - val_acc: 0.4619
    Epoch 6/20
    7s - loss: 1.7717 - acc: 0.4638 - val_loss: 1.7765 - val_acc: 0.4668
    Epoch 7/20
    7s - loss: 1.7049 - acc: 0.4843 - val_loss: 1.7271 - val_acc: 0.4971
    Epoch 8/20
    7s - loss: 1.6493 - acc: 0.5000 - val_loss: 1.7643 - val_acc: 0.4848
    Epoch 9/20
    7s - loss: 1.5912 - acc: 0.5144 - val_loss: 1.7055 - val_acc: 0.4971
    Epoch 10/20
    7s - loss: 1.5444 - acc: 0.5289 - val_loss: 1.6300 - val_acc: 0.5283
    Epoch 11/20
    7s - loss: 1.5011 - acc: 0.5430 - val_loss: 1.6993 - val_acc: 0.5029
    Epoch 12/20
    7s - loss: 1.4524 - acc: 0.5561 - val_loss: 1.5843 - val_acc: 0.5307
    Epoch 13/20
    7s - loss: 1.4112 - acc: 0.5756 - val_loss: 1.5947 - val_acc: 0.5160
    Epoch 14/20
    Epoch 00013: early stopping
    7s - loss: 1.3652 - acc: 0.5853 - val_loss: 1.6137 - val_acc: 0.5143
    PREDICT
    6080/6104 [============================>.] - ETA: 0sPREDICT
    12209/12209 [==============================] - 3s     
    FIT PARAMS : 
    {'nb_filters_1': 32, 'nb_filters_2': 32, 'filter_size_2': 3, 'pool_size_1': 2, 'pool_size_2': 2, 'filter_size_1': 6}
    Train on 10987 samples, validate on 1221 samples
    Epoch 1/20
    9s - loss: 2.4055 - acc: 0.3202 - val_loss: 2.4921 - val_acc: 0.3530
    Epoch 2/20
    6s - loss: 2.1398 - acc: 0.3829 - val_loss: 2.0741 - val_acc: 0.3907
    Epoch 3/20
    6s - loss: 1.9768 - acc: 0.4106 - val_loss: 1.9709 - val_acc: 0.4365
    Epoch 4/20
    6s - loss: 1.8675 - acc: 0.4392 - val_loss: 2.0176 - val_acc: 0.4161
    Epoch 5/20
    6s - loss: 1.7664 - acc: 0.4621 - val_loss: 1.9040 - val_acc: 0.4537
    Epoch 6/20
    6s - loss: 1.6845 - acc: 0.4879 - val_loss: 1.8814 - val_acc: 0.4627
    Epoch 7/20
    6s - loss: 1.6059 - acc: 0.5124 - val_loss: 1.8848 - val_acc: 0.4644
    Epoch 8/20
    Epoch 00007: early stopping
    6s - loss: 1.5295 - acc: 0.5362 - val_loss: 2.0454 - val_acc: 0.4111
    PREDICT
    6105/6105 [==============================] - 3s     
    PREDICT
    12208/12208 [==============================] - 2s     
    FIT PARAMS : 
    {'nb_filters_1': 32, 'nb_filters_2': 32, 'filter_size_2': 3, 'pool_size_1': 2, 'pool_size_2': 2, 'filter_size_1': 6}
    Train on 10988 samples, validate on 1221 samples
    Epoch 1/20
    9s - loss: 2.3851 - acc: 0.3241 - val_loss: 2.2023 - val_acc: 0.3595
    Epoch 2/20
    6s - loss: 2.0998 - acc: 0.3858 - val_loss: 2.0474 - val_acc: 0.3972
    Epoch 3/20
    6s - loss: 1.9588 - acc: 0.4116 - val_loss: 2.0344 - val_acc: 0.4013
    Epoch 4/20
    6s - loss: 1.8580 - acc: 0.4357 - val_loss: 1.9090 - val_acc: 0.4529
    Epoch 5/20
    6s - loss: 1.7698 - acc: 0.4608 - val_loss: 1.8890 - val_acc: 0.4447
    Epoch 6/20
    6s - loss: 1.6866 - acc: 0.4833 - val_loss: 1.8830 - val_acc: 0.4390
    Epoch 7/20
    6s - loss: 1.6075 - acc: 0.5056 - val_loss: 1.8696 - val_acc: 0.4529
    Epoch 8/20
    6s - loss: 1.5270 - acc: 0.5314 - val_loss: 1.8317 - val_acc: 0.4554
    Epoch 9/20
    6s - loss: 1.4465 - acc: 0.5531 - val_loss: 1.8636 - val_acc: 0.4472
    Epoch 10/20
    6s - loss: 1.3677 - acc: 0.5791 - val_loss: 1.8296 - val_acc: 0.4570
    Epoch 11/20
    6s - loss: 1.2893 - acc: 0.6015 - val_loss: 2.0395 - val_acc: 0.3882
    Epoch 12/20
    Epoch 00011: early stopping
    6s - loss: 1.2186 - acc: 0.6235 - val_loss: 1.9279 - val_acc: 0.4611
    PREDICT
    6104/6104 [==============================] - 3s     
    PREDICT
    12209/12209 [==============================] - 2s     
    FIT PARAMS : 
    {'nb_filters_1': 32, 'nb_filters_2': 32, 'filter_size_2': 3, 'pool_size_1': 2, 'pool_size_2': 2, 'filter_size_1': 6}
    Train on 10988 samples, validate on 1221 samples
    Epoch 1/20
    9s - loss: 2.4050 - acc: 0.3197 - val_loss: 2.2098 - val_acc: 0.3890
    Epoch 2/20
    6s - loss: 2.1287 - acc: 0.3817 - val_loss: 2.0859 - val_acc: 0.4111
    Epoch 3/20
    6s - loss: 1.9721 - acc: 0.4130 - val_loss: 1.9732 - val_acc: 0.4283
    Epoch 4/20
    6s - loss: 1.8587 - acc: 0.4378 - val_loss: 1.9068 - val_acc: 0.4423
    Epoch 5/20
    6s - loss: 1.7686 - acc: 0.4603 - val_loss: 1.8905 - val_acc: 0.4562
    Epoch 6/20
    6s - loss: 1.6874 - acc: 0.4840 - val_loss: 1.9304 - val_acc: 0.4275
    Epoch 7/20
    6s - loss: 1.6047 - acc: 0.5096 - val_loss: 1.8620 - val_acc: 0.4595
    Epoch 8/20
    6s - loss: 1.5226 - acc: 0.5319 - val_loss: 1.9636 - val_acc: 0.4455
    Epoch 9/20
    Epoch 00008: early stopping
    6s - loss: 1.4368 - acc: 0.5570 - val_loss: 1.9630 - val_acc: 0.4447
    PREDICT
    6104/6104 [==============================] - 4s     
    PREDICT
    12192/12209 [============================>.] - ETA: 0sFIT PARAMS : 
    {'nb_filters_1': 32, 'nb_filters_2': 32, 'filter_size_2': 3, 'pool_size_1': 2, 'pool_size_2': 4, 'filter_size_1': 6}
    Train on 10987 samples, validate on 1221 samples
    Epoch 1/20
    9s - loss: 2.4190 - acc: 0.3104 - val_loss: 2.2413 - val_acc: 0.3718
    Epoch 2/20
    6s - loss: 2.1779 - acc: 0.3758 - val_loss: 2.1345 - val_acc: 0.3726
    Epoch 3/20
    6s - loss: 2.0404 - acc: 0.3969 - val_loss: 2.0059 - val_acc: 0.4161
    Epoch 4/20
    6s - loss: 1.9366 - acc: 0.4184 - val_loss: 1.9978 - val_acc: 0.4210
    Epoch 5/20
    6s - loss: 1.8686 - acc: 0.4340 - val_loss: 1.9483 - val_acc: 0.4398
    Epoch 6/20
    6s - loss: 1.8114 - acc: 0.4505 - val_loss: 1.8965 - val_acc: 0.4341
    Epoch 7/20
    6s - loss: 1.7686 - acc: 0.4625 - val_loss: 1.8566 - val_acc: 0.4480
    Epoch 8/20
    6s - loss: 1.7220 - acc: 0.4732 - val_loss: 1.8485 - val_acc: 0.4562
    Epoch 9/20
    6s - loss: 1.6838 - acc: 0.4855 - val_loss: 1.8310 - val_acc: 0.4496
    Epoch 10/20
    6s - loss: 1.6407 - acc: 0.5001 - val_loss: 1.8888 - val_acc: 0.4292
    Epoch 11/20
    6s - loss: 1.5991 - acc: 0.5115 - val_loss: 1.7963 - val_acc: 0.4521
    Epoch 12/20
    6s - loss: 1.5599 - acc: 0.5189 - val_loss: 1.7711 - val_acc: 0.4758
    Epoch 13/20
    6s - loss: 1.5236 - acc: 0.5352 - val_loss: 1.7287 - val_acc: 0.4758
    Epoch 14/20
    6s - loss: 1.4860 - acc: 0.5488 - val_loss: 1.7812 - val_acc: 0.4603
    Epoch 15/20
    Epoch 00014: early stopping
    6s - loss: 1.4484 - acc: 0.5516 - val_loss: 1.7897 - val_acc: 0.4726
    PREDICT
    6105/6105 [==============================] - 3s     
    PREDICT
    12192/12208 [============================>.] - ETA: 0sFIT PARAMS : 
    {'nb_filters_1': 32, 'nb_filters_2': 32, 'filter_size_2': 3, 'pool_size_1': 2, 'pool_size_2': 4, 'filter_size_1': 6}
    Train on 10988 samples, validate on 1221 samples
    Epoch 1/20
    9s - loss: 2.4381 - acc: 0.3035 - val_loss: 2.2326 - val_acc: 0.3686
    Epoch 2/20
    6s - loss: 2.1466 - acc: 0.3743 - val_loss: 2.0787 - val_acc: 0.4062
    Epoch 3/20
    6s - loss: 2.0197 - acc: 0.3967 - val_loss: 2.0434 - val_acc: 0.3980
    Epoch 4/20
    6s - loss: 1.9441 - acc: 0.4147 - val_loss: 1.9590 - val_acc: 0.4201
    Epoch 5/20
    6s - loss: 1.8820 - acc: 0.4275 - val_loss: 1.9126 - val_acc: 0.4365
    Epoch 6/20
    6s - loss: 1.8216 - acc: 0.4435 - val_loss: 1.8705 - val_acc: 0.4496
    Epoch 7/20
    6s - loss: 1.7710 - acc: 0.4590 - val_loss: 1.8545 - val_acc: 0.4341
    Epoch 8/20
    6s - loss: 1.7168 - acc: 0.4766 - val_loss: 1.8068 - val_acc: 0.4701
    Epoch 9/20
    6s - loss: 1.6710 - acc: 0.4884 - val_loss: 1.7989 - val_acc: 0.4619
    Epoch 10/20
    6s - loss: 1.6269 - acc: 0.5005 - val_loss: 1.7841 - val_acc: 0.4709
    Epoch 11/20
    6s - loss: 1.5807 - acc: 0.5127 - val_loss: 1.7417 - val_acc: 0.4808
    Epoch 12/20
    6s - loss: 1.5360 - acc: 0.5327 - val_loss: 1.9703 - val_acc: 0.4210
    Epoch 13/20
    Epoch 00012: early stopping
    6s - loss: 1.4950 - acc: 0.5425 - val_loss: 1.7599 - val_acc: 0.4783
    PREDICT
    6104/6104 [==============================] - 4s     
    PREDICT
    12192/12209 [============================>.] - ETA: 0sFIT PARAMS : 
    {'nb_filters_1': 32, 'nb_filters_2': 32, 'filter_size_2': 3, 'pool_size_1': 2, 'pool_size_2': 4, 'filter_size_1': 6}
    Train on 10988 samples, validate on 1221 samples
    Epoch 1/20
    9s - loss: 2.4508 - acc: 0.3025 - val_loss: 2.2746 - val_acc: 0.3841
    Epoch 2/20
    6s - loss: 2.2142 - acc: 0.3679 - val_loss: 2.1691 - val_acc: 0.3980
    Epoch 3/20
    6s - loss: 2.0903 - acc: 0.3858 - val_loss: 2.0451 - val_acc: 0.4103
    Epoch 4/20
    6s - loss: 1.9681 - acc: 0.4082 - val_loss: 1.9697 - val_acc: 0.4226
    Epoch 5/20
    6s - loss: 1.8880 - acc: 0.4320 - val_loss: 1.9513 - val_acc: 0.4292
    Epoch 6/20
    6s - loss: 1.8315 - acc: 0.4416 - val_loss: 1.9022 - val_acc: 0.4439
    Epoch 7/20
    6s - loss: 1.7858 - acc: 0.4553 - val_loss: 1.8554 - val_acc: 0.4562
    Epoch 8/20
    6s - loss: 1.7409 - acc: 0.4702 - val_loss: 1.8729 - val_acc: 0.4472
    Epoch 9/20
    6s - loss: 1.7015 - acc: 0.4783 - val_loss: 1.8412 - val_acc: 0.4693
    Epoch 10/20
    6s - loss: 1.6627 - acc: 0.4911 - val_loss: 1.7811 - val_acc: 0.4914
    Epoch 11/20
    6s - loss: 1.6235 - acc: 0.5050 - val_loss: 1.7444 - val_acc: 0.4816
    Epoch 12/20
    6s - loss: 1.5862 - acc: 0.5144 - val_loss: 1.7494 - val_acc: 0.4955
    Epoch 13/20
    Epoch 00012: early stopping
    6s - loss: 1.5466 - acc: 0.5287 - val_loss: 1.7664 - val_acc: 0.4775
    PREDICT
    6104/6104 [==============================] - 3s     
    PREDICT
    12192/12209 [============================>.] - ETA: 0sFIT PARAMS : 
    {'nb_filters_1': 32, 'nb_filters_2': 32, 'filter_size_2': 3, 'pool_size_1': 4, 'pool_size_2': 2, 'filter_size_1': 6}
    Train on 10987 samples, validate on 1221 samples
    Epoch 1/20
    8s - loss: 2.4604 - acc: 0.2990 - val_loss: 2.3427 - val_acc: 0.3645
    Epoch 2/20
    4s - loss: 2.2135 - acc: 0.3725 - val_loss: 2.1227 - val_acc: 0.3833
    Epoch 3/20
    4s - loss: 2.0596 - acc: 0.3965 - val_loss: 2.0192 - val_acc: 0.4103
    Epoch 4/20
    4s - loss: 1.9484 - acc: 0.4180 - val_loss: 1.9751 - val_acc: 0.4218
    Epoch 5/20
    4s - loss: 1.8732 - acc: 0.4376 - val_loss: 1.8811 - val_acc: 0.4390
    Epoch 6/20
    4s - loss: 1.8125 - acc: 0.4525 - val_loss: 1.9243 - val_acc: 0.4496
    Epoch 7/20
    4s - loss: 1.7587 - acc: 0.4674 - val_loss: 1.8561 - val_acc: 0.4619
    Epoch 8/20
    4s - loss: 1.7065 - acc: 0.4882 - val_loss: 1.7731 - val_acc: 0.4791
    Epoch 9/20
    4s - loss: 1.6612 - acc: 0.5004 - val_loss: 1.7584 - val_acc: 0.4832
    Epoch 10/20
    4s - loss: 1.6191 - acc: 0.5158 - val_loss: 1.7370 - val_acc: 0.4996
    Epoch 11/20
    4s - loss: 1.5758 - acc: 0.5248 - val_loss: 1.7419 - val_acc: 0.4898
    Epoch 12/20
    Epoch 00011: early stopping
    4s - loss: 1.5410 - acc: 0.5400 - val_loss: 1.7420 - val_acc: 0.4922
    PREDICT
    6105/6105 [==============================] - 3s     
    PREDICT
    12192/12208 [============================>.] - ETA: 0sFIT PARAMS : 
    {'nb_filters_1': 32, 'nb_filters_2': 32, 'filter_size_2': 3, 'pool_size_1': 4, 'pool_size_2': 2, 'filter_size_1': 6}
    Train on 10988 samples, validate on 1221 samples
    Epoch 1/20
    8s - loss: 2.4604 - acc: 0.2958 - val_loss: 2.3989 - val_acc: 0.3530
    Epoch 2/20
    4s - loss: 2.2277 - acc: 0.3662 - val_loss: 2.1535 - val_acc: 0.3677
    Epoch 3/20
    4s - loss: 2.0688 - acc: 0.3882 - val_loss: 2.0359 - val_acc: 0.4062
    Epoch 4/20
    4s - loss: 1.9769 - acc: 0.4060 - val_loss: 1.9678 - val_acc: 0.4300
    Epoch 5/20
    4s - loss: 1.9072 - acc: 0.4225 - val_loss: 1.8966 - val_acc: 0.4423
    Epoch 6/20
    4s - loss: 1.8424 - acc: 0.4386 - val_loss: 1.8775 - val_acc: 0.4455
    Epoch 7/20
    4s - loss: 1.7865 - acc: 0.4583 - val_loss: 1.7927 - val_acc: 0.4734
    Epoch 8/20
    4s - loss: 1.7312 - acc: 0.4752 - val_loss: 1.7829 - val_acc: 0.4726
    Epoch 9/20
    5s - loss: 1.6829 - acc: 0.4918 - val_loss: 1.8057 - val_acc: 0.4709
    Epoch 10/20
    5s - loss: 1.6396 - acc: 0.4991 - val_loss: 1.7391 - val_acc: 0.4840
    Epoch 11/20
    5s - loss: 1.5997 - acc: 0.5137 - val_loss: 1.7038 - val_acc: 0.4857
    Epoch 12/20
    4s - loss: 1.5678 - acc: 0.5219 - val_loss: 1.6662 - val_acc: 0.5070
    Epoch 13/20
    4s - loss: 1.5308 - acc: 0.5359 - val_loss: 1.7566 - val_acc: 0.4816
    Epoch 14/20
    Epoch 00013: early stopping
    4s - loss: 1.4983 - acc: 0.5403 - val_loss: 1.7164 - val_acc: 0.4808
    PREDICT
    6104/6104 [==============================] - 3s     
    PREDICT
    12192/12209 [============================>.] - ETA: 0sFIT PARAMS : 
    {'nb_filters_1': 32, 'nb_filters_2': 32, 'filter_size_2': 3, 'pool_size_1': 4, 'pool_size_2': 2, 'filter_size_1': 6}
    Train on 10988 samples, validate on 1221 samples
    Epoch 1/20
    8s - loss: 2.4468 - acc: 0.3070 - val_loss: 2.2646 - val_acc: 0.3784
    Epoch 2/20
    4s - loss: 2.2097 - acc: 0.3711 - val_loss: 2.1599 - val_acc: 0.3923
    Epoch 3/20
    4s - loss: 2.1094 - acc: 0.3805 - val_loss: 2.1042 - val_acc: 0.4005
    Epoch 4/20
    4s - loss: 1.9995 - acc: 0.4033 - val_loss: 2.0257 - val_acc: 0.4161
    Epoch 5/20
    4s - loss: 1.9062 - acc: 0.4251 - val_loss: 1.9103 - val_acc: 0.4382
    Epoch 6/20
    4s - loss: 1.8289 - acc: 0.4432 - val_loss: 1.9078 - val_acc: 0.4455
    Epoch 7/20
    4s - loss: 1.7722 - acc: 0.4664 - val_loss: 1.8449 - val_acc: 0.4496
    Epoch 8/20
    4s - loss: 1.7200 - acc: 0.4829 - val_loss: 1.8013 - val_acc: 0.4767
    Epoch 9/20
    4s - loss: 1.6750 - acc: 0.4935 - val_loss: 1.8192 - val_acc: 0.4726
    Epoch 10/20
    4s - loss: 1.6377 - acc: 0.5026 - val_loss: 1.7978 - val_acc: 0.4865
    Epoch 11/20
    4s - loss: 1.5952 - acc: 0.5132 - val_loss: 1.7725 - val_acc: 0.4824
    Epoch 12/20
    4s - loss: 1.5634 - acc: 0.5224 - val_loss: 1.7628 - val_acc: 0.4922
    Epoch 13/20
    4s - loss: 1.5303 - acc: 0.5349 - val_loss: 1.7263 - val_acc: 0.4873
    Epoch 14/20
    4s - loss: 1.4981 - acc: 0.5432 - val_loss: 1.7041 - val_acc: 0.4988
    Epoch 15/20
    4s - loss: 1.4683 - acc: 0.5518 - val_loss: 1.7761 - val_acc: 0.4799
    Epoch 16/20
    Epoch 00015: early stopping
    4s - loss: 1.4394 - acc: 0.5613 - val_loss: 1.7278 - val_acc: 0.4971
    PREDICT
    6104/6104 [==============================] - 3s     
    PREDICT
    12192/12209 [============================>.] - ETA: 0sFIT PARAMS : 
    {'nb_filters_1': 32, 'nb_filters_2': 32, 'filter_size_2': 3, 'pool_size_1': 4, 'pool_size_2': 4, 'filter_size_1': 6}
    Train on 10987 samples, validate on 1221 samples
    Epoch 1/20
    8s - loss: 2.4788 - acc: 0.2855 - val_loss: 2.4066 - val_acc: 0.2957
    Epoch 2/20
    4s - loss: 2.3289 - acc: 0.3428 - val_loss: 2.2208 - val_acc: 0.3661
    Epoch 3/20
    4s - loss: 2.1699 - acc: 0.3783 - val_loss: 2.1287 - val_acc: 0.3841
    Epoch 4/20
    4s - loss: 2.0712 - acc: 0.3954 - val_loss: 2.0322 - val_acc: 0.3939
    Epoch 5/20
    4s - loss: 2.0025 - acc: 0.4084 - val_loss: 1.9790 - val_acc: 0.4169
    Epoch 6/20
    4s - loss: 1.9401 - acc: 0.4188 - val_loss: 1.9415 - val_acc: 0.4185
    Epoch 7/20
    4s - loss: 1.8886 - acc: 0.4296 - val_loss: 1.9266 - val_acc: 0.4341
    Epoch 8/20
    4s - loss: 1.8441 - acc: 0.4410 - val_loss: 1.8696 - val_acc: 0.4382
    Epoch 9/20
    4s - loss: 1.8037 - acc: 0.4594 - val_loss: 1.8879 - val_acc: 0.4373
    Epoch 10/20
    Epoch 00009: early stopping
    4s - loss: 1.7654 - acc: 0.4651 - val_loss: 1.8760 - val_acc: 0.4406
    PREDICT
    6105/6105 [==============================] - 3s     
    PREDICT
    12192/12208 [============================>.] - ETA: 0sFIT PARAMS : 
    {'nb_filters_1': 32, 'nb_filters_2': 32, 'filter_size_2': 3, 'pool_size_1': 4, 'pool_size_2': 4, 'filter_size_1': 6}
    Train on 10988 samples, validate on 1221 samples
    Epoch 1/20
    8s - loss: 2.4616 - acc: 0.2950 - val_loss: 2.3408 - val_acc: 0.3612
    Epoch 2/20
    4s - loss: 2.2561 - acc: 0.3656 - val_loss: 2.1642 - val_acc: 0.3743
    Epoch 3/20
    4s - loss: 2.1252 - acc: 0.3851 - val_loss: 2.0844 - val_acc: 0.3956
    Epoch 4/20
    4s - loss: 2.0371 - acc: 0.4003 - val_loss: 2.0225 - val_acc: 0.3980
    Epoch 5/20
    4s - loss: 1.9767 - acc: 0.4126 - val_loss: 1.9773 - val_acc: 0.4251
    Epoch 6/20
    4s - loss: 1.9277 - acc: 0.4205 - val_loss: 1.9433 - val_acc: 0.4234
    Epoch 7/20
    4s - loss: 1.8871 - acc: 0.4297 - val_loss: 1.9133 - val_acc: 0.4251
    Epoch 8/20
    4s - loss: 1.8505 - acc: 0.4394 - val_loss: 1.8348 - val_acc: 0.4455
    Epoch 9/20
    4s - loss: 1.8154 - acc: 0.4515 - val_loss: 1.8164 - val_acc: 0.4578
    Epoch 10/20
    4s - loss: 1.7821 - acc: 0.4597 - val_loss: 1.8128 - val_acc: 0.4660
    Epoch 11/20
    4s - loss: 1.7513 - acc: 0.4675 - val_loss: 1.7829 - val_acc: 0.4676
    Epoch 12/20
    4s - loss: 1.7165 - acc: 0.4751 - val_loss: 1.7262 - val_acc: 0.4783
    Epoch 13/20
    4s - loss: 1.6867 - acc: 0.4890 - val_loss: 1.8427 - val_acc: 0.4578
    Epoch 14/20
    4s - loss: 1.6631 - acc: 0.4921 - val_loss: 1.7006 - val_acc: 0.4873
    Epoch 15/20
    4s - loss: 1.6330 - acc: 0.5059 - val_loss: 1.6900 - val_acc: 0.4799
    Epoch 16/20
    4s - loss: 1.6074 - acc: 0.5104 - val_loss: 1.6497 - val_acc: 0.5070
    Epoch 17/20
    4s - loss: 1.5841 - acc: 0.5186 - val_loss: 1.6380 - val_acc: 0.5012
    Epoch 18/20
    4s - loss: 1.5636 - acc: 0.5259 - val_loss: 1.6171 - val_acc: 0.5086
    Epoch 19/20
    4s - loss: 1.5406 - acc: 0.5299 - val_loss: 1.6205 - val_acc: 0.5102
    Epoch 20/20
    Epoch 00019: early stopping
    4s - loss: 1.5217 - acc: 0.5386 - val_loss: 1.6214 - val_acc: 0.5176
    PREDICT
    6104/6104 [==============================] - 3s     
    PREDICT
    12192/12209 [============================>.] - ETA: 0sFIT PARAMS : 
    {'nb_filters_1': 32, 'nb_filters_2': 32, 'filter_size_2': 3, 'pool_size_1': 4, 'pool_size_2': 4, 'filter_size_1': 6}
    Train on 10988 samples, validate on 1221 samples
    Epoch 1/20
    8s - loss: 2.4811 - acc: 0.2859 - val_loss: 2.3992 - val_acc: 0.3047
    Epoch 2/20
    4s - loss: 2.3161 - acc: 0.3508 - val_loss: 2.2270 - val_acc: 0.3841
    Epoch 3/20
    4s - loss: 2.1984 - acc: 0.3719 - val_loss: 2.1605 - val_acc: 0.3948
    Epoch 4/20
    4s - loss: 2.1108 - acc: 0.3871 - val_loss: 2.0809 - val_acc: 0.4013
    Epoch 5/20
    4s - loss: 2.0404 - acc: 0.3973 - val_loss: 2.0245 - val_acc: 0.4144
    Epoch 6/20
    4s - loss: 1.9850 - acc: 0.4088 - val_loss: 1.9981 - val_acc: 0.4242
    Epoch 7/20
    4s - loss: 1.9389 - acc: 0.4149 - val_loss: 1.9423 - val_acc: 0.4242
    Epoch 8/20
    4s - loss: 1.8965 - acc: 0.4264 - val_loss: 1.9141 - val_acc: 0.4365
    Epoch 9/20
    4s - loss: 1.8590 - acc: 0.4355 - val_loss: 1.8908 - val_acc: 0.4333
    Epoch 10/20
    4s - loss: 1.8232 - acc: 0.4423 - val_loss: 1.8530 - val_acc: 0.4455
    Epoch 11/20
    4s - loss: 1.7898 - acc: 0.4544 - val_loss: 1.8872 - val_acc: 0.4406
    Epoch 12/20
    4s - loss: 1.7592 - acc: 0.4651 - val_loss: 1.8190 - val_acc: 0.4701
    Epoch 13/20
    4s - loss: 1.7278 - acc: 0.4741 - val_loss: 1.7854 - val_acc: 0.4693
    Epoch 14/20
    4s - loss: 1.6967 - acc: 0.4813 - val_loss: 1.7500 - val_acc: 0.4791
    Epoch 15/20
    4s - loss: 1.6695 - acc: 0.4916 - val_loss: 1.8538 - val_acc: 0.4545
    Epoch 16/20
    4s - loss: 1.6436 - acc: 0.4995 - val_loss: 1.7440 - val_acc: 0.4840
    Epoch 17/20
    4s - loss: 1.6158 - acc: 0.5114 - val_loss: 1.7288 - val_acc: 0.4914
    Epoch 18/20
    4s - loss: 1.5889 - acc: 0.5177 - val_loss: 1.7192 - val_acc: 0.5070
    Epoch 19/20
    4s - loss: 1.5640 - acc: 0.5245 - val_loss: 1.6786 - val_acc: 0.5127
    Epoch 20/20
    4s - loss: 1.5406 - acc: 0.5311 - val_loss: 1.6547 - val_acc: 0.5233
    PREDICT
    6104/6104 [==============================] - 3s     
    PREDICT
    12192/12209 [============================>.] - ETA: 0sFIT PARAMS : 
    {'nb_filters_1': 32, 'nb_filters_2': 64, 'filter_size_2': 3, 'pool_size_1': 2, 'pool_size_2': 2, 'filter_size_1': 6}
    Train on 10987 samples, validate on 1221 samples
    Epoch 1/20
    11s - loss: 2.3554 - acc: 0.3349 - val_loss: 2.1486 - val_acc: 0.3866
    Epoch 2/20
    8s - loss: 2.0455 - acc: 0.3973 - val_loss: 1.9599 - val_acc: 0.4365
    Epoch 3/20
    8s - loss: 1.8675 - acc: 0.4427 - val_loss: 1.9321 - val_acc: 0.4259
    Epoch 4/20
    8s - loss: 1.7239 - acc: 0.4757 - val_loss: 1.8659 - val_acc: 0.4431
    Epoch 5/20
    8s - loss: 1.6117 - acc: 0.5090 - val_loss: 1.7618 - val_acc: 0.4775
    Epoch 6/20
    8s - loss: 1.5129 - acc: 0.5414 - val_loss: 1.7469 - val_acc: 0.4717
    Epoch 7/20
    8s - loss: 1.4268 - acc: 0.5648 - val_loss: 1.7036 - val_acc: 0.4939
    Epoch 8/20
    8s - loss: 1.3470 - acc: 0.5853 - val_loss: 1.7427 - val_acc: 0.4947
    Epoch 9/20
    Epoch 00008: early stopping
    8s - loss: 1.2696 - acc: 0.6102 - val_loss: 1.8818 - val_acc: 0.4898
    PREDICT
    6105/6105 [==============================] - 4s     
    PREDICT
    12192/12208 [============================>.] - ETA: 0sFIT PARAMS : 
    {'nb_filters_1': 32, 'nb_filters_2': 64, 'filter_size_2': 3, 'pool_size_1': 2, 'pool_size_2': 2, 'filter_size_1': 6}
    Train on 10988 samples, validate on 1221 samples
    Epoch 1/20
    11s - loss: 2.3646 - acc: 0.3284 - val_loss: 2.1948 - val_acc: 0.3604
    Epoch 2/20
    8s - loss: 2.0464 - acc: 0.3955 - val_loss: 2.0322 - val_acc: 0.4087
    Epoch 3/20
    8s - loss: 1.8811 - acc: 0.4305 - val_loss: 1.8752 - val_acc: 0.4464
    Epoch 4/20
    8s - loss: 1.7549 - acc: 0.4667 - val_loss: 1.8289 - val_acc: 0.4734
    Epoch 5/20
    8s - loss: 1.6408 - acc: 0.4992 - val_loss: 2.1435 - val_acc: 0.3579
    Epoch 6/20
    8s - loss: 1.5432 - acc: 0.5288 - val_loss: 1.7280 - val_acc: 0.4889
    Epoch 7/20
    8s - loss: 1.4517 - acc: 0.5531 - val_loss: 1.7530 - val_acc: 0.4922
    Epoch 8/20
    Epoch 00007: early stopping
    8s - loss: 1.3697 - acc: 0.5784 - val_loss: 1.7425 - val_acc: 0.4889
    PREDICT
    6104/6104 [==============================] - 4s     
    PREDICT
    12192/12209 [============================>.] - ETA: 0sFIT PARAMS : 
    {'nb_filters_1': 32, 'nb_filters_2': 64, 'filter_size_2': 3, 'pool_size_1': 2, 'pool_size_2': 2, 'filter_size_1': 6}
    Train on 10988 samples, validate on 1221 samples
    Epoch 1/20
    11s - loss: 2.3708 - acc: 0.3245 - val_loss: 2.1650 - val_acc: 0.3849
    Epoch 2/20
    8s - loss: 2.0554 - acc: 0.3951 - val_loss: 2.0506 - val_acc: 0.3989
    Epoch 3/20
    8s - loss: 1.8963 - acc: 0.4319 - val_loss: 1.9415 - val_acc: 0.4333
    Epoch 4/20
    8s - loss: 1.7784 - acc: 0.4590 - val_loss: 1.9054 - val_acc: 0.4406
    Epoch 5/20
    8s - loss: 1.6568 - acc: 0.4905 - val_loss: 1.9053 - val_acc: 0.4545
    Epoch 6/20
    8s - loss: 1.5552 - acc: 0.5252 - val_loss: 1.9038 - val_acc: 0.4193
    Epoch 7/20
    8s - loss: 1.4627 - acc: 0.5497 - val_loss: 1.8145 - val_acc: 0.4717
    Epoch 8/20
    8s - loss: 1.3648 - acc: 0.5829 - val_loss: 1.7497 - val_acc: 0.4988
    Epoch 9/20
    8s - loss: 1.2787 - acc: 0.6068 - val_loss: 1.7365 - val_acc: 0.4996
    Epoch 10/20
    8s - loss: 1.1876 - acc: 0.6371 - val_loss: 1.8025 - val_acc: 0.4734
    Epoch 11/20
    Epoch 00010: early stopping
    8s - loss: 1.0971 - acc: 0.6580 - val_loss: 1.9009 - val_acc: 0.4726
    PREDICT
    6080/6104 [============================>.] - ETA: 0sPREDICT
    12209/12209 [==============================] - 3s     
    FIT PARAMS : 
    {'nb_filters_1': 32, 'nb_filters_2': 64, 'filter_size_2': 3, 'pool_size_1': 2, 'pool_size_2': 4, 'filter_size_1': 6}
    Train on 10987 samples, validate on 1221 samples
    Epoch 1/20
    10s - loss: 2.4024 - acc: 0.3233 - val_loss: 2.2374 - val_acc: 0.3735
    Epoch 2/20
    7s - loss: 2.1430 - acc: 0.3843 - val_loss: 2.0892 - val_acc: 0.3857
    Epoch 3/20
    7s - loss: 1.9896 - acc: 0.4129 - val_loss: 1.9676 - val_acc: 0.4324
    Epoch 4/20
    7s - loss: 1.8891 - acc: 0.4321 - val_loss: 1.9208 - val_acc: 0.4292
    Epoch 5/20
    7s - loss: 1.8164 - acc: 0.4504 - val_loss: 1.8883 - val_acc: 0.4357
    Epoch 6/20
    7s - loss: 1.7507 - acc: 0.4702 - val_loss: 1.8663 - val_acc: 0.4496
    Epoch 7/20
    7s - loss: 1.6972 - acc: 0.4823 - val_loss: 1.8627 - val_acc: 0.4603
    Epoch 8/20
    7s - loss: 1.6424 - acc: 0.4958 - val_loss: 1.8001 - val_acc: 0.4660
    Epoch 9/20
    7s - loss: 1.5919 - acc: 0.5137 - val_loss: 1.8829 - val_acc: 0.4545
    Epoch 10/20
    7s - loss: 1.5378 - acc: 0.5304 - val_loss: 1.7755 - val_acc: 0.4668
    Epoch 11/20
    7s - loss: 1.4895 - acc: 0.5454 - val_loss: 1.7510 - val_acc: 0.4537
    Epoch 12/20
    7s - loss: 1.4425 - acc: 0.5599 - val_loss: 1.6830 - val_acc: 0.4922
    Epoch 13/20
    7s - loss: 1.3987 - acc: 0.5762 - val_loss: 1.7789 - val_acc: 0.4701
    Epoch 14/20
    Epoch 00013: early stopping
    7s - loss: 1.3511 - acc: 0.5837 - val_loss: 1.7298 - val_acc: 0.4783
    PREDICT
    6105/6105 [==============================] - 4s     
    PREDICT
    12192/12208 [============================>.] - ETA: 0sFIT PARAMS : 
    {'nb_filters_1': 32, 'nb_filters_2': 64, 'filter_size_2': 3, 'pool_size_1': 2, 'pool_size_2': 4, 'filter_size_1': 6}
    Train on 10988 samples, validate on 1221 samples
    Epoch 1/20
    10s - loss: 2.4344 - acc: 0.3088 - val_loss: 2.2417 - val_acc: 0.3653
    Epoch 2/20
    7s - loss: 2.1345 - acc: 0.3796 - val_loss: 2.0799 - val_acc: 0.4144
    Epoch 3/20
    7s - loss: 1.9843 - acc: 0.4039 - val_loss: 2.0144 - val_acc: 0.3915
    Epoch 4/20
    7s - loss: 1.9000 - acc: 0.4256 - val_loss: 1.9455 - val_acc: 0.4169
    Epoch 5/20
    7s - loss: 1.8335 - acc: 0.4430 - val_loss: 1.9167 - val_acc: 0.4341
    Epoch 6/20
    7s - loss: 1.7712 - acc: 0.4610 - val_loss: 1.8171 - val_acc: 0.4537
    Epoch 7/20
    7s - loss: 1.7080 - acc: 0.4803 - val_loss: 1.7715 - val_acc: 0.4824
    Epoch 8/20
    7s - loss: 1.6471 - acc: 0.4924 - val_loss: 1.7886 - val_acc: 0.4717
    Epoch 9/20
    7s - loss: 1.5851 - acc: 0.5168 - val_loss: 1.7635 - val_acc: 0.4791
    Epoch 10/20
    7s - loss: 1.5279 - acc: 0.5291 - val_loss: 1.7387 - val_acc: 0.4791
    Epoch 11/20
    7s - loss: 1.4684 - acc: 0.5521 - val_loss: 1.7737 - val_acc: 0.4406
    Epoch 12/20
    7s - loss: 1.4148 - acc: 0.5634 - val_loss: 1.6828 - val_acc: 0.4898
    Epoch 13/20
    7s - loss: 1.3650 - acc: 0.5820 - val_loss: 1.6356 - val_acc: 0.5119
    Epoch 14/20
    7s - loss: 1.3169 - acc: 0.5936 - val_loss: 1.6930 - val_acc: 0.4873
    Epoch 15/20
    Epoch 00014: early stopping
    7s - loss: 1.2705 - acc: 0.6086 - val_loss: 1.6389 - val_acc: 0.5020
    PREDICT
    6104/6104 [==============================] - 4s     
    PREDICT
    12192/12209 [============================>.] - ETA: 0sFIT PARAMS : 
    {'nb_filters_1': 32, 'nb_filters_2': 64, 'filter_size_2': 3, 'pool_size_1': 2, 'pool_size_2': 4, 'filter_size_1': 6}
    Train on 10988 samples, validate on 1221 samples
    Epoch 1/20
    11s - loss: 2.4357 - acc: 0.3062 - val_loss: 2.2412 - val_acc: 0.3874
    Epoch 2/20
    7s - loss: 2.1891 - acc: 0.3716 - val_loss: 2.0972 - val_acc: 0.3939
    Epoch 3/20
    7s - loss: 2.0377 - acc: 0.3910 - val_loss: 1.9982 - val_acc: 0.4144
    Epoch 4/20
    7s - loss: 1.9198 - acc: 0.4185 - val_loss: 1.9330 - val_acc: 0.4251
    Epoch 5/20
    7s - loss: 1.8384 - acc: 0.4405 - val_loss: 1.8703 - val_acc: 0.4693
    Epoch 6/20
    7s - loss: 1.7730 - acc: 0.4570 - val_loss: 1.8219 - val_acc: 0.4750
    Epoch 7/20
    7s - loss: 1.7092 - acc: 0.4749 - val_loss: 1.8347 - val_acc: 0.4521
    Epoch 8/20
    7s - loss: 1.6535 - acc: 0.4933 - val_loss: 1.7558 - val_acc: 0.4939
    Epoch 9/20
    7s - loss: 1.5954 - acc: 0.5111 - val_loss: 1.7461 - val_acc: 0.4980
    Epoch 10/20
    7s - loss: 1.5424 - acc: 0.5315 - val_loss: 1.6755 - val_acc: 0.5184
    Epoch 11/20
    7s - loss: 1.4864 - acc: 0.5446 - val_loss: 1.7273 - val_acc: 0.5012
    Epoch 12/20
    Epoch 00011: early stopping
    7s - loss: 1.4363 - acc: 0.5581 - val_loss: 1.7190 - val_acc: 0.5029
    PREDICT
    6104/6104 [==============================] - 4s     
    PREDICT
    12192/12209 [============================>.] - ETA: 0sFIT PARAMS : 
    {'nb_filters_1': 32, 'nb_filters_2': 64, 'filter_size_2': 3, 'pool_size_1': 4, 'pool_size_2': 2, 'filter_size_1': 6}
    Train on 10987 samples, validate on 1221 samples
    Epoch 1/20
    8s - loss: 2.4217 - acc: 0.3138 - val_loss: 2.2474 - val_acc: 0.3669
    Epoch 2/20
    5s - loss: 2.1598 - acc: 0.3809 - val_loss: 2.1225 - val_acc: 0.3841
    Epoch 3/20
    5s - loss: 2.0089 - acc: 0.4071 - val_loss: 2.0218 - val_acc: 0.4177
    Epoch 4/20
    5s - loss: 1.8922 - acc: 0.4340 - val_loss: 1.9610 - val_acc: 0.4259
    Epoch 5/20
    5s - loss: 1.8048 - acc: 0.4572 - val_loss: 1.8330 - val_acc: 0.4701
    Epoch 6/20
    5s - loss: 1.7245 - acc: 0.4848 - val_loss: 1.8311 - val_acc: 0.4668
    Epoch 7/20
    5s - loss: 1.6616 - acc: 0.4965 - val_loss: 1.7513 - val_acc: 0.4791
    Epoch 8/20
    5s - loss: 1.6072 - acc: 0.5161 - val_loss: 1.7830 - val_acc: 0.4742
    Epoch 9/20
    5s - loss: 1.5549 - acc: 0.5297 - val_loss: 1.7059 - val_acc: 0.4947
    Epoch 10/20
    5s - loss: 1.5068 - acc: 0.5390 - val_loss: 1.6767 - val_acc: 0.5094
    Epoch 11/20
    5s - loss: 1.4619 - acc: 0.5564 - val_loss: 1.6543 - val_acc: 0.5242
    Epoch 12/20
    5s - loss: 1.4177 - acc: 0.5672 - val_loss: 1.6988 - val_acc: 0.4734
    Epoch 13/20
    Epoch 00012: early stopping
    5s - loss: 1.3804 - acc: 0.5815 - val_loss: 1.7461 - val_acc: 0.4889
    PREDICT
    6105/6105 [==============================] - 4s     
    PREDICT
    12192/12208 [============================>.] - ETA: 0sFIT PARAMS : 
    {'nb_filters_1': 32, 'nb_filters_2': 64, 'filter_size_2': 3, 'pool_size_1': 4, 'pool_size_2': 2, 'filter_size_1': 6}
    Train on 10988 samples, validate on 1221 samples
    Epoch 1/20
    9s - loss: 2.4312 - acc: 0.3092 - val_loss: 2.2451 - val_acc: 0.3587
    Epoch 2/20
    5s - loss: 2.1554 - acc: 0.3766 - val_loss: 2.0780 - val_acc: 0.3972
    Epoch 3/20
    5s - loss: 2.0112 - acc: 0.4000 - val_loss: 2.0140 - val_acc: 0.4087
    Epoch 4/20
    5s - loss: 1.9168 - acc: 0.4204 - val_loss: 1.9772 - val_acc: 0.4431
    Epoch 5/20
    5s - loss: 1.8357 - acc: 0.4471 - val_loss: 1.8696 - val_acc: 0.4496
    Epoch 6/20
    5s - loss: 1.7595 - acc: 0.4667 - val_loss: 1.7768 - val_acc: 0.4808
    Epoch 7/20
    5s - loss: 1.6944 - acc: 0.4868 - val_loss: 1.7342 - val_acc: 0.4832
    Epoch 8/20
    5s - loss: 1.6362 - acc: 0.4967 - val_loss: 1.7240 - val_acc: 0.5061
    Epoch 9/20
    5s - loss: 1.5885 - acc: 0.5176 - val_loss: 1.7384 - val_acc: 0.4783
    Epoch 10/20
    5s - loss: 1.5387 - acc: 0.5263 - val_loss: 1.6794 - val_acc: 0.5037
    Epoch 11/20
    5s - loss: 1.4938 - acc: 0.5427 - val_loss: 1.6461 - val_acc: 0.5135
    Epoch 12/20
    5s - loss: 1.4502 - acc: 0.5551 - val_loss: 1.6581 - val_acc: 0.5152
    Epoch 13/20
    5s - loss: 1.4108 - acc: 0.5633 - val_loss: 1.6347 - val_acc: 0.4939
    Epoch 14/20
    5s - loss: 1.3689 - acc: 0.5780 - val_loss: 1.6250 - val_acc: 0.5078
    Epoch 15/20
    5s - loss: 1.3327 - acc: 0.5892 - val_loss: 1.6307 - val_acc: 0.5160
    Epoch 16/20
    5s - loss: 1.2907 - acc: 0.6056 - val_loss: 1.5995 - val_acc: 0.5242
    Epoch 17/20
    5s - loss: 1.2529 - acc: 0.6127 - val_loss: 1.5889 - val_acc: 0.5332
    Epoch 18/20
    5s - loss: 1.2195 - acc: 0.6217 - val_loss: 1.6026 - val_acc: 0.5307
    Epoch 19/20
    Epoch 00018: early stopping
    5s - loss: 1.1834 - acc: 0.6365 - val_loss: 1.6614 - val_acc: 0.5012
    PREDICT
    6104/6104 [==============================] - 4s     
    PREDICT
    12192/12209 [============================>.] - ETA: 0sFIT PARAMS : 
    {'nb_filters_1': 32, 'nb_filters_2': 64, 'filter_size_2': 3, 'pool_size_1': 4, 'pool_size_2': 2, 'filter_size_1': 6}
    Train on 10988 samples, validate on 1221 samples
    Epoch 1/20
    8s - loss: 2.4454 - acc: 0.3057 - val_loss: 2.2546 - val_acc: 0.3866
    Epoch 2/20
    5s - loss: 2.1667 - acc: 0.3776 - val_loss: 2.0615 - val_acc: 0.4136
    Epoch 3/20
    5s - loss: 2.0034 - acc: 0.4020 - val_loss: 2.0730 - val_acc: 0.4095
    Epoch 4/20
    5s - loss: 1.8988 - acc: 0.4322 - val_loss: 1.9221 - val_acc: 0.4357
    Epoch 5/20
    5s - loss: 1.8159 - acc: 0.4515 - val_loss: 1.8788 - val_acc: 0.4513
    Epoch 6/20
    5s - loss: 1.7457 - acc: 0.4722 - val_loss: 1.7843 - val_acc: 0.4701
    Epoch 7/20
    5s - loss: 1.6881 - acc: 0.4874 - val_loss: 1.7921 - val_acc: 0.4685
    Epoch 8/20
    5s - loss: 1.6328 - acc: 0.5059 - val_loss: 1.7345 - val_acc: 0.4996
    Epoch 9/20
    5s - loss: 1.5826 - acc: 0.5162 - val_loss: 1.7176 - val_acc: 0.4988
    Epoch 10/20
    5s - loss: 1.5400 - acc: 0.5325 - val_loss: 1.7086 - val_acc: 0.4947
    Epoch 11/20
    5s - loss: 1.4949 - acc: 0.5440 - val_loss: 1.6838 - val_acc: 0.5037
    Epoch 12/20
    5s - loss: 1.4565 - acc: 0.5554 - val_loss: 1.6887 - val_acc: 0.5176
    Epoch 13/20
    5s - loss: 1.4099 - acc: 0.5700 - val_loss: 1.6485 - val_acc: 0.5143
    Epoch 14/20
    5s - loss: 1.3759 - acc: 0.5767 - val_loss: 1.6604 - val_acc: 0.5070
    Epoch 15/20
    Epoch 00014: early stopping
    5s - loss: 1.3351 - acc: 0.5916 - val_loss: 1.6505 - val_acc: 0.5012
    PREDICT
    6104/6104 [==============================] - 4s     
    PREDICT
    12192/12209 [============================>.] - ETA: 0sFIT PARAMS : 
    {'nb_filters_1': 32, 'nb_filters_2': 64, 'filter_size_2': 3, 'pool_size_1': 4, 'pool_size_2': 4, 'filter_size_1': 6}
    Train on 10987 samples, validate on 1221 samples
    Epoch 1/20
    9s - loss: 2.4633 - acc: 0.2954 - val_loss: 2.3463 - val_acc: 0.3579
    Epoch 2/20
    5s - loss: 2.2561 - acc: 0.3698 - val_loss: 2.2162 - val_acc: 0.3620
    Epoch 3/20
    5s - loss: 2.1541 - acc: 0.3809 - val_loss: 2.0906 - val_acc: 0.3833
    Epoch 4/20
    5s - loss: 2.0514 - acc: 0.3974 - val_loss: 2.0019 - val_acc: 0.4144
    Epoch 5/20
    5s - loss: 1.9697 - acc: 0.4087 - val_loss: 2.0469 - val_acc: 0.3907
    Epoch 6/20
    5s - loss: 1.9038 - acc: 0.4262 - val_loss: 1.9133 - val_acc: 0.4292
    Epoch 7/20
    5s - loss: 1.8466 - acc: 0.4425 - val_loss: 1.8458 - val_acc: 0.4390
    Epoch 8/20
    5s - loss: 1.7969 - acc: 0.4583 - val_loss: 1.8168 - val_acc: 0.4619
    Epoch 9/20
    5s - loss: 1.7496 - acc: 0.4743 - val_loss: 1.8043 - val_acc: 0.4537
    Epoch 10/20
    5s - loss: 1.7073 - acc: 0.4863 - val_loss: 1.7493 - val_acc: 0.4758
    Epoch 11/20
    5s - loss: 1.6687 - acc: 0.4951 - val_loss: 1.7302 - val_acc: 0.4734
    Epoch 12/20
    5s - loss: 1.6340 - acc: 0.5112 - val_loss: 1.7338 - val_acc: 0.4668
    Epoch 13/20
    5s - loss: 1.5981 - acc: 0.5188 - val_loss: 1.6718 - val_acc: 0.5086
    Epoch 14/20
    5s - loss: 1.5617 - acc: 0.5269 - val_loss: 1.7501 - val_acc: 0.4881
    Epoch 15/20
    Epoch 00014: early stopping
    5s - loss: 1.5324 - acc: 0.5377 - val_loss: 1.7124 - val_acc: 0.4889
    PREDICT
    6105/6105 [==============================] - 4s     
    PREDICT
    12192/12208 [============================>.] - ETA: 0sFIT PARAMS : 
    {'nb_filters_1': 32, 'nb_filters_2': 64, 'filter_size_2': 3, 'pool_size_1': 4, 'pool_size_2': 4, 'filter_size_1': 6}
    Train on 10988 samples, validate on 1221 samples
    Epoch 1/20
    9s - loss: 2.4718 - acc: 0.2909 - val_loss: 2.3850 - val_acc: 0.3235
    Epoch 2/20
    5s - loss: 2.2883 - acc: 0.3532 - val_loss: 2.1847 - val_acc: 0.3735
    Epoch 3/20
    5s - loss: 2.1184 - acc: 0.3853 - val_loss: 2.0546 - val_acc: 0.3989
    Epoch 4/20
    5s - loss: 2.0182 - acc: 0.3978 - val_loss: 1.9821 - val_acc: 0.4152
    Epoch 5/20
    5s - loss: 1.9520 - acc: 0.4133 - val_loss: 1.9726 - val_acc: 0.4144
    Epoch 6/20
    5s - loss: 1.8970 - acc: 0.4256 - val_loss: 1.8888 - val_acc: 0.4382
    Epoch 7/20
    5s - loss: 1.8451 - acc: 0.4402 - val_loss: 1.8621 - val_acc: 0.4513
    Epoch 8/20
    5s - loss: 1.8008 - acc: 0.4512 - val_loss: 1.8317 - val_acc: 0.4447
    Epoch 9/20
    5s - loss: 1.7539 - acc: 0.4651 - val_loss: 1.8003 - val_acc: 0.4636
    Epoch 10/20
    5s - loss: 1.7096 - acc: 0.4800 - val_loss: 1.7671 - val_acc: 0.4734
    Epoch 11/20
    5s - loss: 1.6690 - acc: 0.4903 - val_loss: 1.7465 - val_acc: 0.4717
    Epoch 12/20
    5s - loss: 1.6253 - acc: 0.5025 - val_loss: 1.6724 - val_acc: 0.4717
    Epoch 13/20
    5s - loss: 1.5865 - acc: 0.5126 - val_loss: 1.6656 - val_acc: 0.4980
    Epoch 14/20
    5s - loss: 1.5505 - acc: 0.5268 - val_loss: 1.6369 - val_acc: 0.4980
    Epoch 15/20
    5s - loss: 1.5183 - acc: 0.5339 - val_loss: 1.6195 - val_acc: 0.5217
    Epoch 16/20
    5s - loss: 1.4839 - acc: 0.5485 - val_loss: 1.5492 - val_acc: 0.5242
    Epoch 17/20
    5s - loss: 1.4550 - acc: 0.5563 - val_loss: 1.5726 - val_acc: 0.5209
    Epoch 18/20
    Epoch 00017: early stopping
    5s - loss: 1.4243 - acc: 0.5604 - val_loss: 1.5551 - val_acc: 0.5266
    PREDICT
    6104/6104 [==============================] - 4s     
    PREDICT
    12192/12209 [============================>.] - ETA: 0sFIT PARAMS : 
    {'nb_filters_1': 32, 'nb_filters_2': 64, 'filter_size_2': 3, 'pool_size_1': 4, 'pool_size_2': 4, 'filter_size_1': 6}
    Train on 10988 samples, validate on 1221 samples
    Epoch 1/20
    9s - loss: 2.4790 - acc: 0.2874 - val_loss: 2.4003 - val_acc: 0.3456
    Epoch 2/20
    5s - loss: 2.2832 - acc: 0.3587 - val_loss: 2.2296 - val_acc: 0.3890
    Epoch 3/20
    5s - loss: 2.1218 - acc: 0.3848 - val_loss: 2.0466 - val_acc: 0.4120
    Epoch 4/20
    5s - loss: 2.0162 - acc: 0.4022 - val_loss: 2.0108 - val_acc: 0.4111
    Epoch 5/20
    5s - loss: 1.9449 - acc: 0.4158 - val_loss: 1.9498 - val_acc: 0.4161
    Epoch 6/20
    5s - loss: 1.8886 - acc: 0.4308 - val_loss: 1.9283 - val_acc: 0.4390
    Epoch 7/20
    5s - loss: 1.8420 - acc: 0.4422 - val_loss: 1.8927 - val_acc: 0.4414
    Epoch 8/20
    5s - loss: 1.8020 - acc: 0.4522 - val_loss: 1.8405 - val_acc: 0.4521
    Epoch 9/20
    5s - loss: 1.7600 - acc: 0.4652 - val_loss: 1.8250 - val_acc: 0.4521
    Epoch 10/20
    5s - loss: 1.7234 - acc: 0.4723 - val_loss: 1.7756 - val_acc: 0.4758
    Epoch 11/20
    5s - loss: 1.6851 - acc: 0.4881 - val_loss: 1.7776 - val_acc: 0.4644
    Epoch 12/20
    5s - loss: 1.6511 - acc: 0.4924 - val_loss: 1.7489 - val_acc: 0.4848
    Epoch 13/20
    5s - loss: 1.6182 - acc: 0.5053 - val_loss: 1.7316 - val_acc: 0.4922
    Epoch 14/20
    5s - loss: 1.5818 - acc: 0.5168 - val_loss: 1.7539 - val_acc: 0.4857
    Epoch 15/20
    5s - loss: 1.5553 - acc: 0.5236 - val_loss: 1.7239 - val_acc: 0.4693
    Epoch 16/20
    5s - loss: 1.5200 - acc: 0.5353 - val_loss: 1.6749 - val_acc: 0.4947
    Epoch 17/20
    5s - loss: 1.4906 - acc: 0.5446 - val_loss: 1.6517 - val_acc: 0.5168
    Epoch 18/20
    5s - loss: 1.4637 - acc: 0.5531 - val_loss: 1.6738 - val_acc: 0.5012
    Epoch 19/20
    5s - loss: 1.4339 - acc: 0.5600 - val_loss: 1.6152 - val_acc: 0.5217
    Epoch 20/20
    5s - loss: 1.4069 - acc: 0.5705 - val_loss: 1.5931 - val_acc: 0.5315
    PREDICT
    6104/6104 [==============================] - 4s     
    PREDICT
    12192/12209 [============================>.] - ETA: 0sFIT PARAMS : 
    {'nb_filters_1': 64, 'nb_filters_2': 32, 'filter_size_2': 3, 'pool_size_1': 2, 'pool_size_2': 2, 'filter_size_1': 6}
    Train on 10987 samples, validate on 1221 samples
    Epoch 1/20
    13s - loss: 2.3998 - acc: 0.3170 - val_loss: 2.2000 - val_acc: 0.3759
    Epoch 2/20
    10s - loss: 2.0948 - acc: 0.3922 - val_loss: 2.0845 - val_acc: 0.3948
    Epoch 3/20
    10s - loss: 1.9619 - acc: 0.4149 - val_loss: 1.9861 - val_acc: 0.4242
    Epoch 4/20
    10s - loss: 1.8591 - acc: 0.4391 - val_loss: 1.9167 - val_acc: 0.4447
    Epoch 5/20
    10s - loss: 1.7672 - acc: 0.4686 - val_loss: 1.8998 - val_acc: 0.4431
    Epoch 6/20
    10s - loss: 1.6797 - acc: 0.4869 - val_loss: 1.8104 - val_acc: 0.4660
    Epoch 7/20
    10s - loss: 1.5935 - acc: 0.5141 - val_loss: 1.8342 - val_acc: 0.4627
    Epoch 8/20
    10s - loss: 1.5195 - acc: 0.5400 - val_loss: 1.8058 - val_acc: 0.4603
    Epoch 9/20
    10s - loss: 1.4362 - acc: 0.5578 - val_loss: 1.7728 - val_acc: 0.4775
    Epoch 10/20
    10s - loss: 1.3644 - acc: 0.5789 - val_loss: 1.7357 - val_acc: 0.4726
    Epoch 11/20
    10s - loss: 1.2790 - acc: 0.6064 - val_loss: 1.7984 - val_acc: 0.4873
    Epoch 12/20
    Epoch 00011: early stopping
    10s - loss: 1.2040 - acc: 0.6260 - val_loss: 1.8233 - val_acc: 0.4488
    PREDICT
    6105/6105 [==============================] - 5s     
    PREDICT
    11424/12208 [===========================>..] - ETA: 0s10s - loss: 1.9849 - acc: 0.4044 - val_loss: 1.9826 - val_acc: 0.4373
    Epoch 4/20
    10s - loss: 1.8785 - acc: 0.4322 - val_loss: 1.9168 - val_acc: 0.4398
    Epoch 5/20
    10s - loss: 1.7805 - acc: 0.4566 - val_loss: 1.8605 - val_acc: 0.4480
    Epoch 6/20
    10s - loss: 1.6787 - acc: 0.4858 - val_loss: 1.8344 - val_acc: 0.4595
    Epoch 7/20
    10s - loss: 1.5896 - acc: 0.5055 - val_loss: 1.9058 - val_acc: 0.4414
    Epoch 8/20
    Epoch 00007: early stopping
    10s - loss: 1.4944 - acc: 0.5410 - val_loss: 1.8775 - val_acc: 0.4595
    PREDICT
    6104/6104 [==============================] - 5s     
    PREDICT
    11648/12209 [===========================>..] - ETA: 0s10s - loss: 1.9318 - acc: 0.4172 - val_loss: 1.9349 - val_acc: 0.4324
    Epoch 4/20
    10s - loss: 1.8142 - acc: 0.4498 - val_loss: 1.9193 - val_acc: 0.4414
    Epoch 5/20
    10s - loss: 1.7179 - acc: 0.4746 - val_loss: 1.8599 - val_acc: 0.4619
    Epoch 6/20
    10s - loss: 1.6293 - acc: 0.5020 - val_loss: 1.8196 - val_acc: 0.4636
    Epoch 7/20
    10s - loss: 1.5443 - acc: 0.5247 - val_loss: 1.7717 - val_acc: 0.4808
    Epoch 8/20
    10s - loss: 1.4604 - acc: 0.5534 - val_loss: 1.8410 - val_acc: 0.4398
    Epoch 9/20
    Epoch 00008: early stopping
    10s - loss: 1.3829 - acc: 0.5732 - val_loss: 1.7738 - val_acc: 0.4758
    PREDICT
    6080/6104 [============================>.] - ETA: 0sPREDICT
    12192/12209 [============================>.] - ETA: 0sFIT PARAMS : 
    {'nb_filters_1': 64, 'nb_filters_2': 32, 'filter_size_2': 3, 'pool_size_1': 2, 'pool_size_2': 4, 'filter_size_1': 6}
    Train on 10987 samples, validate on 1221 samples
    Epoch 1/20
    13s - loss: 2.4281 - acc: 0.3120 - val_loss: 2.2563 - val_acc: 0.3743
    Epoch 2/20
    10s - loss: 2.1787 - acc: 0.3792 - val_loss: 2.0948 - val_acc: 0.3882
    Epoch 3/20
    10s - loss: 2.0338 - acc: 0.3990 - val_loss: 2.0237 - val_acc: 0.4152
    Epoch 4/20
    10s - loss: 1.9413 - acc: 0.4195 - val_loss: 1.9674 - val_acc: 0.4292
    Epoch 5/20
    10s - loss: 1.8710 - acc: 0.4356 - val_loss: 1.9125 - val_acc: 0.4423
    Epoch 6/20
    10s - loss: 1.8104 - acc: 0.4521 - val_loss: 1.8644 - val_acc: 0.4603
    Epoch 7/20
    10s - loss: 1.7544 - acc: 0.4694 - val_loss: 1.8802 - val_acc: 0.4636
    Epoch 8/20
    10s - loss: 1.6999 - acc: 0.4820 - val_loss: 1.8394 - val_acc: 0.4701
    Epoch 9/20
    10s - loss: 1.6543 - acc: 0.4990 - val_loss: 1.7846 - val_acc: 0.4840
    Epoch 10/20
    10s - loss: 1.6091 - acc: 0.5102 - val_loss: 1.7818 - val_acc: 0.4726
    Epoch 11/20
    10s - loss: 1.5640 - acc: 0.5251 - val_loss: 1.7561 - val_acc: 0.4824
    Epoch 12/20
    10s - loss: 1.5193 - acc: 0.5389 - val_loss: 1.8763 - val_acc: 0.4521
    Epoch 13/20
    Epoch 00012: early stopping
    10s - loss: 1.4816 - acc: 0.5473 - val_loss: 1.7927 - val_acc: 0.4513
    PREDICT
    6080/6105 [============================>.] - ETA: 0sPREDICT
    12192/12208 [============================>.] - ETA: 0sFIT PARAMS : 
    {'nb_filters_1': 64, 'nb_filters_2': 32, 'filter_size_2': 3, 'pool_size_1': 2, 'pool_size_2': 4, 'filter_size_1': 6}
    12192/12209 [============================>.] - ETA: 0sFIT PARAMS : 
    {'nb_filters_1': 32, 'nb_filters_2': 32, 'filter_size_2': 6, 'pool_size_1': 2, 'pool_size_2': 4, 'filter_size_1': 6}
    Train on 10987 samples, validate on 1221 samples
    Epoch 1/20
    13s - loss: 2.4060 - acc: 0.3224 - val_loss: 2.2344 - val_acc: 0.3710
    Epoch 2/20
    8s - loss: 2.1400 - acc: 0.3818 - val_loss: 2.0709 - val_acc: 0.3997
    Epoch 3/20
    8s - loss: 1.9997 - acc: 0.4098 - val_loss: 1.9587 - val_acc: 0.4341
    Epoch 4/20
    8s - loss: 1.9033 - acc: 0.4304 - val_loss: 1.9389 - val_acc: 0.4357
    Epoch 5/20
    8s - loss: 1.8279 - acc: 0.4504 - val_loss: 1.8853 - val_acc: 0.4496
    Epoch 6/20
    8s - loss: 1.7612 - acc: 0.4667 - val_loss: 1.8351 - val_acc: 0.4570
    Epoch 7/20
    8s - loss: 1.6956 - acc: 0.4854 - val_loss: 1.9423 - val_acc: 0.4529
    Epoch 8/20
    Epoch 00007: early stopping
    8s - loss: 1.6368 - acc: 0.5081 - val_loss: 1.9621 - val_acc: 0.4177
    PREDICT
    6080/6105 [============================>.] - ETA: 0sPREDICT
    12192/12208 [============================>.] - ETA: 0sFIT PARAMS : 
    {'nb_filters_1': 32, 'nb_filters_2': 32, 'filter_size_2': 6, 'pool_size_1': 2, 'pool_size_2': 4, 'filter_size_1': 6}
    Train on 10988 samples, validate on 1221 samples
    Epoch 1/20
    13s - loss: 2.4158 - acc: 0.3111 - val_loss: 2.2327 - val_acc: 0.3620
    Epoch 2/20
    8s - loss: 2.1309 - acc: 0.3772 - val_loss: 2.0991 - val_acc: 0.4029
    Epoch 3/20
    8s - loss: 1.9924 - acc: 0.4088 - val_loss: 2.0807 - val_acc: 0.3604
    Epoch 4/20
    8s - loss: 1.8934 - acc: 0.4286 - val_loss: 1.8910 - val_acc: 0.4513
    Epoch 5/20
    8s - loss: 1.8101 - acc: 0.4492 - val_loss: 1.8986 - val_acc: 0.4439
    Epoch 6/20
    8s - loss: 1.7441 - acc: 0.4720 - val_loss: 1.7931 - val_acc: 0.4726
    Epoch 7/20
    8s - loss: 1.6765 - acc: 0.4894 - val_loss: 1.7538 - val_acc: 0.4832
    Epoch 8/20
    8s - loss: 1.6126 - acc: 0.5089 - val_loss: 1.7307 - val_acc: 0.4816
    Epoch 9/20
    8s - loss: 1.5558 - acc: 0.5233 - val_loss: 1.7491 - val_acc: 0.4922
    Epoch 10/20
    8s - loss: 1.5043 - acc: 0.5375 - val_loss: 1.6648 - val_acc: 0.5168
    Epoch 11/20
    8s - loss: 1.4529 - acc: 0.5553 - val_loss: 1.6581 - val_acc: 0.5086
    Epoch 12/20
    8s - loss: 1.4034 - acc: 0.5710 - val_loss: 1.7668 - val_acc: 0.4889
    Epoch 13/20
    9s - loss: 1.3642 - acc: 0.5845 - val_loss: 1.6327 - val_acc: 0.5209
    Epoch 14/20
    9s - loss: 1.3166 - acc: 0.5917 - val_loss: 1.6351 - val_acc: 0.5201
    Epoch 15/20
    8s - loss: 1.2772 - acc: 0.6029 - val_loss: 1.6255 - val_acc: 0.5209
    Epoch 16/20
    8s - loss: 1.2347 - acc: 0.6156 - val_loss: 1.6666 - val_acc: 0.5012
    Epoch 17/20
    Epoch 00016: early stopping
    9s - loss: 1.2018 - acc: 0.6256 - val_loss: 2.0147 - val_acc: 0.4201
    PREDICT
    6080/6104 [============================>.] - ETA: 0sPREDICT
    12192/12209 [============================>.] - ETA: 0sFIT PARAMS : 
    {'nb_filters_1': 32, 'nb_filters_2': 32, 'filter_size_2': 6, 'pool_size_1': 2, 'pool_size_2': 4, 'filter_size_1': 6}
    Train on 10988 samples, validate on 1221 samples
    Epoch 1/20
    13s - loss: 2.4562 - acc: 0.2934 - val_loss: 2.3145 - val_acc: 0.3767
    Epoch 2/20
    8s - loss: 2.2383 - acc: 0.3645 - val_loss: 2.1654 - val_acc: 0.3882
    Epoch 3/20
    8s - loss: 2.1137 - acc: 0.3786 - val_loss: 2.0561 - val_acc: 0.4152
    Epoch 4/20
    8s - loss: 1.9859 - acc: 0.4065 - val_loss: 2.0149 - val_acc: 0.4193
    Epoch 5/20
    8s - loss: 1.8966 - acc: 0.4283 - val_loss: 1.9489 - val_acc: 0.4275
    Epoch 6/20
    8s - loss: 1.8227 - acc: 0.4490 - val_loss: 1.8387 - val_acc: 0.4562
    Epoch 7/20
    8s - loss: 1.7567 - acc: 0.4639 - val_loss: 1.8920 - val_acc: 0.4341
    Epoch 8/20
    8s - loss: 1.6932 - acc: 0.4903 - val_loss: 1.7442 - val_acc: 0.4914
    Epoch 9/20
    8s - loss: 1.6356 - acc: 0.5047 - val_loss: 1.8296 - val_acc: 0.4570
    Epoch 10/20
    8s - loss: 1.5877 - acc: 0.5165 - val_loss: 1.7132 - val_acc: 0.4881
    Epoch 11/20
    8s - loss: 1.5373 - acc: 0.5359 - val_loss: 1.8495 - val_acc: 0.4644
    Epoch 12/20
    Epoch 00011: early stopping
    8s - loss: 1.4960 - acc: 0.5491 - val_loss: 1.7242 - val_acc: 0.4889
    PREDICT
    6080/6104 [============================>.] - ETA: 0sPREDICT
    12192/12209 [============================>.] - ETA: 0sFIT PARAMS : 
    {'nb_filters_1': 32, 'nb_filters_2': 32, 'filter_size_2': 6, 'pool_size_1': 4, 'pool_size_2': 2, 'filter_size_1': 6}
    Train on 10987 samples, validate on 1221 samples
    Epoch 1/20
    10s - loss: 2.4273 - acc: 0.3106 - val_loss: 2.2680 - val_acc: 0.3595
    Epoch 2/20
    5s - loss: 2.1559 - acc: 0.3798 - val_loss: 2.0990 - val_acc: 0.4021
    Epoch 3/20
    5s - loss: 1.9868 - acc: 0.4110 - val_loss: 2.0189 - val_acc: 0.4242
    Epoch 4/20
    5s - loss: 1.8640 - acc: 0.4406 - val_loss: 1.9423 - val_acc: 0.4275
    Epoch 5/20
    5s - loss: 1.7722 - acc: 0.4643 - val_loss: 1.9081 - val_acc: 0.4333
    Epoch 6/20
    5s - loss: 1.7057 - acc: 0.4831 - val_loss: 1.7935 - val_acc: 0.4636
    Epoch 7/20
    5s - loss: 1.6468 - acc: 0.4985 - val_loss: 1.7391 - val_acc: 0.4824
    Epoch 8/20
    5s - loss: 1.5935 - acc: 0.5130 - val_loss: 1.7254 - val_acc: 0.4857
    Epoch 9/20
    5s - loss: 1.5538 - acc: 0.5286 - val_loss: 1.6869 - val_acc: 0.5078
    Epoch 10/20
    5s - loss: 1.5071 - acc: 0.5403 - val_loss: 1.6465 - val_acc: 0.5070
    Epoch 11/20
    5s - loss: 1.4671 - acc: 0.5493 - val_loss: 1.7780 - val_acc: 0.4701
    Epoch 12/20
    Epoch 00011: early stopping
    5s - loss: 1.4277 - acc: 0.5646 - val_loss: 1.6865 - val_acc: 0.5078
    PREDICT
    6105/6105 [==============================] - 4s     
    PREDICT
    12192/12208 [============================>.] - ETA: 0sFIT PARAMS : 
    {'nb_filters_1': 32, 'nb_filters_2': 32, 'filter_size_2': 6, 'pool_size_1': 4, 'pool_size_2': 2, 'filter_size_1': 6}
    Train on 10988 samples, validate on 1221 samples
    Epoch 1/20
    10s - loss: 2.4357 - acc: 0.3112 - val_loss: 2.2632 - val_acc: 0.3612
    Epoch 2/20
    5s - loss: 2.1700 - acc: 0.3754 - val_loss: 2.0641 - val_acc: 0.3898
    Epoch 3/20
    5s - loss: 2.0077 - acc: 0.4005 - val_loss: 1.9749 - val_acc: 0.4201
    Epoch 4/20
    5s - loss: 1.8892 - acc: 0.4286 - val_loss: 1.8935 - val_acc: 0.4496
    Epoch 5/20
    5s - loss: 1.8018 - acc: 0.4548 - val_loss: 1.8406 - val_acc: 0.4447
    Epoch 6/20
    5s - loss: 1.7255 - acc: 0.4772 - val_loss: 1.8475 - val_acc: 0.4373
    Epoch 7/20
    5s - loss: 1.6622 - acc: 0.4942 - val_loss: 1.7186 - val_acc: 0.4906
    Epoch 8/20
    5s - loss: 1.6031 - acc: 0.5141 - val_loss: 1.7538 - val_acc: 0.4889
    Epoch 9/20
    5s - loss: 1.5505 - acc: 0.5269 - val_loss: 1.6928 - val_acc: 0.5012
    Epoch 10/20
    5s - loss: 1.5012 - acc: 0.5376 - val_loss: 1.6767 - val_acc: 0.5020
    Epoch 11/20
    5s - loss: 1.4523 - acc: 0.5548 - val_loss: 1.6350 - val_acc: 0.5192
    Epoch 12/20
    5s - loss: 1.4069 - acc: 0.5643 - val_loss: 1.7669 - val_acc: 0.4701
    Epoch 13/20
    5s - loss: 1.3622 - acc: 0.5825 - val_loss: 1.6294 - val_acc: 0.5201
    Epoch 14/20
    5s - loss: 1.3188 - acc: 0.5943 - val_loss: 1.6405 - val_acc: 0.5176
    Epoch 15/20
    Epoch 00014: early stopping
    5s - loss: 1.2800 - acc: 0.6027 - val_loss: 1.6300 - val_acc: 0.5037
    PREDICT
    6104/6104 [==============================] - 5s     
    PREDICT
    12192/12209 [============================>.] - ETA: 0sFIT PARAMS : 
    {'nb_filters_1': 32, 'nb_filters_2': 32, 'filter_size_2': 6, 'pool_size_1': 4, 'pool_size_2': 2, 'filter_size_1': 6}
    Train on 10988 samples, validate on 1221 samples
    Epoch 1/20
    10s - loss: 2.4615 - acc: 0.2964 - val_loss: 2.3417 - val_acc: 0.3342
    Epoch 2/20
    5s - loss: 2.2075 - acc: 0.3675 - val_loss: 2.1327 - val_acc: 0.3989
    Epoch 3/20
    5s - loss: 2.0415 - acc: 0.3975 - val_loss: 2.0680 - val_acc: 0.4038
    Epoch 4/20
    5s - loss: 1.9258 - acc: 0.4185 - val_loss: 1.9012 - val_acc: 0.4373
    Epoch 5/20
    5s - loss: 1.8398 - acc: 0.4448 - val_loss: 1.8770 - val_acc: 0.4464
    Epoch 6/20
    5s - loss: 1.7633 - acc: 0.4661 - val_loss: 1.8676 - val_acc: 0.4300
    Epoch 7/20
    5s - loss: 1.6994 - acc: 0.4815 - val_loss: 1.8110 - val_acc: 0.4717
    Epoch 8/20
    5s - loss: 1.6412 - acc: 0.5017 - val_loss: 1.7908 - val_acc: 0.4824
    Epoch 9/20
    5s - loss: 1.5890 - acc: 0.5117 - val_loss: 1.7060 - val_acc: 0.4930
    Epoch 10/20
    5s - loss: 1.5420 - acc: 0.5298 - val_loss: 2.1313 - val_acc: 0.3694
    Epoch 11/20
    5s - loss: 1.4938 - acc: 0.5383 - val_loss: 1.6767 - val_acc: 0.5094
    Epoch 12/20
    5s - loss: 1.4540 - acc: 0.5506 - val_loss: 1.6539 - val_acc: 0.5209
    Epoch 13/20
    5s - loss: 1.4080 - acc: 0.5633 - val_loss: 1.7065 - val_acc: 0.4947
    Epoch 14/20
    Epoch 00013: early stopping
    5s - loss: 1.3666 - acc: 0.5811 - val_loss: 1.7790 - val_acc: 0.4767
    PREDICT
    6104/6104 [==============================] - 4s     
    PREDICT
    12192/12209 [============================>.] - ETA: 0sFIT PARAMS : 
    {'nb_filters_1': 32, 'nb_filters_2': 32, 'filter_size_2': 6, 'pool_size_1': 4, 'pool_size_2': 4, 'filter_size_1': 6}
    Train on 10987 samples, validate on 1221 samples
    Epoch 1/20
    10s - loss: 2.4428 - acc: 0.3046 - val_loss: 2.3391 - val_acc: 0.3415
    Epoch 2/20
    5s - loss: 2.2539 - acc: 0.3655 - val_loss: 2.1913 - val_acc: 0.3604
    Epoch 3/20
    5s - loss: 2.1159 - acc: 0.3893 - val_loss: 2.0737 - val_acc: 0.3997
    Epoch 4/20
    5s - loss: 2.0009 - acc: 0.4139 - val_loss: 2.0064 - val_acc: 0.4152
    Epoch 5/20
    5s - loss: 1.9142 - acc: 0.4297 - val_loss: 2.0065 - val_acc: 0.4079
    Epoch 6/20
    5s - loss: 1.8402 - acc: 0.4502 - val_loss: 1.8971 - val_acc: 0.4333
    Epoch 7/20
    5s - loss: 1.7751 - acc: 0.4638 - val_loss: 1.8513 - val_acc: 0.4676
    Epoch 8/20
    5s - loss: 1.7233 - acc: 0.4842 - val_loss: 1.9481 - val_acc: 0.4390
    Epoch 9/20
    5s - loss: 1.6728 - acc: 0.4998 - val_loss: 1.7469 - val_acc: 0.4709
    Epoch 10/20
    5s - loss: 1.6261 - acc: 0.5164 - val_loss: 1.7339 - val_acc: 0.4914
    Epoch 11/20
    5s - loss: 1.5889 - acc: 0.5244 - val_loss: 1.6927 - val_acc: 0.4922
    Epoch 12/20
    5s - loss: 1.5436 - acc: 0.5312 - val_loss: 1.7171 - val_acc: 0.4963
    Epoch 13/20
    5s - loss: 1.5112 - acc: 0.5461 - val_loss: 1.6722 - val_acc: 0.5086
    Epoch 14/20
    5s - loss: 1.4766 - acc: 0.5562 - val_loss: 1.6852 - val_acc: 0.5111
    Epoch 15/20
    Epoch 00014: early stopping
    5s - loss: 1.4411 - acc: 0.5635 - val_loss: 1.6898 - val_acc: 0.4939
    PREDICT
    6105/6105 [==============================] - 5s     
    PREDICT
    12192/12208 [============================>.] - ETA: 0sFIT PARAMS : 
    {'nb_filters_1': 32, 'nb_filters_2': 32, 'filter_size_2': 6, 'pool_size_1': 4, 'pool_size_2': 4, 'filter_size_1': 6}
    Train on 10988 samples, validate on 1221 samples
    Epoch 1/20
    10s - loss: 2.4664 - acc: 0.2958 - val_loss: 2.5602 - val_acc: 0.3219
    Epoch 2/20
    5s - loss: 2.2755 - acc: 0.3589 - val_loss: 2.2030 - val_acc: 0.3645
    Epoch 3/20
    5s - loss: 2.1619 - acc: 0.3734 - val_loss: 2.1381 - val_acc: 0.3767
    Epoch 4/20
    5s - loss: 2.0702 - acc: 0.3914 - val_loss: 2.0469 - val_acc: 0.4038
    Epoch 5/20
    5s - loss: 1.9981 - acc: 0.4056 - val_loss: 2.0093 - val_acc: 0.4062
    Epoch 6/20
    5s - loss: 1.9410 - acc: 0.4177 - val_loss: 1.9604 - val_acc: 0.4242
    Epoch 7/20
    5s - loss: 1.8911 - acc: 0.4311 - val_loss: 1.9158 - val_acc: 0.4275
    Epoch 8/20
    5s - loss: 1.8445 - acc: 0.4454 - val_loss: 1.8894 - val_acc: 0.4365
    Epoch 9/20
    5s - loss: 1.7992 - acc: 0.4537 - val_loss: 1.8550 - val_acc: 0.4488
    Epoch 10/20
    5s - loss: 1.7542 - acc: 0.4694 - val_loss: 1.8806 - val_acc: 0.4480
    Epoch 11/20
    5s - loss: 1.7175 - acc: 0.4782 - val_loss: 1.7997 - val_acc: 0.4562
    Epoch 12/20
    5s - loss: 1.6781 - acc: 0.4883 - val_loss: 1.8396 - val_acc: 0.4636
    Epoch 13/20
    5s - loss: 1.6446 - acc: 0.5009 - val_loss: 1.7876 - val_acc: 0.4652
    Epoch 14/20
    5s - loss: 1.6057 - acc: 0.5133 - val_loss: 1.7926 - val_acc: 0.4578
    Epoch 15/20
    Epoch 00014: early stopping
    5s - loss: 1.5694 - acc: 0.5229 - val_loss: 1.8359 - val_acc: 0.4545
    PREDICT
    6104/6104 [==============================] - 5s     
    PREDICT
    12192/12209 [============================>.] - ETA: 0sFIT PARAMS : 
    {'nb_filters_1': 32, 'nb_filters_2': 32, 'filter_size_2': 6, 'pool_size_1': 4, 'pool_size_2': 4, 'filter_size_1': 6}
    Train on 10988 samples, validate on 1221 samples
    Epoch 1/20
    10s - loss: 2.4665 - acc: 0.2943 - val_loss: 2.3433 - val_acc: 0.3669
    Epoch 2/20
    5s - loss: 2.2703 - acc: 0.3625 - val_loss: 2.1762 - val_acc: 0.3964
    Epoch 3/20
    5s - loss: 2.1212 - acc: 0.3870 - val_loss: 2.1295 - val_acc: 0.3915
    Epoch 4/20
    5s - loss: 2.0171 - acc: 0.4035 - val_loss: 1.9951 - val_acc: 0.4177
    Epoch 5/20
    5s - loss: 1.9425 - acc: 0.4182 - val_loss: 1.9950 - val_acc: 0.4169
    Epoch 6/20
    5s - loss: 1.8782 - acc: 0.4370 - val_loss: 1.9317 - val_acc: 0.4357
    Epoch 7/20
    5s - loss: 1.8207 - acc: 0.4509 - val_loss: 1.8985 - val_acc: 0.4676
    Epoch 8/20
    5s - loss: 1.7670 - acc: 0.4681 - val_loss: 1.8611 - val_acc: 0.4701
    Epoch 9/20
    5s - loss: 1.7164 - acc: 0.4800 - val_loss: 1.8152 - val_acc: 0.4652
    Epoch 10/20
    5s - loss: 1.6694 - acc: 0.4938 - val_loss: 1.8311 - val_acc: 0.4783
    Epoch 11/20
    Epoch 00010: early stopping
    5s - loss: 1.6290 - acc: 0.5086 - val_loss: 1.8258 - val_acc: 0.4644
    PREDICT
    6104/6104 [==============================] - 5s     
    PREDICT
    12192/12209 [============================>.] - ETA: 0sFIT PARAMS : 
    {'nb_filters_1': 32, 'nb_filters_2': 64, 'filter_size_2': 6, 'pool_size_1': 2, 'pool_size_2': 2, 'filter_size_1': 6}
    Train on 10987 samples, validate on 1221 samples
    Epoch 1/20
    16s - loss: 2.3846 - acc: 0.3237 - val_loss: 2.1347 - val_acc: 0.3808
    Epoch 2/20
    12s - loss: 2.0435 - acc: 0.4026 - val_loss: 1.9902 - val_acc: 0.3939
    Epoch 3/20
    12s - loss: 1.8709 - acc: 0.4415 - val_loss: 1.9800 - val_acc: 0.4259
    Epoch 4/20
    12s - loss: 1.7361 - acc: 0.4800 - val_loss: 1.7780 - val_acc: 0.4717
    Epoch 5/20
    12s - loss: 1.6088 - acc: 0.5130 - val_loss: 1.8949 - val_acc: 0.4398
    Epoch 6/20
    Epoch 00005: early stopping
    11s - loss: 1.4953 - acc: 0.5456 - val_loss: 1.8591 - val_acc: 0.4652
    PREDICT
    6105/6105 [==============================] - 6s     
    PREDICT
    12192/12208 [============================>.] - ETA: 0sFIT PARAMS : 
    {'nb_filters_1': 32, 'nb_filters_2': 64, 'filter_size_2': 6, 'pool_size_1': 2, 'pool_size_2': 2, 'filter_size_1': 6}
    Train on 10988 samples, validate on 1221 samples
    Epoch 1/20
    16s - loss: 2.3866 - acc: 0.3273 - val_loss: 2.1987 - val_acc: 0.3792
    Epoch 2/20
    12s - loss: 2.1018 - acc: 0.3822 - val_loss: 2.0433 - val_acc: 0.4111
    Epoch 3/20
    12s - loss: 1.8969 - acc: 0.4288 - val_loss: 1.9923 - val_acc: 0.4365
    Epoch 4/20
    12s - loss: 1.7183 - acc: 0.4841 - val_loss: 1.8455 - val_acc: 0.4676
    Epoch 5/20
    12s - loss: 1.5843 - acc: 0.5153 - val_loss: 1.8113 - val_acc: 0.4505
    Epoch 6/20
    12s - loss: 1.4600 - acc: 0.5485 - val_loss: 1.7275 - val_acc: 0.4873
    Epoch 7/20
    12s - loss: 1.3339 - acc: 0.5857 - val_loss: 1.7582 - val_acc: 0.5053
    Epoch 8/20
    Epoch 00007: early stopping
    12s - loss: 1.2119 - acc: 0.6234 - val_loss: 1.8598 - val_acc: 0.4824
    PREDICT
    6104/6104 [==============================] - 6s     
    PREDICT
    11968/12209 [============================>.] - ETA: 0s12s - loss: 2.0694 - acc: 0.3908 - val_loss: 2.0192 - val_acc: 0.4079
    Epoch 3/20
    12s - loss: 1.8957 - acc: 0.4297 - val_loss: 1.9193 - val_acc: 0.4513
    Epoch 4/20
    12s - loss: 1.7523 - acc: 0.4710 - val_loss: 1.8577 - val_acc: 0.4570
    Epoch 5/20
    12s - loss: 1.6249 - acc: 0.5067 - val_loss: 1.7664 - val_acc: 0.4717
    Epoch 6/20
    12s - loss: 1.5024 - acc: 0.5380 - val_loss: 1.7113 - val_acc: 0.4881
    Epoch 7/20
    12s - loss: 1.3819 - acc: 0.5742 - val_loss: 1.7201 - val_acc: 0.4832
    Epoch 8/20
    Epoch 00007: early stopping
    12s - loss: 1.2635 - acc: 0.6126 - val_loss: 1.8281 - val_acc: 0.4414
    PREDICT
    6104/6104 [==============================] - 6s     
    PREDICT
    12192/12209 [============================>.] - ETA: 0sFIT PARAMS : 
    {'nb_filters_1': 32, 'nb_filters_2': 64, 'filter_size_2': 6, 'pool_size_1': 2, 'pool_size_2': 4, 'filter_size_1': 6}
    Train on 10987 samples, validate on 1221 samples
    Epoch 1/20
    16s - loss: 2.4529 - acc: 0.2993 - val_loss: 2.3449 - val_acc: 0.3186
    Epoch 2/20
    11s - loss: 2.1711 - acc: 0.3776 - val_loss: 2.0941 - val_acc: 0.3669
    Epoch 3/20
    11s - loss: 1.9900 - acc: 0.4086 - val_loss: 1.9565 - val_acc: 0.4275
    Epoch 4/20
    11s - loss: 1.8796 - acc: 0.4326 - val_loss: 1.9619 - val_acc: 0.4275
    Epoch 5/20
    11s - loss: 1.7965 - acc: 0.4571 - val_loss: 1.9472 - val_acc: 0.3972
    Epoch 6/20
    11s - loss: 1.7129 - acc: 0.4798 - val_loss: 1.8008 - val_acc: 0.4808
    Epoch 7/20
    11s - loss: 1.6355 - acc: 0.5028 - val_loss: 1.7848 - val_acc: 0.4840
    Epoch 8/20
    11s - loss: 1.5623 - acc: 0.5262 - val_loss: 1.7393 - val_acc: 0.4873
    Epoch 9/20
    11s - loss: 1.4937 - acc: 0.5448 - val_loss: 2.0096 - val_acc: 0.3964
    Epoch 10/20
    11s - loss: 1.4321 - acc: 0.5633 - val_loss: 1.6775 - val_acc: 0.5004
    Epoch 11/20
    11s - loss: 1.3652 - acc: 0.5799 - val_loss: 1.7715 - val_acc: 0.4988
    Epoch 12/20
    Epoch 00011: early stopping
    11s - loss: 1.3075 - acc: 0.6007 - val_loss: 1.6934 - val_acc: 0.4980
    PREDICT
    6105/6105 [==============================] - 6s     
    PREDICT
    12192/12208 [============================>.] - ETA: 0sFIT PARAMS : 
    {'nb_filters_1': 32, 'nb_filters_2': 64, 'filter_size_2': 6, 'pool_size_1': 2, 'pool_size_2': 4, 'filter_size_1': 6}
    Train on 10988 samples, validate on 1221 samples
    Epoch 1/20
    11s - loss: 2.4318 - acc: 0.3091 - val_loss: 2.3041 - val_acc: 0.3677
    Epoch 2/20
    6s - loss: 2.1886 - acc: 0.3730 - val_loss: 2.1465 - val_acc: 0.4070
    Epoch 3/20
    6s - loss: 2.0377 - acc: 0.3996 - val_loss: 2.0285 - val_acc: 0.4046
    Epoch 4/20
    6s - loss: 1.9300 - acc: 0.4231 - val_loss: 1.9191 - val_acc: 0.4161
    Epoch 5/20
    6s - loss: 1.8370 - acc: 0.4442 - val_loss: 1.9111 - val_acc: 0.4349
    Epoch 6/20
    6s - loss: 1.7560 - acc: 0.4668 - val_loss: 1.8021 - val_acc: 0.4734
    Epoch 7/20
    6s - loss: 1.6843 - acc: 0.4904 - val_loss: 1.8042 - val_acc: 0.4848
    Epoch 8/20
    6s - loss: 1.6248 - acc: 0.5054 - val_loss: 1.7673 - val_acc: 0.4775
    Epoch 9/20
    6s - loss: 1.5651 - acc: 0.5235 - val_loss: 1.6870 - val_acc: 0.4980
    Epoch 10/20
    6s - loss: 1.5116 - acc: 0.5433 - val_loss: 1.7176 - val_acc: 0.4873
    Epoch 11/20
    Epoch 00010: early stopping
    6s - loss: 1.4561 - acc: 0.5587 - val_loss: 1.7064 - val_acc: 0.5045
    PREDICT
    6080/6104 [============================>.] - ETA: 0sPREDICT
    12192/12209 [============================>.] - ETA: 0sFIT PARAMS : 
    {'nb_filters_1': 64, 'nb_filters_2': 32, 'filter_size_2': 6, 'pool_size_1': 2, 'pool_size_2': 2, 'filter_size_1': 6}
    Train on 10987 samples, validate on 1221 samples
    Epoch 1/20
    21s - loss: 2.4133 - acc: 0.3175 - val_loss: 2.2007 - val_acc: 0.3784
    Epoch 2/20
    16s - loss: 2.0810 - acc: 0.3980 - val_loss: 2.0398 - val_acc: 0.4111
    Epoch 3/20
    16s - loss: 1.8992 - acc: 0.4367 - val_loss: 1.9668 - val_acc: 0.4300
    Epoch 4/20
    16s - loss: 1.7703 - acc: 0.4711 - val_loss: 1.9990 - val_acc: 0.4259
    Epoch 5/20
    16s - loss: 1.6515 - acc: 0.5011 - val_loss: 1.8168 - val_acc: 0.4676
    Epoch 6/20
    16s - loss: 1.5484 - acc: 0.5273 - val_loss: 1.8071 - val_acc: 0.4505
    Epoch 7/20
    16s - loss: 1.4467 - acc: 0.5567 - val_loss: 1.8437 - val_acc: 0.4816
    Epoch 8/20
    Epoch 00007: early stopping
    16s - loss: 1.3516 - acc: 0.5822 - val_loss: 1.9929 - val_acc: 0.4316
    PREDICT
    6105/6105 [==============================] - 6s     
    PREDICT
    12192/12208 [============================>.] - ETA: 0sFIT PARAMS : 
    {'nb_filters_1': 64, 'nb_filters_2': 32, 'filter_size_2': 6, 'pool_size_1': 2, 'pool_size_2': 2, 'filter_size_1': 6}
    Train on 10988 samples, validate on 1221 samples
    Epoch 1/20
    21s - loss: 2.4144 - acc: 0.3155 - val_loss: 2.1948 - val_acc: 0.3702
    Epoch 2/20
    16s - loss: 2.0986 - acc: 0.3851 - val_loss: 2.0060 - val_acc: 0.4193
    Epoch 3/20
    16s - loss: 1.9236 - acc: 0.4210 - val_loss: 1.9512 - val_acc: 0.4439
    Epoch 4/20
    16s - loss: 1.7843 - acc: 0.4562 - val_loss: 1.8420 - val_acc: 0.4496
    Epoch 5/20
    16s - loss: 1.6554 - acc: 0.4942 - val_loss: 1.7659 - val_acc: 0.4873
    Epoch 6/20
    16s - loss: 1.5418 - acc: 0.5270 - val_loss: 1.7152 - val_acc: 0.4644
    Epoch 7/20
    16s - loss: 1.4369 - acc: 0.5521 - val_loss: 1.7513 - val_acc: 0.4709
    Epoch 8/20
    Epoch 00007: early stopping
    16s - loss: 1.3356 - acc: 0.5829 - val_loss: 1.7606 - val_acc: 0.4660
    PREDICT
    6104/6104 [==============================] - 7s     
    PREDICT
    12192/12209 [============================>.] - ETA: 0sFIT PARAMS : 
    {'nb_filters_1': 64, 'nb_filters_2': 32, 'filter_size_2': 6, 'pool_size_1': 2, 'pool_size_2': 2, 'filter_size_1': 6}
    Train on 10988 samples, validate on 1221 samples
    Epoch 1/20
    21s - loss: 2.4053 - acc: 0.3215 - val_loss: 2.2302 - val_acc: 0.3841
    Epoch 2/20
    16s - loss: 2.1166 - acc: 0.3841 - val_loss: 2.0397 - val_acc: 0.4144
    Epoch 3/20
    16s - loss: 1.9592 - acc: 0.4151 - val_loss: 2.0076 - val_acc: 0.4300
    Epoch 4/20
    16s - loss: 1.8377 - acc: 0.4481 - val_loss: 1.9895 - val_acc: 0.4226
    Epoch 5/20
    16s - loss: 1.7329 - acc: 0.4782 - val_loss: 1.8051 - val_acc: 0.4808
    Epoch 6/20
    16s - loss: 1.6309 - acc: 0.5040 - val_loss: 1.8735 - val_acc: 0.4472
    Epoch 7/20
    Epoch 00006: early stopping
    16s - loss: 1.5325 - acc: 0.5329 - val_loss: 1.8275 - val_acc: 0.4439
    PREDICT
    6104/6104 [==============================] - 7s     
    PREDICT
    12209/12209 [==============================] - 6s     
    FIT PARAMS : 
    {'nb_filters_1': 64, 'nb_filters_2': 32, 'filter_size_2': 6, 'pool_size_1': 2, 'pool_size_2': 4, 'filter_size_1': 6}
    Train on 10987 samples, validate on 1221 samples
    Epoch 1/20
    21s - loss: 2.4365 - acc: 0.3072 - val_loss: 2.3821 - val_acc: 0.3358
    Epoch 2/20
    16s - loss: 2.1528 - acc: 0.3829 - val_loss: 2.0981 - val_acc: 0.4029
    Epoch 3/20
    16s - loss: 1.9790 - acc: 0.4144 - val_loss: 1.9802 - val_acc: 0.4103
    Epoch 4/20
    16s - loss: 1.8675 - acc: 0.4420 - val_loss: 1.9134 - val_acc: 0.4545
    Epoch 5/20
    16s - loss: 1.7727 - acc: 0.4675 - val_loss: 1.8149 - val_acc: 0.4660
    Epoch 6/20
    16s - loss: 1.6881 - acc: 0.4944 - val_loss: 1.7682 - val_acc: 0.4848
    Epoch 7/20
    16s - loss: 1.6072 - acc: 0.5125 - val_loss: 1.7537 - val_acc: 0.4832
    Epoch 8/20
    16s - loss: 1.5415 - acc: 0.5327 - val_loss: 1.7418 - val_acc: 0.4873
    Epoch 9/20
    16s - loss: 1.4806 - acc: 0.5519 - val_loss: 1.7264 - val_acc: 0.4930
    Epoch 10/20
    16s - loss: 1.4231 - acc: 0.5683 - val_loss: 1.7064 - val_acc: 0.4767
    Epoch 11/20
    16s - loss: 1.3710 - acc: 0.5833 - val_loss: 1.6552 - val_acc: 0.4963
    Epoch 12/20
    16s - loss: 1.3223 - acc: 0.5984 - val_loss: 1.7924 - val_acc: 0.4865
    Epoch 13/20
    16s - loss: 1.2679 - acc: 0.6169 - val_loss: 1.6130 - val_acc: 0.5053
    Epoch 14/20
    16s - loss: 1.2310 - acc: 0.6221 - val_loss: 1.6382 - val_acc: 0.5135
    Epoch 15/20
    Epoch 00014: early stopping
    16s - loss: 1.1799 - acc: 0.6358 - val_loss: 1.6490 - val_acc: 0.5152
    PREDICT
    6105/6105 [==============================] - 6s     
    PREDICT
    11008/12208 [==========================>...] - ETA: 0s21s - loss: 2.4251 - acc: 0.3150 - val_loss: 2.2371 - val_acc: 0.3612
    Epoch 2/20
    16s - loss: 2.1461 - acc: 0.3755 - val_loss: 2.0529 - val_acc: 0.4005
    Epoch 3/20
    16s - loss: 1.9762 - acc: 0.4088 - val_loss: 1.9210 - val_acc: 0.4496
    Epoch 4/20
    16s - loss: 1.8568 - acc: 0.4418 - val_loss: 1.8634 - val_acc: 0.4529
    Epoch 5/20
    16s - loss: 1.7551 - acc: 0.4732 - val_loss: 1.8352 - val_acc: 0.4570
    Epoch 6/20
    16s - loss: 1.6723 - acc: 0.4934 - val_loss: 1.7381 - val_acc: 0.4939
    Epoch 7/20
    16s - loss: 1.5992 - acc: 0.5119 - val_loss: 1.7426 - val_acc: 0.4791
    Epoch 8/20
    16s - loss: 1.5292 - acc: 0.5348 - val_loss: 1.7363 - val_acc: 0.4873
    Epoch 9/20
    16s - loss: 1.4673 - acc: 0.5469 - val_loss: 1.6747 - val_acc: 0.5053
    Epoch 10/20
    16s - loss: 1.4093 - acc: 0.5680 - val_loss: 1.6265 - val_acc: 0.5217
    Epoch 11/20
    16s - loss: 1.3552 - acc: 0.5803 - val_loss: 1.7796 - val_acc: 0.4652
    Epoch 12/20
    Epoch 00011: early stopping
    16s - loss: 1.3022 - acc: 0.5980 - val_loss: 1.6817 - val_acc: 0.4808
    PREDICT
    6104/6104 [==============================] - 7s     
    PREDICT
    11456/12209 [===========================>..] - ETA: 0s21s - loss: 2.4241 - acc: 0.3167 - val_loss: 2.2213 - val_acc: 0.3857
    Epoch 2/20
    16s - loss: 2.1827 - acc: 0.3711 - val_loss: 2.1209 - val_acc: 0.4046
    Epoch 3/20
    16s - loss: 2.0382 - acc: 0.3997 - val_loss: 2.0175 - val_acc: 0.3980
    Epoch 4/20
    16s - loss: 1.9295 - acc: 0.4225 - val_loss: 1.9010 - val_acc: 0.4316
    Epoch 5/20
    16s - loss: 1.8368 - acc: 0.4451 - val_loss: 1.8650 - val_acc: 0.4545
    Epoch 6/20
    16s - loss: 1.7571 - acc: 0.4689 - val_loss: 1.8397 - val_acc: 0.4529
    Epoch 7/20
    16s - loss: 1.6798 - acc: 0.4901 - val_loss: 1.7282 - val_acc: 0.4996
    Epoch 8/20
    16s - loss: 1.6092 - acc: 0.5100 - val_loss: 1.8681 - val_acc: 0.4595
    Epoch 9/20
    Epoch 00008: early stopping
    16s - loss: 1.5480 - acc: 0.5263 - val_loss: 1.7696 - val_acc: 0.4816
    PREDICT
    6104/6104 [==============================] - 7s     
    PREDICT
    12192/12209 [============================>.] - ETA: 0sFIT PARAMS : 
    {'nb_filters_1': 64, 'nb_filters_2': 32, 'filter_size_2': 6, 'pool_size_1': 4, 'pool_size_2': 2, 'filter_size_1': 6}
    Train on 10987 samples, validate on 1221 samples
    Epoch 1/20
    15s - loss: 2.4168 - acc: 0.3187 - val_loss: 2.3414 - val_acc: 0.3456
    Epoch 2/20
    10s - loss: 2.1491 - acc: 0.3828 - val_loss: 2.1253 - val_acc: 0.3636
    Epoch 3/20
    10s - loss: 1.9769 - acc: 0.4119 - val_loss: 1.9883 - val_acc: 0.4275
    Epoch 4/20
    10s - loss: 1.8419 - acc: 0.4453 - val_loss: 1.8608 - val_acc: 0.4496
    Epoch 5/20
    10s - loss: 1.7486 - acc: 0.4747 - val_loss: 1.8763 - val_acc: 0.4398
    Epoch 6/20
    10s - loss: 1.6690 - acc: 0.4964 - val_loss: 1.7686 - val_acc: 0.4742
    Epoch 7/20
    10s - loss: 1.6070 - acc: 0.5177 - val_loss: 1.7213 - val_acc: 0.4955
    Epoch 8/20
    10s - loss: 1.5440 - acc: 0.5288 - val_loss: 1.7150 - val_acc: 0.4988
    Epoch 9/20
    10s - loss: 1.4892 - acc: 0.5449 - val_loss: 1.7327 - val_acc: 0.4914
    Epoch 10/20
    10s - loss: 1.4404 - acc: 0.5588 - val_loss: 1.6397 - val_acc: 0.5061
    Epoch 11/20
    10s - loss: 1.3886 - acc: 0.5738 - val_loss: 1.7635 - val_acc: 0.4889
    Epoch 12/20
    Epoch 00011: early stopping
    10s - loss: 1.3408 - acc: 0.5867 - val_loss: 1.6435 - val_acc: 0.5283
    PREDICT
    6080/6105 [============================>.] - ETA: 0sPREDICT
    12192/12208 [============================>.] - ETA: 0sFIT PARAMS : 
    {'nb_filters_1': 64, 'nb_filters_2': 32, 'filter_size_2': 6, 'pool_size_1': 4, 'pool_size_2': 2, 'filter_size_1': 6}


### Evaluating the best model's performances on the test set 


```python
print("Best parameters set found:")
print(clf.best_params_)
print()
    
print("Detailed classification report:")
print()
y_true, y_pred = y_test, clf.predict(X_test)
print(classification_report(y_true, y_pred))
```
    Best parameters set found:
    {'nb_filters_1': 64, 'nb_filters_2': 64, 'filter_size_2': 6, 'filter_size_1': 3, 'pool_size_1': 2, 'pool_size_2': 4}
    
    Detailed classification report:
    
    PREDICT
    2035/2035 [==============================] - 2s     
                 precision    recall  f1-score   support
    
              0       0.70      0.39      0.50       588
              1       0.42      0.10      0.16        52
              2       0.29      0.39      0.34        59
              3       0.88      0.11      0.20        62
              4       0.45      0.35      0.40       114
              5       0.48      0.31      0.38       173
              6       0.58      0.36      0.44        39
              7       0.28      0.10      0.15        50
              8       0.56      0.20      0.30        44
              9       0.00      0.00      0.00        35
             10       0.25      0.08      0.12        66
             11       0.29      0.96      0.45       315
             12       0.00      0.00      0.00        93
             13       0.79      0.68      0.73        85
             14       0.29      0.17      0.21        71
             15       0.46      0.51      0.49       105
             16       0.00      0.00      0.00        26
             17       0.44      0.45      0.44        58
    
    avg / total       0.48      0.42      0.39      2035
    


The average classification accuracy is 48% on the test set. As regards the most represented class, the precision goes up to 70%. Precision can drop to 0% for certain classes, but this actually concerns classes that are under-represented in the dataset. This issue might be adressed by artificially increasing the number of instances for those under-represented classes to get a dataset that would be more balanced. This can be done by applying some transformations to available images, such as translations, rotations or changing luminosity, for example.

## In a nutshell : Best model architecture and  hyperparameters

The retained model architecture consists of two successive steps of feature extraction, each one being composed of :
- a convolutional layer 
- a rectification layer with relu activation
- a pooling layer (max pooling)

The hyperparameters that gave the best performances are listed below : 
- nb_filters_1 = 64
- filter_size_1 = 3
- pool_size_1 = 2
- nb_filters_2 = 64
- filter_size_2 = 6
- pool_size_2 = 4

Interestingly, the filter and pooling matrices sizes are greater at the second stage, which means that the process of feature extraction passes consecutively through a "fine" step followed by a "coarse" step.

In a way, pooling corresponds to "losing" information, that's why intuitively the contrary (going from "coarse" to "fine" feature extraction) might be useless : a refined step would be pointless after having thrown away some information !


## Refining the classification step

Until now, we mainly focused on the feature extraction step to enhance the model's performances. In the above, the classification step simply consists in one fully-connected NN layer with softmax activation, which corresponds to a linear separation with respect to the output of the convolutional layers.

In the following, we propose to refine the classification step. To do so, we use the "best model" described above to extract features and then plug them into a more elaborated classifier.


```python
class Classifier(BaseEstimator):  

    def __init__(self, nb_filters_1=64, filter_size_1=3, pool_size_1=2,
                 nb_filters_2=64, filter_size_2=6, pool_size_2=4):
        self.nb_filters_1 = nb_filters_1
        self.filter_size_1 = filter_size_1
        self.pool_size_1 = pool_size_1
        self.nb_filters_2 = nb_filters_2
        self.filter_size_2 = filter_size_2
        self.pool_size_2 = pool_size_2
        
    def preprocess(self, X):
        X = X.reshape((X.shape[0],64,64,3))
        X = (X / 255.)
        X = X.astype(np.float32)
        return X
    
    def preprocess_y(self, y):
        return np_utils.to_categorical(y)
    
    def fit(self, X, y):
        X = self.preprocess(X)
        y = self.preprocess_y(y)
        
        hyper_parameters = dict(
        nb_filters_1 = self.nb_filters_1,
        filter_size_1 = self.filter_size_1,
        pool_size_1 = self.pool_size_1,
        nb_filters_2 = self.nb_filters_2,
        filter_size_2 = self.filter_size_2,
        pool_size_2 = self.pool_size_2
        )
        
        print("FIT PARAMS : ")
        print(hyper_parameters)
        
        self.model = build_model(hyper_parameters)
        
        earlyStopping = EarlyStopping(monitor='val_loss', patience=1, verbose=1, mode='auto')
        self.model.fit(X, y, nb_epoch=20, verbose=1, callbacks=[earlyStopping], validation_split=0.1, 
                       validation_data=None, shuffle=True)
        time.sleep(0.1)
        return self

    def predict(self, X):
        print("PREDICT")
        X = self.preprocess(X)
        return self.model.predict_classes(X)

    def predict_proba(self, X):
        X = self.preprocess(X)
        return self.model.predict(X)
    
    def score(self, X, y):
        print("SCORE")
        print(self.model.evaluate(self, X, y, batch_size=32, verbose=1, sample_weight=None))
        return self.model.evaluate(self, X, y, batch_size=32, verbose=1, sample_weight=None) 
   
```
### Add one hidden layer + relu non-linearity 


```python
def build_model(hp):
    net = Sequential()
    net.add(Convolution2D(hp['nb_filters_1'], hp['filter_size_1'], hp['filter_size_1'], border_mode='same', 
                          input_shape=(64,64,3)))
    net.add(Activation("relu"))
    net.add(MaxPooling2D(pool_size=(hp['pool_size_1'],hp['pool_size_1'])))
    net.add(Convolution2D(hp['nb_filters_2'], hp['filter_size_2'], hp['filter_size_2'], border_mode='same'))
    net.add(Activation("relu"))
    net.add(MaxPooling2D(pool_size=(hp['pool_size_2'],hp['pool_size_2'])))
    net.add(Flatten())
    net.add(Dense(output_dim=200))
    net.add(Activation("relu"))
    net.add(Dense(output_dim=18))
    net.add(Activation("softmax"))
    
    net.compile(loss='categorical_crossentropy', optimizer='adadelta', metrics=['accuracy'])
    
    return net
    
print(unit_test(Classifier,nb_iter=3))
```
    FIT PARAMS : 
    {'nb_filters_1': 64, 'nb_filters_2': 64, 'filter_size_2': 6, 'pool_size_1': 2, 'pool_size_2': 4, 'filter_size_1': 3}
    Train on 14650 samples, validate on 1628 samples
    Epoch 1/20
    14650/14650 [==============================] - 25s - loss: 2.3999 - acc: 0.3186 - val_loss: 2.1897 - val_acc: 0.3821
    Epoch 2/20
    14650/14650 [==============================] - 24s - loss: 2.0281 - acc: 0.4016 - val_loss: 1.8925 - val_acc: 0.4496
    Epoch 3/20
    14650/14650 [==============================] - 24s - loss: 1.7947 - acc: 0.4569 - val_loss: 1.7332 - val_acc: 0.4939
    Epoch 4/20
    14650/14650 [==============================] - 24s - loss: 1.6431 - acc: 0.5005 - val_loss: 1.6444 - val_acc: 0.5092
    Epoch 5/20
    14650/14650 [==============================] - 24s - loss: 1.5272 - acc: 0.5296 - val_loss: 1.5935 - val_acc: 0.5276
    Epoch 6/20
    14650/14650 [==============================] - 24s - loss: 1.4327 - acc: 0.5589 - val_loss: 1.5002 - val_acc: 0.5442
    Epoch 7/20
    14650/14650 [==============================] - 24s - loss: 1.3413 - acc: 0.5845 - val_loss: 1.4692 - val_acc: 0.5516
    Epoch 8/20
    14650/14650 [==============================] - 24s - loss: 1.2568 - acc: 0.6079 - val_loss: 1.4514 - val_acc: 0.5534
    Epoch 9/20
    14650/14650 [==============================] - 24s - loss: 1.1729 - acc: 0.6286 - val_loss: 1.4077 - val_acc: 0.5749
    Epoch 10/20
    14650/14650 [==============================] - 24s - loss: 1.0934 - acc: 0.6581 - val_loss: 1.4175 - val_acc: 0.5737
    Epoch 11/20
    14650/14650 [==============================] - 24s - loss: 1.0039 - acc: 0.6825 - val_loss: 1.4051 - val_acc: 0.5854
    Epoch 12/20
    14650/14650 [==============================] - 24s - loss: 0.9242 - acc: 0.7082 - val_loss: 1.4393 - val_acc: 0.5731
    Epoch 13/20
    14624/14650 [============================>.] - ETA: 0s - loss: 0.8396 - acc: 0.7333Epoch 00012: early stopping
    14650/14650 [==============================] - 24s - loss: 0.8395 - acc: 0.7334 - val_loss: 1.5525 - val_acc: 0.5571
    PREDICT
    4070/4070 [==============================] - 2s     
    FIT PARAMS : 
    {'nb_filters_1': 64, 'nb_filters_2': 64, 'filter_size_2': 6, 'pool_size_1': 2, 'pool_size_2': 4, 'filter_size_1': 3}
    Train on 14650 samples, validate on 1628 samples
    Epoch 1/20
    14650/14650 [==============================] - 24s - loss: 2.3336 - acc: 0.3377 - val_loss: 2.1312 - val_acc: 0.3808
    Epoch 2/20
    14650/14650 [==============================] - 24s - loss: 1.9927 - acc: 0.4091 - val_loss: 1.8993 - val_acc: 0.4189
    Epoch 3/20
    14650/14650 [==============================] - 24s - loss: 1.7855 - acc: 0.4571 - val_loss: 1.7262 - val_acc: 0.4638
    Epoch 4/20
    14650/14650 [==============================] - 24s - loss: 1.6239 - acc: 0.5021 - val_loss: 1.6332 - val_acc: 0.4932
    Epoch 5/20
    14650/14650 [==============================] - 24s - loss: 1.5044 - acc: 0.5356 - val_loss: 1.5486 - val_acc: 0.5178
    Epoch 6/20
    14650/14650 [==============================] - 24s - loss: 1.4059 - acc: 0.5670 - val_loss: 1.5475 - val_acc: 0.5055
    Epoch 7/20
    14650/14650 [==============================] - 24s - loss: 1.3183 - acc: 0.5907 - val_loss: 1.5841 - val_acc: 0.5141
    Epoch 8/20
    14650/14650 [==============================] - 24s - loss: 1.2318 - acc: 0.6158 - val_loss: 1.4497 - val_acc: 0.5418
    Epoch 9/20
    14650/14650 [==============================] - 24s - loss: 1.1420 - acc: 0.6394 - val_loss: 1.4419 - val_acc: 0.5448
    Epoch 10/20
    14650/14650 [==============================] - 24s - loss: 1.0530 - acc: 0.6694 - val_loss: 1.4651 - val_acc: 0.5418
    Epoch 11/20
    14650/14650 [==============================] - 24s - loss: 0.9637 - acc: 0.7003 - val_loss: 1.4246 - val_acc: 0.5516
    Epoch 12/20
    14650/14650 [==============================] - 24s - loss: 0.8739 - acc: 0.7264 - val_loss: 1.5400 - val_acc: 0.5504
    Epoch 13/20
    14624/14650 [============================>.] - ETA: 0s - loss: 0.7698 - acc: 0.7619Epoch 00012: early stopping
    14650/14650 [==============================] - 24s - loss: 0.7699 - acc: 0.7619 - val_loss: 1.4947 - val_acc: 0.5565
    PREDICT
    4064/4070 [============================>.] - ETA: 0sFIT PARAMS : 
    {'nb_filters_1': 64, 'nb_filters_2': 64, 'filter_size_2': 6, 'pool_size_1': 2, 'pool_size_2': 4, 'filter_size_1': 3}
    Train on 14650 samples, validate on 1628 samples
    Epoch 1/20
    14650/14650 [==============================] - 24s - loss: 2.3803 - acc: 0.3227 - val_loss: 2.1862 - val_acc: 0.3765
    Epoch 2/20
    14650/14650 [==============================] - 24s - loss: 2.0157 - acc: 0.4058 - val_loss: 1.8752 - val_acc: 0.4435
    Epoch 3/20
    14650/14650 [==============================] - 24s - loss: 1.7928 - acc: 0.4622 - val_loss: 1.6978 - val_acc: 0.4883
    Epoch 4/20
    14650/14650 [==============================] - 24s - loss: 1.6301 - acc: 0.4988 - val_loss: 1.6056 - val_acc: 0.5068
    Epoch 5/20
    14650/14650 [==============================] - 24s - loss: 1.5174 - acc: 0.5309 - val_loss: 1.5628 - val_acc: 0.5111
    Epoch 6/20
    14650/14650 [==============================] - 24s - loss: 1.4182 - acc: 0.5605 - val_loss: 1.5319 - val_acc: 0.5276
    Epoch 7/20
    14650/14650 [==============================] - 24s - loss: 1.3243 - acc: 0.5863 - val_loss: 1.4949 - val_acc: 0.5350
    Epoch 8/20
    14650/14650 [==============================] - 24s - loss: 1.2323 - acc: 0.6130 - val_loss: 1.4365 - val_acc: 0.5577
    Epoch 9/20
    14650/14650 [==============================] - 24s - loss: 1.1479 - acc: 0.6416 - val_loss: 1.4545 - val_acc: 0.5596
    Epoch 10/20
    14624/14650 [============================>.] - ETA: 0s - loss: 1.0606 - acc: 0.6685Epoch 00009: early stopping
    14650/14650 [==============================] - 24s - loss: 1.0606 - acc: 0.6684 - val_loss: 1.5055 - val_acc: 0.5565
    PREDICT
    4064/4070 [============================>.] - ETA: 0s[ 0.54889435  0.56953317  0.55945946]


### Do we need more hidden layers ?


```python
def build_model(hp):
    net = Sequential()
    net.add(Convolution2D(hp['nb_filters_1'], hp['filter_size_1'], hp['filter_size_1'], border_mode='same', 
                          input_shape=(64,64,3)))
    net.add(Activation("relu"))
    net.add(MaxPooling2D(pool_size=(hp['pool_size_1'],hp['pool_size_1'])))
    net.add(Convolution2D(hp['nb_filters_2'], hp['filter_size_2'], hp['filter_size_2'], border_mode='same'))
    net.add(Activation("relu"))
    net.add(MaxPooling2D(pool_size=(hp['pool_size_2'],hp['pool_size_2'])))
    net.add(Flatten())
    net.add(Dense(output_dim=200))
    net.add(Activation("relu"))
    net.add(Dense(output_dim=200))
    net.add(Activation("relu"))
    net.add(Dense(output_dim=18))
    net.add(Activation("softmax"))
    
    net.compile(loss='categorical_crossentropy', optimizer='adadelta', metrics=['accuracy'])
    
    return net
 
print(unit_test(Classifier,nb_iter=3))
```
    FIT PARAMS : 
    {'nb_filters_1': 64, 'nb_filters_2': 64, 'filter_size_2': 6, 'pool_size_1': 2, 'pool_size_2': 4, 'filter_size_1': 3}
    Train on 14650 samples, validate on 1628 samples
    Epoch 1/20
    14650/14650 [==============================] - 24s - loss: 2.3934 - acc: 0.3220 - val_loss: 2.1610 - val_acc: 0.3913
    Epoch 2/20
    14650/14650 [==============================] - 24s - loss: 2.0271 - acc: 0.4033 - val_loss: 1.8836 - val_acc: 0.4502
    Epoch 3/20
    14650/14650 [==============================] - 24s - loss: 1.7776 - acc: 0.4608 - val_loss: 1.7530 - val_acc: 0.4846
    Epoch 4/20
    14650/14650 [==============================] - 24s - loss: 1.6251 - acc: 0.5016 - val_loss: 1.6260 - val_acc: 0.4957
    Epoch 5/20
    14650/14650 [==============================] - 24s - loss: 1.5078 - acc: 0.5346 - val_loss: 1.5516 - val_acc: 0.5184
    Epoch 6/20
    14650/14650 [==============================] - 24s - loss: 1.4036 - acc: 0.5653 - val_loss: 1.5312 - val_acc: 0.5319
    Epoch 7/20
    14650/14650 [==============================] - 24s - loss: 1.3111 - acc: 0.5931 - val_loss: 1.4656 - val_acc: 0.5516
    Epoch 8/20
    14650/14650 [==============================] - 24s - loss: 1.2182 - acc: 0.6184 - val_loss: 1.5036 - val_acc: 0.5430
    Epoch 9/20
    14650/14650 [==============================] - 24s - loss: 1.1238 - acc: 0.6474 - val_loss: 1.4503 - val_acc: 0.5504
    Epoch 10/20
    14650/14650 [==============================] - 24s - loss: 1.0280 - acc: 0.6777 - val_loss: 1.5220 - val_acc: 0.5719
    Epoch 11/20
    14624/14650 [============================>.] - ETA: 0s - loss: 0.9293 - acc: 0.7105Epoch 00010: early stopping
    14650/14650 [==============================] - 24s - loss: 0.9293 - acc: 0.7104 - val_loss: 1.4935 - val_acc: 0.5498
    PREDICT
    4064/4070 [============================>.] - ETA: 0sFIT PARAMS : 
    {'nb_filters_1': 64, 'nb_filters_2': 64, 'filter_size_2': 6, 'pool_size_1': 2, 'pool_size_2': 4, 'filter_size_1': 3}
    Train on 14650 samples, validate on 1628 samples
    Epoch 1/20
    14650/14650 [==============================] - 24s - loss: 2.3768 - acc: 0.3258 - val_loss: 2.1810 - val_acc: 0.3741
    Epoch 2/20
    14650/14650 [==============================] - 24s - loss: 2.0344 - acc: 0.4038 - val_loss: 1.9726 - val_acc: 0.4165
    Epoch 3/20
    14650/14650 [==============================] - 24s - loss: 1.8287 - acc: 0.4475 - val_loss: 1.7936 - val_acc: 0.4496
    Epoch 4/20
    14650/14650 [==============================] - 24s - loss: 1.6687 - acc: 0.4889 - val_loss: 1.6999 - val_acc: 0.4779
    Epoch 5/20
    14650/14650 [==============================] - 24s - loss: 1.5498 - acc: 0.5237 - val_loss: 1.6077 - val_acc: 0.4951
    Epoch 6/20
    14650/14650 [==============================] - 24s - loss: 1.4623 - acc: 0.5512 - val_loss: 1.5893 - val_acc: 0.5037
    Epoch 7/20
    14650/14650 [==============================] - 24s - loss: 1.3777 - acc: 0.5707 - val_loss: 1.6207 - val_acc: 0.5074
    Epoch 8/20
    14650/14650 [==============================] - 24s - loss: 1.2912 - acc: 0.5971 - val_loss: 1.4847 - val_acc: 0.5283
    Epoch 9/20
    14650/14650 [==============================] - 24s - loss: 1.1997 - acc: 0.6253 - val_loss: 1.4619 - val_acc: 0.5399
    Epoch 10/20
    14650/14650 [==============================] - 24s - loss: 1.1191 - acc: 0.6481 - val_loss: 1.4607 - val_acc: 0.5442
    Epoch 11/20
    14650/14650 [==============================] - 24s - loss: 1.0206 - acc: 0.6768 - val_loss: 1.5098 - val_acc: 0.5424
    Epoch 12/20
    14624/14650 [============================>.] - ETA: 0s - loss: 0.9262 - acc: 0.7062Epoch 00011: early stopping
    14650/14650 [==============================] - 24s - loss: 0.9265 - acc: 0.7063 - val_loss: 1.4709 - val_acc: 0.5534
    PREDICT
    4064/4070 [============================>.] - ETA: 0sFIT PARAMS : 
    {'nb_filters_1': 64, 'nb_filters_2': 64, 'filter_size_2': 6, 'pool_size_1': 2, 'pool_size_2': 4, 'filter_size_1': 3}
    Train on 14650 samples, validate on 1628 samples
    Epoch 1/20
    14650/14650 [==============================] - 25s - loss: 2.3783 - acc: 0.3283 - val_loss: 2.1200 - val_acc: 0.4005
    Epoch 2/20
    14650/14650 [==============================] - 24s - loss: 2.0497 - acc: 0.3990 - val_loss: 1.9008 - val_acc: 0.4545
    Epoch 3/20
    14650/14650 [==============================] - 24s - loss: 1.8448 - acc: 0.4436 - val_loss: 1.7727 - val_acc: 0.4791
    Epoch 4/20
    14650/14650 [==============================] - 24s - loss: 1.6631 - acc: 0.4913 - val_loss: 1.6110 - val_acc: 0.5031
    Epoch 5/20
    14650/14650 [==============================] - 24s - loss: 1.5216 - acc: 0.5265 - val_loss: 1.4961 - val_acc: 0.5393
    Epoch 6/20
    14650/14650 [==============================] - 24s - loss: 1.4021 - acc: 0.5612 - val_loss: 1.5380 - val_acc: 0.5252
    Epoch 7/20
    14650/14650 [==============================] - 24s - loss: 1.3001 - acc: 0.5937 - val_loss: 1.4248 - val_acc: 0.5541
    Epoch 8/20
    14650/14650 [==============================] - 24s - loss: 1.2016 - acc: 0.6247 - val_loss: 1.4159 - val_acc: 0.5547
    Epoch 9/20
    14650/14650 [==============================] - 24s - loss: 1.0986 - acc: 0.6571 - val_loss: 1.3803 - val_acc: 0.5762
    Epoch 10/20
    14650/14650 [==============================] - 24s - loss: 0.9946 - acc: 0.6853 - val_loss: 1.4876 - val_acc: 0.5657
    Epoch 11/20
    14624/14650 [============================>.] - ETA: 0s - loss: 0.8857 - acc: 0.7190Epoch 00010: early stopping
    14650/14650 [==============================] - 24s - loss: 0.8862 - acc: 0.7187 - val_loss: 1.4393 - val_acc: 0.5651
    PREDICT
    4064/4070 [============================>.] - ETA: 0s[ 0.55233415  0.57002457  0.56584767]


The classification accuracies aren't improved by the additionnal hidden layer. In the following, we stick to a classification step that includes only one hidden layer. We propose to use grid search to tune its number of hidden units.

### Tuning the number of units in the fully-connected hidden layer


```python
class Classifier(BaseEstimator):  

    def __init__(self, nb_filters_1=64, filter_size_1=3, pool_size_1=2,
                 nb_filters_2=64, filter_size_2=6, pool_size_2=4, nb_hunits=200):
        self.nb_filters_1 = nb_filters_1
        self.filter_size_1 = filter_size_1
        self.pool_size_1 = pool_size_1
        self.nb_filters_2 = nb_filters_2
        self.filter_size_2 = filter_size_2
        self.pool_size_2 = pool_size_2
        self.nb_hunits = nb_hunits
        
    def preprocess(self, X):
        X = X.reshape((X.shape[0],64,64,3))
        X = (X / 255.)
        X = X.astype(np.float32)
        return X
    
    def preprocess_y(self, y):
        return np_utils.to_categorical(y)
    
    def fit(self, X, y):
        X = self.preprocess(X)
        y = self.preprocess_y(y)
        
        hyper_parameters = dict(
        nb_filters_1 = self.nb_filters_1,
        filter_size_1 = self.filter_size_1,
        pool_size_1 = self.pool_size_1,
        nb_filters_2 = self.nb_filters_2,
        filter_size_2 = self.filter_size_2,
        pool_size_2 = self.pool_size_2,
        nb_hunits = self.nb_hunits
        )
        
        print("FIT PARAMS : ")
        print(hyper_parameters)
        
        self.model = build_model(hyper_parameters)
        
        earlyStopping = EarlyStopping(monitor='val_loss', patience=1, verbose=1, mode='auto')
        self.model.fit(X, y, nb_epoch=20, verbose=1, callbacks=[earlyStopping], validation_split=0.1, 
                       validation_data=None, shuffle=True)
        time.sleep(0.1)
        return self

    def predict(self, X):
        print("PREDICT")
        X = self.preprocess(X)
        return self.model.predict_classes(X)

    def predict_proba(self, X):
        X = self.preprocess(X)
        return self.model.predict(X)
    
    def score(self, X, y):
        print("SCORE")
        print(self.model.evaluate(self, X, y, batch_size=32, verbose=1, sample_weight=None))
        return self.model.evaluate(self, X, y, batch_size=32, verbose=1, sample_weight=None) 
   
```

```python
def build_model(hp):
    net = Sequential()
    net.add(Convolution2D(hp['nb_filters_1'], hp['filter_size_1'], hp['filter_size_1'], border_mode='same', 
                          input_shape=(64,64,3)))
    net.add(Activation("relu"))
    net.add(MaxPooling2D(pool_size=(hp['pool_size_1'],hp['pool_size_1'])))
    net.add(Convolution2D(hp['nb_filters_2'], hp['filter_size_2'], hp['filter_size_2'], border_mode='same'))
    net.add(Activation("relu"))
    net.add(MaxPooling2D(pool_size=(hp['pool_size_2'],hp['pool_size_2'])))
    net.add(Flatten())
    net.add(Dense(output_dim=hp['nb_hunits']))
    net.add(Activation("relu"))
    net.add(Dense(output_dim=18))
    net.add(Activation("softmax"))
    
    net.compile(loss='categorical_crossentropy', optimizer='adadelta', metrics=['accuracy'])
    
    return net

params = {
    'nb_hunits': [100,200,300,400,500]
}
clf = hyperparameter_optim(Classifier,params)

print("Detailed classification report:")
print()
y_true, y_pred = y_test, clf.predict(X_test)
print(classification_report(y_true, y_pred))
```
    FIT PARAMS : 
    {'nb_filters_1': 64, 'nb_filters_2': 64, 'nb_hunits': 100, 'filter_size_2': 6, 'pool_size_1': 2, 'pool_size_2': 4, 'filter_size_1': 3}
    Train on 10987 samples, validate on 1221 samples
    Epoch 1/20
    10987/10987 [==============================] - 18s - loss: 2.4155 - acc: 0.3197 - val_loss: 2.2754 - val_acc: 0.3677
    Epoch 2/20
    10987/10987 [==============================] - 18s - loss: 2.1640 - acc: 0.3815 - val_loss: 2.0492 - val_acc: 0.3972
    Epoch 3/20
    10987/10987 [==============================] - 17s - loss: 1.9982 - acc: 0.4082 - val_loss: 1.9215 - val_acc: 0.4398
    Epoch 4/20
    10987/10987 [==============================] - 17s - loss: 1.8699 - acc: 0.4371 - val_loss: 1.9145 - val_acc: 0.4513
    Epoch 5/20
    10987/10987 [==============================] - 18s - loss: 1.7515 - acc: 0.4729 - val_loss: 1.9440 - val_acc: 0.4308
    Epoch 6/20
    10987/10987 [==============================] - 18s - loss: 1.6441 - acc: 0.5020 - val_loss: 1.7143 - val_acc: 0.4947
    Epoch 7/20
    10987/10987 [==============================] - 18s - loss: 1.5524 - acc: 0.5291 - val_loss: 1.7507 - val_acc: 0.4955
    Epoch 8/20
    10987/10987 [==============================] - 18s - loss: 1.4682 - acc: 0.5495 - val_loss: 1.6926 - val_acc: 0.4922
    Epoch 9/20
    10987/10987 [==============================] - 18s - loss: 1.3876 - acc: 0.5750 - val_loss: 1.6165 - val_acc: 0.5266
    Epoch 10/20
    10987/10987 [==============================] - 18s - loss: 1.3207 - acc: 0.5941 - val_loss: 1.7604 - val_acc: 0.4676
    Epoch 11/20
    10976/10987 [============================>.] - ETA: 0s - loss: 1.2490 - acc: 0.6133Epoch 00010: early stopping
    10987/10987 [==============================] - 18s - loss: 1.2501 - acc: 0.6131 - val_loss: 1.6425 - val_acc: 0.5037
    PREDICT
    6105/6105 [==============================] - 3s     
    PREDICT
    12208/12208 [==============================] - 6s     
    FIT PARAMS : 
    {'nb_filters_1': 64, 'nb_filters_2': 64, 'nb_hunits': 100, 'filter_size_2': 6, 'pool_size_1': 2, 'pool_size_2': 4, 'filter_size_1': 3}
    Train on 10988 samples, validate on 1221 samples
    Epoch 1/20
    10988/10988 [==============================] - 18s - loss: 2.4572 - acc: 0.3057 - val_loss: 2.2610 - val_acc: 0.3604
    Epoch 2/20
    10988/10988 [==============================] - 18s - loss: 2.1766 - acc: 0.3702 - val_loss: 2.1372 - val_acc: 0.3735
    Epoch 3/20
    10988/10988 [==============================] - 18s - loss: 1.9747 - acc: 0.4070 - val_loss: 2.0284 - val_acc: 0.3817
    Epoch 4/20
    10988/10988 [==============================] - 18s - loss: 1.8258 - acc: 0.4487 - val_loss: 1.9646 - val_acc: 0.4038
    Epoch 5/20
    10988/10988 [==============================] - 18s - loss: 1.7006 - acc: 0.4816 - val_loss: 1.7813 - val_acc: 0.4832
    Epoch 6/20
    10988/10988 [==============================] - 18s - loss: 1.6011 - acc: 0.5112 - val_loss: 1.6354 - val_acc: 0.5168
    Epoch 7/20
    10988/10988 [==============================] - 18s - loss: 1.5215 - acc: 0.5288 - val_loss: 1.6020 - val_acc: 0.5348
    Epoch 8/20
    10988/10988 [==============================] - 18s - loss: 1.4471 - acc: 0.5519 - val_loss: 1.5795 - val_acc: 0.5332
    Epoch 9/20
    10988/10988 [==============================] - 18s - loss: 1.3783 - acc: 0.5686 - val_loss: 1.5356 - val_acc: 0.5405
    Epoch 10/20
    10988/10988 [==============================] - 18s - loss: 1.3151 - acc: 0.5935 - val_loss: 1.5983 - val_acc: 0.5487
    Epoch 11/20
    10976/10988 [============================>.] - ETA: 0s - loss: 1.2472 - acc: 0.6100Epoch 00010: early stopping
    10988/10988 [==============================] - 18s - loss: 1.2474 - acc: 0.6099 - val_loss: 1.6720 - val_acc: 0.4922
    PREDICT
    6104/6104 [==============================] - 3s     
    PREDICT
    12209/12209 [==============================] - 6s     
    FIT PARAMS : 
    {'nb_filters_1': 64, 'nb_filters_2': 64, 'nb_hunits': 100, 'filter_size_2': 6, 'pool_size_1': 2, 'pool_size_2': 4, 'filter_size_1': 3}
    Train on 10988 samples, validate on 1221 samples
    Epoch 1/20
    10988/10988 [==============================] - 18s - loss: 2.4525 - acc: 0.3044 - val_loss: 2.3119 - val_acc: 0.3726
    Epoch 2/20
    10988/10988 [==============================] - 18s - loss: 2.1964 - acc: 0.3714 - val_loss: 2.0734 - val_acc: 0.4128
    Epoch 3/20
    10988/10988 [==============================] - 18s - loss: 1.9726 - acc: 0.4122 - val_loss: 1.9916 - val_acc: 0.4357
    Epoch 4/20
    10988/10988 [==============================] - 18s - loss: 1.7996 - acc: 0.4565 - val_loss: 1.8747 - val_acc: 0.4406
    Epoch 5/20
    10988/10988 [==============================] - 18s - loss: 1.6843 - acc: 0.4884 - val_loss: 1.7946 - val_acc: 0.4767
    Epoch 6/20
    10988/10988 [==============================] - 18s - loss: 1.5937 - acc: 0.5100 - val_loss: 1.6616 - val_acc: 0.5176
    Epoch 7/20
    10988/10988 [==============================] - 18s - loss: 1.5136 - acc: 0.5349 - val_loss: 1.6244 - val_acc: 0.5225
    Epoch 8/20
    10988/10988 [==============================] - 18s - loss: 1.4448 - acc: 0.5547 - val_loss: 1.5722 - val_acc: 0.5373
    Epoch 9/20
    10988/10988 [==============================] - 18s - loss: 1.3744 - acc: 0.5763 - val_loss: 1.5902 - val_acc: 0.5291
    Epoch 10/20
    10976/10988 [============================>.] - ETA: 0s - loss: 1.3040 - acc: 0.5940Epoch 00009: early stopping
    10988/10988 [==============================] - 18s - loss: 1.3051 - acc: 0.5938 - val_loss: 1.8242 - val_acc: 0.4496
    PREDICT
    6104/6104 [==============================] - 3s     
    PREDICT
    12209/12209 [==============================] - 6s     
    FIT PARAMS : 
    {'nb_filters_1': 64, 'nb_filters_2': 64, 'nb_hunits': 200, 'filter_size_2': 6, 'pool_size_1': 2, 'pool_size_2': 4, 'filter_size_1': 3}
    Train on 10987 samples, validate on 1221 samples
    Epoch 1/20
    10987/10987 [==============================] - 19s - loss: 2.4080 - acc: 0.3198 - val_loss: 2.2972 - val_acc: 0.3710
    Epoch 2/20
    10987/10987 [==============================] - 18s - loss: 2.0777 - acc: 0.3935 - val_loss: 2.0274 - val_acc: 0.4120
    Epoch 3/20
    10987/10987 [==============================] - 18s - loss: 1.8885 - acc: 0.4299 - val_loss: 1.8568 - val_acc: 0.4660
    Epoch 4/20
    10987/10987 [==============================] - 18s - loss: 1.7151 - acc: 0.4825 - val_loss: 1.7617 - val_acc: 0.4636
    Epoch 5/20
    10987/10987 [==============================] - 18s - loss: 1.5774 - acc: 0.5230 - val_loss: 1.6427 - val_acc: 0.4996
    Epoch 6/20
    10987/10987 [==============================] - 18s - loss: 1.4637 - acc: 0.5514 - val_loss: 1.6581 - val_acc: 0.5012
    Epoch 7/20
    10976/10987 [============================>.] - ETA: 0s - loss: 1.3650 - acc: 0.5804Epoch 00006: early stopping
    10987/10987 [==============================] - 18s - loss: 1.3657 - acc: 0.5804 - val_loss: 1.6499 - val_acc: 0.4824
    PREDICT
    6105/6105 [==============================] - 3s     
    PREDICT
    12208/12208 [==============================] - 5s     
    FIT PARAMS : 
    {'nb_filters_1': 64, 'nb_filters_2': 64, 'nb_hunits': 200, 'filter_size_2': 6, 'pool_size_1': 2, 'pool_size_2': 4, 'filter_size_1': 3}
    Train on 10988 samples, validate on 1221 samples
    Epoch 1/20
    10988/10988 [==============================] - 19s - loss: 2.4550 - acc: 0.3013 - val_loss: 2.2828 - val_acc: 0.3620
    Epoch 2/20
    10988/10988 [==============================] - 19s - loss: 2.1937 - acc: 0.3680 - val_loss: 2.0671 - val_acc: 0.3841
    Epoch 3/20
    10988/10988 [==============================] - 19s - loss: 2.0118 - acc: 0.3992 - val_loss: 1.9980 - val_acc: 0.4095
    Epoch 4/20
    10988/10988 [==============================] - 19s - loss: 1.8781 - acc: 0.4287 - val_loss: 1.8584 - val_acc: 0.4488
    Epoch 5/20
    10988/10988 [==============================] - 19s - loss: 1.7498 - acc: 0.4663 - val_loss: 1.7429 - val_acc: 0.4824
    Epoch 6/20
    10988/10988 [==============================] - 19s - loss: 1.6324 - acc: 0.4992 - val_loss: 1.7429 - val_acc: 0.4865
    Epoch 7/20
    10988/10988 [==============================] - 19s - loss: 1.5288 - acc: 0.5277 - val_loss: 1.6344 - val_acc: 0.5012
    Epoch 8/20
    10988/10988 [==============================] - 19s - loss: 1.4471 - acc: 0.5527 - val_loss: 1.6172 - val_acc: 0.5299
    Epoch 9/20
    10988/10988 [==============================] - 19s - loss: 1.3630 - acc: 0.5683 - val_loss: 1.6137 - val_acc: 0.5078
    Epoch 10/20
    10988/10988 [==============================] - 19s - loss: 1.2806 - acc: 0.6011 - val_loss: 1.5047 - val_acc: 0.5545
    Epoch 11/20
    10988/10988 [==============================] - 19s - loss: 1.2023 - acc: 0.6204 - val_loss: 1.5692 - val_acc: 0.5291
    Epoch 12/20
    10976/10988 [============================>.] - ETA: 0s - loss: 1.1194 - acc: 0.6509Epoch 00011: early stopping
    10988/10988 [==============================] - 19s - loss: 1.1200 - acc: 0.6504 - val_loss: 1.7702 - val_acc: 0.4824
    PREDICT
    6104/6104 [==============================] - 3s     
    PREDICT
    12209/12209 [==============================] - 5s     
    FIT PARAMS : 
    {'nb_filters_1': 64, 'nb_filters_2': 64, 'nb_hunits': 200, 'filter_size_2': 6, 'pool_size_1': 2, 'pool_size_2': 4, 'filter_size_1': 3}
    Train on 10988 samples, validate on 1221 samples
    Epoch 1/20
    10988/10988 [==============================] - 21s - loss: 2.4250 - acc: 0.3133 - val_loss: 2.3032 - val_acc: 0.3563
    Epoch 2/20
    10988/10988 [==============================] - 20s - loss: 2.1564 - acc: 0.3750 - val_loss: 2.1164 - val_acc: 0.3849
    Epoch 3/20
    10988/10988 [==============================] - 20s - loss: 1.9203 - acc: 0.4271 - val_loss: 1.8739 - val_acc: 0.4611
    Epoch 4/20
    10988/10988 [==============================] - 20s - loss: 1.7519 - acc: 0.4687 - val_loss: 1.7526 - val_acc: 0.4840
    Epoch 5/20
    10988/10988 [==============================] - 20s - loss: 1.6318 - acc: 0.5029 - val_loss: 1.6630 - val_acc: 0.5160
    Epoch 6/20
    10988/10988 [==============================] - 20s - loss: 1.5282 - acc: 0.5253 - val_loss: 1.7710 - val_acc: 0.4562
    Epoch 7/20
    10988/10988 [==============================] - 21s - loss: 1.4378 - acc: 0.5562 - val_loss: 1.5830 - val_acc: 0.5348
    Epoch 8/20
    10988/10988 [==============================] - 21s - loss: 1.3477 - acc: 0.5785 - val_loss: 1.5832 - val_acc: 0.5102
    Epoch 9/20
    10988/10988 [==============================] - 20s - loss: 1.2519 - acc: 0.6118 - val_loss: 1.5273 - val_acc: 0.5471
    Epoch 10/20
    10988/10988 [==============================] - 21s - loss: 1.1659 - acc: 0.6350 - val_loss: 1.5967 - val_acc: 0.5307
    Epoch 11/20
    10988/10988 [==============================] - 20s - loss: 1.0730 - acc: 0.6643 - val_loss: 1.4853 - val_acc: 0.5577
    Epoch 12/20
    10988/10988 [==============================] - 20s - loss: 0.9751 - acc: 0.6956 - val_loss: 1.5272 - val_acc: 0.5586
    Epoch 13/20
    10976/10988 [============================>.] - ETA: 0s - loss: 0.8777 - acc: 0.7269Epoch 00012: early stopping
    10988/10988 [==============================] - 21s - loss: 0.8780 - acc: 0.7270 - val_loss: 1.6167 - val_acc: 0.5455
    PREDICT
    6104/6104 [==============================] - 3s     
    PREDICT
    12209/12209 [==============================] - 5s     
    FIT PARAMS : 
    {'nb_filters_1': 64, 'nb_filters_2': 64, 'nb_hunits': 300, 'filter_size_2': 6, 'pool_size_1': 2, 'pool_size_2': 4, 'filter_size_1': 3}
    Train on 10987 samples, validate on 1221 samples
    Epoch 1/20
    10987/10987 [==============================] - 23s - loss: 2.4177 - acc: 0.3178 - val_loss: 2.3123 - val_acc: 0.3456
    Epoch 2/20
    10987/10987 [==============================] - 23s - loss: 2.1038 - acc: 0.3916 - val_loss: 2.0212 - val_acc: 0.4079
    Epoch 3/20
    10987/10987 [==============================] - 23s - loss: 1.9361 - acc: 0.4235 - val_loss: 1.9108 - val_acc: 0.4234
    Epoch 4/20
    10987/10987 [==============================] - 22s - loss: 1.7824 - acc: 0.4587 - val_loss: 1.8251 - val_acc: 0.4627
    Epoch 5/20
    10987/10987 [==============================] - 22s - loss: 1.6363 - acc: 0.5024 - val_loss: 1.7526 - val_acc: 0.4930
    Epoch 6/20
    10987/10987 [==============================] - 22s - loss: 1.5132 - acc: 0.5338 - val_loss: 1.7047 - val_acc: 0.4881
    Epoch 7/20
    10987/10987 [==============================] - 22s - loss: 1.4092 - acc: 0.5654 - val_loss: 1.6098 - val_acc: 0.5250
    Epoch 8/20
    10987/10987 [==============================] - 22s - loss: 1.3050 - acc: 0.5956 - val_loss: 1.5846 - val_acc: 0.5250
    Epoch 9/20
    10987/10987 [==============================] - 22s - loss: 1.2117 - acc: 0.6218 - val_loss: 1.5549 - val_acc: 0.5324
    Epoch 10/20
    10987/10987 [==============================] - 22s - loss: 1.1103 - acc: 0.6504 - val_loss: 1.5160 - val_acc: 0.5553
    Epoch 11/20
    10987/10987 [==============================] - 22s - loss: 1.0012 - acc: 0.6856 - val_loss: 1.5704 - val_acc: 0.5020
    Epoch 12/20
    10976/10987 [============================>.] - ETA: 0s - loss: 0.8949 - acc: 0.7214Epoch 00011: early stopping
    10987/10987 [==============================] - 22s - loss: 0.8945 - acc: 0.7216 - val_loss: 1.6808 - val_acc: 0.5266
    PREDICT
    6105/6105 [==============================] - 3s     
    PREDICT
    12208/12208 [==============================] - 5s     
    FIT PARAMS : 
    {'nb_filters_1': 64, 'nb_filters_2': 64, 'nb_hunits': 300, 'filter_size_2': 6, 'pool_size_1': 2, 'pool_size_2': 4, 'filter_size_1': 3}
    Train on 10988 samples, validate on 1221 samples
    Epoch 1/20
    10988/10988 [==============================] - 26s - loss: 2.3946 - acc: 0.3232 - val_loss: 2.2615 - val_acc: 0.3759
    Epoch 2/20
    10988/10988 [==============================] - 26s - loss: 2.0635 - acc: 0.3915 - val_loss: 2.0997 - val_acc: 0.3972
    Epoch 3/20
    10988/10988 [==============================] - 25s - loss: 1.8482 - acc: 0.4437 - val_loss: 1.8302 - val_acc: 0.4537
    Epoch 4/20
    10988/10988 [==============================] - 25s - loss: 1.6820 - acc: 0.4862 - val_loss: 1.7573 - val_acc: 0.4701
    Epoch 5/20
    10988/10988 [==============================] - 26s - loss: 1.5509 - acc: 0.5188 - val_loss: 1.5952 - val_acc: 0.5217
    Epoch 6/20
    10988/10988 [==============================] - 26s - loss: 1.4378 - acc: 0.5518 - val_loss: 1.5287 - val_acc: 0.5446
    Epoch 7/20
    10988/10988 [==============================] - 26s - loss: 1.3448 - acc: 0.5787 - val_loss: 1.6232 - val_acc: 0.5053
    Epoch 8/20
    10988/10988 [==============================] - 26s - loss: 1.2435 - acc: 0.6080 - val_loss: 1.5052 - val_acc: 0.5250
    Epoch 9/20
    10988/10988 [==============================] - 25s - loss: 1.1466 - acc: 0.6412 - val_loss: 1.4904 - val_acc: 0.5676
    Epoch 10/20
    10988/10988 [==============================] - 25s - loss: 1.0429 - acc: 0.6714 - val_loss: 1.5376 - val_acc: 0.5340
    Epoch 11/20
    10976/10988 [============================>.] - ETA: 0s - loss: 0.9398 - acc: 0.7044Epoch 00010: early stopping
    10988/10988 [==============================] - 25s - loss: 0.9395 - acc: 0.7045 - val_loss: 1.5111 - val_acc: 0.5667
    PREDICT
    6104/6104 [==============================] - 3s     
    PREDICT
    12209/12209 [==============================] - 6s     
    FIT PARAMS : 
    {'nb_filters_1': 64, 'nb_filters_2': 64, 'nb_hunits': 300, 'filter_size_2': 6, 'pool_size_1': 2, 'pool_size_2': 4, 'filter_size_1': 3}
    Train on 10988 samples, validate on 1221 samples
    Epoch 1/20
    10988/10988 [==============================] - 34s - loss: 2.4094 - acc: 0.3173 - val_loss: 2.2661 - val_acc: 0.3808
    Epoch 2/20
    10988/10988 [==============================] - 33s - loss: 2.0764 - acc: 0.3897 - val_loss: 2.0057 - val_acc: 0.3939
    Epoch 3/20
    10988/10988 [==============================] - 33s - loss: 1.8495 - acc: 0.4468 - val_loss: 1.8762 - val_acc: 0.4562
    Epoch 4/20
    10988/10988 [==============================] - 33s - loss: 1.6749 - acc: 0.4884 - val_loss: 1.7906 - val_acc: 0.4734
    Epoch 5/20
    10988/10988 [==============================] - 33s - loss: 1.5504 - acc: 0.5236 - val_loss: 1.7715 - val_acc: 0.4464
    Epoch 6/20
    10988/10988 [==============================] - 33s - loss: 1.4367 - acc: 0.5531 - val_loss: 1.7650 - val_acc: 0.4824
    Epoch 7/20
    10988/10988 [==============================] - 33s - loss: 1.3349 - acc: 0.5830 - val_loss: 1.5986 - val_acc: 0.5176
    Epoch 8/20
    10988/10988 [==============================] - 33s - loss: 1.2263 - acc: 0.6127 - val_loss: 1.5875 - val_acc: 0.4988
    Epoch 9/20
    10988/10988 [==============================] - 33s - loss: 1.1279 - acc: 0.6421 - val_loss: 1.5763 - val_acc: 0.4988
    Epoch 10/20
    10988/10988 [==============================] - 33s - loss: 1.0178 - acc: 0.6797 - val_loss: 1.5477 - val_acc: 0.5389
    Epoch 11/20
    10988/10988 [==============================] - 33s - loss: 0.9015 - acc: 0.7164 - val_loss: 1.7152 - val_acc: 0.5184
    Epoch 12/20
    10976/10988 [============================>.] - ETA: 0s - loss: 0.7852 - acc: 0.7556Epoch 00011: early stopping
    10988/10988 [==============================] - 33s - loss: 0.7850 - acc: 0.7556 - val_loss: 1.6023 - val_acc: 0.5528
    PREDICT
    6104/6104 [==============================] - 4s     
    PREDICT
    12209/12209 [==============================] - 7s     
    FIT PARAMS : 
    {'nb_filters_1': 64, 'nb_filters_2': 64, 'nb_hunits': 400, 'filter_size_2': 6, 'pool_size_1': 2, 'pool_size_2': 4, 'filter_size_1': 3}
    Train on 10987 samples, validate on 1221 samples
    Epoch 1/20
    10987/10987 [==============================] - 42s - loss: 2.4039 - acc: 0.3196 - val_loss: 2.1886 - val_acc: 0.3710
    Epoch 2/20
    10987/10987 [==============================] - 42s - loss: 2.0595 - acc: 0.4004 - val_loss: 1.9828 - val_acc: 0.4136
    Epoch 3/20
    10987/10987 [==============================] - 42s - loss: 1.8498 - acc: 0.4433 - val_loss: 1.8536 - val_acc: 0.4619
    Epoch 4/20
    10987/10987 [==============================] - 42s - loss: 1.6737 - acc: 0.4936 - val_loss: 1.7553 - val_acc: 0.4808
    Epoch 5/20
    10987/10987 [==============================] - 41s - loss: 1.5367 - acc: 0.5301 - val_loss: 1.6240 - val_acc: 0.5160
    Epoch 6/20
    10987/10987 [==============================] - 41s - loss: 1.4307 - acc: 0.5557 - val_loss: 1.5797 - val_acc: 0.5283
    Epoch 7/20
    10987/10987 [==============================] - 42s - loss: 1.3122 - acc: 0.5946 - val_loss: 1.5549 - val_acc: 0.5274
    Epoch 8/20
    10987/10987 [==============================] - 42s - loss: 1.2076 - acc: 0.6246 - val_loss: 1.5979 - val_acc: 0.5127
    Epoch 9/20
    10976/10987 [============================>.] - ETA: 0s - loss: 1.0896 - acc: 0.6600Epoch 00008: early stopping
    10987/10987 [==============================] - 43s - loss: 1.0896 - acc: 0.6600 - val_loss: 1.6945 - val_acc: 0.4881
    PREDICT
    6105/6105 [==============================] - 5s     
    PREDICT
    12208/12208 [==============================] - 11s    
    FIT PARAMS : 
    {'nb_filters_1': 64, 'nb_filters_2': 64, 'nb_hunits': 400, 'filter_size_2': 6, 'pool_size_1': 2, 'pool_size_2': 4, 'filter_size_1': 3}
    Train on 10988 samples, validate on 1221 samples
    Epoch 1/20
    10988/10988 [==============================] - 43s - loss: 2.4278 - acc: 0.3156 - val_loss: 2.1990 - val_acc: 0.3726
    Epoch 2/20
    10988/10988 [==============================] - 42s - loss: 2.1099 - acc: 0.3857 - val_loss: 2.0324 - val_acc: 0.4259
    Epoch 3/20
    10988/10988 [==============================] - 42s - loss: 1.8852 - acc: 0.4317 - val_loss: 1.8110 - val_acc: 0.4832
    Epoch 4/20
    10988/10988 [==============================] - 42s - loss: 1.7207 - acc: 0.4736 - val_loss: 1.6826 - val_acc: 0.5078
    Epoch 5/20
    10988/10988 [==============================] - 42s - loss: 1.5852 - acc: 0.5111 - val_loss: 1.6799 - val_acc: 0.5004
    Epoch 6/20
    10988/10988 [==============================] - 42s - loss: 1.4777 - acc: 0.5405 - val_loss: 1.5787 - val_acc: 0.5192
    Epoch 7/20
    10988/10988 [==============================] - 42s - loss: 1.3821 - acc: 0.5713 - val_loss: 1.5296 - val_acc: 0.5201
    Epoch 8/20
    10988/10988 [==============================] - 42s - loss: 1.2935 - acc: 0.5920 - val_loss: 1.6381 - val_acc: 0.5070
    Epoch 9/20
    10988/10988 [==============================] - 42s - loss: 1.1939 - acc: 0.6216 - val_loss: 1.4624 - val_acc: 0.5397
    Epoch 10/20
    10988/10988 [==============================] - 42s - loss: 1.1028 - acc: 0.6561 - val_loss: 1.4665 - val_acc: 0.5430
    Epoch 11/20
    10976/10988 [============================>.] - ETA: 0s - loss: 1.0005 - acc: 0.6896Epoch 00010: early stopping
    10988/10988 [==============================] - 42s - loss: 1.0006 - acc: 0.6894 - val_loss: 1.4785 - val_acc: 0.5463
    PREDICT
    6104/6104 [==============================] - 7s     
    PREDICT
    12209/12209 [==============================] - 13s    
    FIT PARAMS : 
    {'nb_filters_1': 64, 'nb_filters_2': 64, 'nb_hunits': 400, 'filter_size_2': 6, 'pool_size_1': 2, 'pool_size_2': 4, 'filter_size_1': 3}
    Train on 10988 samples, validate on 1221 samples
    Epoch 1/20
    10988/10988 [==============================] - 45s - loss: 2.4333 - acc: 0.3107 - val_loss: 2.2258 - val_acc: 0.3767
    Epoch 2/20
    10988/10988 [==============================] - 44s - loss: 2.0996 - acc: 0.3895 - val_loss: 2.0153 - val_acc: 0.4201
    Epoch 3/20
    10988/10988 [==============================] - 44s - loss: 1.8838 - acc: 0.4295 - val_loss: 1.8958 - val_acc: 0.4365
    Epoch 4/20
    10988/10988 [==============================] - 44s - loss: 1.7042 - acc: 0.4790 - val_loss: 1.6739 - val_acc: 0.5029
    Epoch 5/20
    10988/10988 [==============================] - 44s - loss: 1.5658 - acc: 0.5175 - val_loss: 1.7198 - val_acc: 0.4742
    Epoch 6/20
    10988/10988 [==============================] - 44s - loss: 1.4411 - acc: 0.5504 - val_loss: 1.5861 - val_acc: 0.5225
    Epoch 7/20
    10988/10988 [==============================] - 44s - loss: 1.3373 - acc: 0.5845 - val_loss: 1.5482 - val_acc: 0.5307
    Epoch 8/20
    10988/10988 [==============================] - 44s - loss: 1.2278 - acc: 0.6152 - val_loss: 1.6918 - val_acc: 0.5070
    Epoch 9/20
    10976/10988 [============================>.] - ETA: 0s - loss: 1.1299 - acc: 0.6440Epoch 00008: early stopping
    10988/10988 [==============================] - 44s - loss: 1.1305 - acc: 0.6438 - val_loss: 1.6038 - val_acc: 0.5274
    PREDICT
    6104/6104 [==============================] - 7s     
    PREDICT
    12209/12209 [==============================] - 13s    
    FIT PARAMS : 
    {'nb_filters_1': 64, 'nb_filters_2': 64, 'nb_hunits': 500, 'filter_size_2': 6, 'pool_size_1': 2, 'pool_size_2': 4, 'filter_size_1': 3}
    Train on 10987 samples, validate on 1221 samples
    Epoch 1/20
    10987/10987 [==============================] - 47s - loss: 2.4051 - acc: 0.3167 - val_loss: 2.1784 - val_acc: 0.3743
    Epoch 2/20
    10987/10987 [==============================] - 47s - loss: 2.0671 - acc: 0.4001 - val_loss: 2.0751 - val_acc: 0.3956
    Epoch 3/20
    10987/10987 [==============================] - 46s - loss: 1.8562 - acc: 0.4448 - val_loss: 1.7869 - val_acc: 0.4668
    Epoch 4/20
    10987/10987 [==============================] - 46s - loss: 1.6758 - acc: 0.4906 - val_loss: 1.7471 - val_acc: 0.4824
    Epoch 5/20
    10987/10987 [==============================] - 46s - loss: 1.5358 - acc: 0.5266 - val_loss: 1.6034 - val_acc: 0.5135
    Epoch 6/20
    10987/10987 [==============================] - 46s - loss: 1.4143 - acc: 0.5627 - val_loss: 1.6166 - val_acc: 0.5061
    Epoch 7/20
    10987/10987 [==============================] - 46s - loss: 1.2883 - acc: 0.6023 - val_loss: 1.5405 - val_acc: 0.5266
    Epoch 8/20
    10987/10987 [==============================] - 46s - loss: 1.1723 - acc: 0.6317 - val_loss: 1.7257 - val_acc: 0.4955
    Epoch 9/20
    10976/10987 [============================>.] - ETA: 0s - loss: 1.0461 - acc: 0.6686Epoch 00008: early stopping
    10987/10987 [==============================] - 46s - loss: 1.0459 - acc: 0.6687 - val_loss: 1.6196 - val_acc: 0.5111
    PREDICT
    6105/6105 [==============================] - 7s     
    PREDICT
    12208/12208 [==============================] - 13s    
    FIT PARAMS : 
    {'nb_filters_1': 64, 'nb_filters_2': 64, 'nb_hunits': 500, 'filter_size_2': 6, 'pool_size_1': 2, 'pool_size_2': 4, 'filter_size_1': 3}
    Train on 10988 samples, validate on 1221 samples
    Epoch 1/20
    10988/10988 [==============================] - 47s - loss: 2.3829 - acc: 0.3242 - val_loss: 2.2198 - val_acc: 0.3661
    Epoch 2/20
    10988/10988 [==============================] - 47s - loss: 2.0736 - acc: 0.3922 - val_loss: 2.0020 - val_acc: 0.4177
    Epoch 3/20
    10988/10988 [==============================] - 47s - loss: 1.8764 - acc: 0.4347 - val_loss: 1.9891 - val_acc: 0.4210
    Epoch 4/20
    10988/10988 [==============================] - 47s - loss: 1.6918 - acc: 0.4847 - val_loss: 1.7410 - val_acc: 0.4726
    Epoch 5/20
    10988/10988 [==============================] - 47s - loss: 1.5654 - acc: 0.5131 - val_loss: 1.6314 - val_acc: 0.5061
    Epoch 6/20
    10988/10988 [==============================] - 47s - loss: 1.4583 - acc: 0.5483 - val_loss: 1.5790 - val_acc: 0.5135
    Epoch 7/20
    10988/10988 [==============================] - 47s - loss: 1.3415 - acc: 0.5798 - val_loss: 1.5202 - val_acc: 0.5455
    Epoch 8/20
    10988/10988 [==============================] - 47s - loss: 1.2406 - acc: 0.6080 - val_loss: 1.4926 - val_acc: 0.5266
    Epoch 9/20
    10988/10988 [==============================] - 47s - loss: 1.1317 - acc: 0.6439 - val_loss: 1.7536 - val_acc: 0.4775
    Epoch 10/20
    10976/10988 [============================>.] - ETA: 0s - loss: 1.0128 - acc: 0.6808Epoch 00009: early stopping
    10988/10988 [==============================] - 47s - loss: 1.0131 - acc: 0.6807 - val_loss: 1.6706 - val_acc: 0.5004
    PREDICT
    6104/6104 [==============================] - 7s     
    PREDICT
    12209/12209 [==============================] - 13s    
    FIT PARAMS : 
    {'nb_filters_1': 64, 'nb_filters_2': 64, 'nb_hunits': 500, 'filter_size_2': 6, 'pool_size_1': 2, 'pool_size_2': 4, 'filter_size_1': 3}
    Train on 10988 samples, validate on 1221 samples
    Epoch 1/20
    10988/10988 [==============================] - 47s - loss: 2.4148 - acc: 0.3134 - val_loss: 2.2467 - val_acc: 0.3604
    Epoch 2/20
    10988/10988 [==============================] - 47s - loss: 2.0612 - acc: 0.3983 - val_loss: 1.9846 - val_acc: 0.4128
    Epoch 3/20
    10988/10988 [==============================] - 47s - loss: 1.8386 - acc: 0.4438 - val_loss: 2.0807 - val_acc: 0.3759
    Epoch 4/20
    10988/10988 [==============================] - 47s - loss: 1.6653 - acc: 0.4884 - val_loss: 1.7855 - val_acc: 0.4857
    Epoch 5/20
    10988/10988 [==============================] - 47s - loss: 1.5453 - acc: 0.5229 - val_loss: 1.6374 - val_acc: 0.5201
    Epoch 6/20
    10988/10988 [==============================] - 47s - loss: 1.4382 - acc: 0.5509 - val_loss: 1.6389 - val_acc: 0.5102
    Epoch 7/20
    10976/10988 [============================>.] - ETA: 0s - loss: 1.3350 - acc: 0.5833Epoch 00006: early stopping
    10988/10988 [==============================] - 47s - loss: 1.3348 - acc: 0.5833 - val_loss: 1.6499 - val_acc: 0.5004
    PREDICT
    6104/6104 [==============================] - 7s     
    PREDICT
    12209/12209 [==============================] - 13s    
    FIT PARAMS : 
    {'nb_filters_1': 64, 'nb_filters_2': 64, 'nb_hunits': 300, 'filter_size_2': 6, 'pool_size_1': 2, 'pool_size_2': 4, 'filter_size_1': 3}
    Train on 16481 samples, validate on 1832 samples
    Epoch 1/20
    16481/16481 [==============================] - 66s - loss: 2.3455 - acc: 0.3335 - val_loss: 2.2480 - val_acc: 0.3444
    Epoch 2/20
    16481/16481 [==============================] - 65s - loss: 1.9754 - acc: 0.4161 - val_loss: 2.1353 - val_acc: 0.3493
    Epoch 3/20
    16481/16481 [==============================] - 65s - loss: 1.7375 - acc: 0.4767 - val_loss: 2.4780 - val_acc: 0.2740
    Epoch 4/20
    16481/16481 [==============================] - 65s - loss: 1.5885 - acc: 0.5156 - val_loss: 1.8358 - val_acc: 0.4476
    Epoch 5/20
    16481/16481 [==============================] - 65s - loss: 1.4753 - acc: 0.5451 - val_loss: 1.9023 - val_acc: 0.4334
    Epoch 6/20
    16481/16481 [==============================] - 65s - loss: 1.3770 - acc: 0.5768 - val_loss: 1.5426 - val_acc: 0.5355
    Epoch 7/20
    16481/16481 [==============================] - 65s - loss: 1.2875 - acc: 0.5992 - val_loss: 1.4253 - val_acc: 0.5529
    Epoch 8/20
    16481/16481 [==============================] - 65s - loss: 1.1921 - acc: 0.6291 - val_loss: 1.5385 - val_acc: 0.5191
    Epoch 9/20
    16480/16481 [============================>.] - ETA: 0s - loss: 1.1005 - acc: 0.6537Epoch 00008: early stopping
    16481/16481 [==============================] - 65s - loss: 1.1006 - acc: 0.6537 - val_loss: 1.8830 - val_acc: 0.5311
    Best parameters set found:
    {'nb_hunits': 300}
    
    Grid scores:
    0.471 (+/-0.029) for {'nb_hunits': 100}
    0.483 (+/-0.058) for {'nb_hunits': 200}
    0.536 (+/-0.035) for {'nb_hunits': 300}
    0.516 (+/-0.053) for {'nb_hunits': 400}
    0.492 (+/-0.009) for {'nb_hunits': 500}
    
    Detailed classification report:
    
    PREDICT
    2035/2035 [==============================] - 2s     
                 precision    recall  f1-score   support
    
              0       0.45      0.98      0.62       588
              1       0.57      0.33      0.41        52
              2       0.35      0.49      0.41        59
              3       0.83      0.16      0.27        62
              4       0.57      0.43      0.49       114
              5       0.72      0.18      0.29       173
              6       0.53      0.41      0.46        39
              7       0.00      0.00      0.00        50
              8       0.62      0.23      0.33        44
              9       0.00      0.00      0.00        35
             10       0.00      0.00      0.00        66
             11       0.69      0.66      0.67       315
             12       0.60      0.06      0.12        93
             13       0.95      0.47      0.63        85
             14       0.86      0.08      0.15        71
             15       0.61      0.37      0.46       105
             16       0.00      0.00      0.00        26
             17       0.54      0.33      0.41        58
    
    avg / total       0.55      0.52      0.45      2035
    


The average model's accuracy has now increased up to 55%. Depending on the classes, precision and recall are quite heterogeneous : thay span from 0% to almost 100%.

To answer the question "are we satisfied with such performances ?", well, it depends on *what* aim we want to achieve here.

We can't say that this model is the best suited to classify accurately all pictures, independetely of the class they belong to. In particular, the model is very bad for under-represented classes. On the contrary, if the model is meant to detect in an efficient way all the instances of class 0, we can say that it does the job, as recall for this class is 98%.
