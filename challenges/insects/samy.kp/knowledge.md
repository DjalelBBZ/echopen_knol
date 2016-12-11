---
title: Insects classification
authors:
- Samy
tags:
- RAMP
- ConvNets
- Challenge
created_at: 2016-11-08 00:00:00
updated_at: 2016-12-11 13:47:11.277148
tldr: Insect classification challenge
---
# Pollenating Insects classification - Samy Blusseau

## Method

### 1) Setting up the working environment
To address the assigned task, I downloaded the necessary files (data, starting kit) and installed helpful software (Jupyter, TensorFlow, Theano, Keras).


### 2) Documentation
I learnt the very basics of Convolutional Neural Networks on a Standford course website (http://vision.stanford.edu/teaching/cs231n/syllabus.html) for which the corresponding videos can be found here http://academictorrents.com/details/46c5af9e2075d9af06f280b55b65cf9b44eb9fe7/collections.

To get started with Keras, I relied on Keras Documentation and on this tutorial http://online.cambridgecoding.com/notebooks/cca_admin/convolutional-neural-networks-with-keras.


### 3) Strategy
Quite soon I realized that I would need more time to come up with an interesting solution the problem. Not only my knowledge and experience of CNNs are very limited, but testing different architectures and monitoring the training in order to find the right strategy also take long.

Therefore I decided to implement a very simple CNN, inspired at examples I found in the above references. The justification for the chosen architecture is that it seems to follow some common and good practices.


```python
from keras.models import Model
from keras.layers import Input, Convolution2D, MaxPooling2D, Activation, Dense, Dropout, Flatten
from keras.utils import np_utils
from keras.models import model_from_json
from keras.models import Sequential
import numpy as np
```

```python
# Read data
data = np.load("train_64x64.npz")
X, y = data['X'], data['y'] # X contains the images, y the labels

# Split data into training and test sets
# We checked that the 18 classes were roughly equally represented in both sets
num_train=18000 # 18000 training examples
X_train = X[0:num_train, :, :, :]
y_train = y[0:num_train]
X_test = X[num_train:, :, :, :]
y_test = y[num_train:]

depth, height, width = X_train.shape[1:4] # size of the data
num_test = X_test.shape[0] # number of test examples
class_id = np.unique(y_train) # the numerical class labels
num_classes = class_id.shape[0] # the number of insect classes

# Convert the original numerical class labels into their indexes from 0 to num_classes
cv_dict = dict()
for i in range(0, len(class_id)):
    cv_dict[str(class_id[i])] = i
y_train2 = np.zeros(y_train.shape)
for i in range(len(y_train2)):
    y_train2[i] = int(cv_dict[str(y_train[i])])
y_test2 = np.zeros(y_test.shape)
for i in range(len(y_test2)):
    y_test2[i] = int(cv_dict[str(y_test[i])])
    
    
# Pre-process training and test sets
X_train = (X_train / 255.)
X_train = X_train.astype(np.float32)

X_test = (X_test / 255.)
X_test = X_test.astype(np.float32)

Y_train = np_utils.to_categorical(y_train2, num_classes) # One-hot encode the labels
Y_train =Y_train.astype(np.float32)

Y_test = np_utils.to_categorical(y_test2, num_classes) # One-hot encode the labels
Y_test =Y_test.astype(np.float32)
```

```python
#-------------------------------------------------------------------------#
#----------------------------- MODEL -------------------------------------#
#-------------------------------------------------------------------------#

# CNN parameters
batch_size = 32
num_epochs = 20
kernel_size = 3 #  3x3 filters
pool_sz = 2 # 2x2 pooling
conv_depth_1 = 32 # 32 kernels for the first two convolutional layers
conv_depth_2 = 64 # 64 kernels for the last two convolutional layers
drop_prob_1 = 0.25 # dropout probability after the convolutional layers
drop_prob_2 = 0.5 # dropout probability after the fully connected layer
hidden_size = 512 # the FC layer will have 512 neurons

# Layers
model = Sequential()
# Conv [32] -> Conv [32] -> Pool (with dropout on the pooling layer)
model.add(Convolution2D(conv_depth_1, kernel_size, kernel_size, border_mode='same', input_shape=(depth, height, width)))
model.add(Activation('relu'))
model.add(Convolution2D(conv_depth_1, kernel_size, kernel_size, border_mode='same'))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(pool_sz, pool_sz)))
model.add(Dropout(drop_prob_1))
# Conv [64] -> Conv [64] -> Pool (with dropout on the pooling layer)
model.add(Convolution2D(conv_depth_2, kernel_size, kernel_size, border_mode='same'))
model.add(Activation('relu'))
model.add(Convolution2D(conv_depth_2, kernel_size, kernel_size, border_mode='same'))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(pool_sz, pool_sz)))
model.add(Dropout(drop_prob_1))
# Now flatten to 1D, apply FC -> ReLU (with dropout) -> softmax
model.add(Flatten())
model.add(Dense(hidden_size))
model.add(Activation('relu'))
model.add(Dropout(drop_prob_2))
model.add(Dense(num_classes))
model.add(Activation('softmax'))

# Save model
json_string = model.to_json()
fid = open('model.json', 'w')
fid.write(json_string)
fid.close

# Compile model
model.compile(loss='categorical_crossentropy', # using the cross-entropy loss function
              optimizer='adam', # using the Adam optimiser
              metrics=['accuracy']) # reporting the accuracy
```

```python
# Train model
model.fit(X_train, Y_train, # Train the model using the training set...
          batch_size=batch_size, nb_epoch=num_epochs,
          verbose=1, validation_split=0.1) # ...holding out 10% of the data for validation

# Fit model on training set
model.fit(X_train, Y_train, batch_size=batch_size, nb_epoch=num_epochs,
          verbose=1, validation_split=0.1) # ...holding out 10% of the data for validation

# Evaluate model on test set
score=model.evaluate(X_test, Y_test, verbose=1)
print(score)

# Save weights
model.save_weights('model_weights.hdf5')
```

```python
# This piece of code is meant to load the just trained model and its weights, and evaluate it
# For it to work, X_test and Y_test need to be defined

# Load model
fid = open('model.json', 'r')
json_string = fid.read()
fid.close()

# Load weights
model.load_weights('model_weights.hdf5', by_name=False)
model.compile(loss='categorical_crossentropy', # using the cross-entropy loss function
              optimizer='adam', # using the Adam optimiser
              metrics=['accuracy']) # reporting the accuracy

# Evaluate model
score=model.evaluate(X_test, Y_test, verbose=1) # Evaluate the trained model on the test set
print(score)

# Predicted classes
classes = model.predict_classes(X_test, verbose=1)
ind = np.where(classes > 0) # indices of samples which were not labeled Apis mellifera
print(ind) # Empty tuple: all samples were labeled Apis mellifera
```
## Results

The accuracy of the model is about 29%. This is very low, but what is more worrying is that along training the loss decreased very slowly and the training accuracy remained constant after the second epoch.

These clues made me suspect that one of the problems may be the over-representation of the first class in the data set. Indeed, the accuracy is roughly equal to the proportion of Apis mellifera in the data set, as if the network had been trained to the detecttion of that class only.

I checked the predicted classes of the test set (see code here above), and as suspected all the samples were assigned the Apis mellifera class. I am not sure if this is due to the structure of the data or to a bug in my implementation.

## What I would do next

The most urgent task is to figure out if there is a bug in the code causing the classification of all samples into the first class. A simple thing to try would be build a training set with uniform representation of classes. This would mean a much smaller training set, but should help spot a bug.

If not, then different architectures should be tested.
To that aim, I would try and follow an advice I read in the previously mentionned references: 
- download a pre-trained model that already prooved high accuracy in image classification (ImageNet?)
- perform fine tuning of that model, that is, further optimize its parameters to the present specific task.

If no architecture shows improvement, then it might be relevant to change the structure of the training data set, so that no class be as over represented as Apis mellifera so far.

Furthermore, I would spend more time understanding the methods that help make the right architectural choices.
Finally, speeding up the training, by running it on GPU, seems necessary so that more variations can be tested.
