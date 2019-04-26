"""Test ImageNet pretrained DenseNet"""

import os

os.environ["CUDA_VISIBLE_DEVICES"] = "3"

import tensorflow as tf

import cv2
import numpy as np
import keras
from keras.optimizers import SGD
import keras.backend as K
import glob
import matplotlib.pyplot as plt
import sklearn.metrics as sklm
import itertools
from skimage.io import imread
from skimage.transform import resize
import numpy as np

# We only test DenseNet-121 in this script for demo purpose
from densenet169 import DenseNet

def plot_confusion_matrix(cm, classes,
                          normalize=False,
                          title='Confusion matrix',
                          cmap=plt.cm.Blues):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')

    print(cm)

    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, format(cm[i, j], fmt),
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.tight_layout()

classes=2

# Use pre-trained weights for Tensorflow backend
weights_path = 'imagenet_models/densenet169_weights_tf.h5'

# Test pretrained model
model = DenseNet(reduction=0.5, classes=classes, weights_path=weights_path, NumNonTrainable=10)

# Learning rate is changed to 1e-3
sgd = SGD(lr=1e-2, decay=1e-6, momentum=0.9, nesterov=True)
model.compile(optimizer=sgd, loss='categorical_crossentropy', metrics=['accuracy'])

print('Start Reading Data')

# "0" = Botnet Traffic Data and "1" = Normal Traffic Data
X_train = []
Y_train = []

for i1 in range(0,12):
    for filename in glob.glob('/home/shayan/PycharmProjects/Dataset/Normal/Train/' + str(i1) + '/*.png'):
        im=cv2.imread(filename)
        im = cv2.resize(im, (224, 224)).astype(np.float32)
        # Subtract mean pixel and multiple by scaling constant
        # Reference: https://github.com/shicai/DenseNet-Caffe
        im[:, :, 0] = (im[:, :, 0] - 103.94) * 0.017
        im[:, :, 1] = (im[:, :, 1] - 116.78) * 0.017
        im[:, :, 2] = (im[:, :, 2] - 123.68) * 0.017
        X_train.append(im)
        Y_train.append([1])

for i2 in range(0,11):
    for filename in glob.glob('/home/shayan/PycharmProjects/Dataset/Botnet/Train/' + str(i2) + '/*.png'):
        im=cv2.imread(filename)
        im = cv2.resize(im, (224, 224)).astype(np.float32)
        # Subtract mean pixel and multiple by scaling constant
        # Reference: https://github.com/shicai/DenseNet-Caffe
        im[:, :, 0] = (im[:, :, 0] - 103.94) * 0.017
        im[:, :, 1] = (im[:, :, 1] - 116.78) * 0.017
        im[:, :, 2] = (im[:, :, 2] - 123.68) * 0.017
        X_train.append(im)
        Y_train.append([0])

print('Finish Reading Data')

X_train = np.array(X_train)
Y_train = np.array(Y_train)

batch_size = 16

from keras.utils import to_categorical
Y_train = to_categorical(Y_train)

print('\nSpliting data')

ind = np.random.permutation(X_train.shape[0])
X_train, Y_train = X_train[ind], Y_train[ind]

VALIDATION_SPLIT = 0.1
n = int(X_train.shape[0] * (1-VALIDATION_SPLIT))
X_valid = X_train[n:]
X_train = X_train[:n]
Y_valid = Y_train[n:]
Y_train = Y_train[:n]

# Start Fine-tuning
model.fit(X_train, Y_train,
          batch_size=batch_size,
          epochs=3,
          shuffle=True,
          verbose=1,
          validation_data=(X_valid, Y_valid))


'''


X_test = []
Y_test = []

for i3 in range(1,11):
    for filename in glob.glob('/home/shayan/PycharmProjects/Dataset/Normal/Test/' + str(i3) + '/*.png'):
        im=cv2.imread(filename)
        im = cv2.resize(im, (224, 224)).astype(np.float32)
        # Subtract mean pixel and multiple by scaling constant
        # Reference: https://github.com/shicai/DenseNet-Caffe
        im[:, :, 0] = (im[:, :, 0] - 103.94) * 0.017
        im[:, :, 1] = (im[:, :, 1] - 116.78) * 0.017
        im[:, :, 2] = (im[:, :, 2] - 123.68) * 0.017
        X_test.append(im)
        Y_test.append([1])

print(X_test)
'''