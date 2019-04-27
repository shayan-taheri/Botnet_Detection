"""Test ImageNet pretrained DenseNet"""

import os
os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "3"

import cv2
import numpy as np
import keras
from keras.optimizers import SGD
import keras.backend as K
import glob
import matplotlib.pyplot as plt
import sklearn.metrics as sklm
import itertools
from keras.utils import Sequence
import time

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

print('Start Reading Data')

# "0" = Botnet Traffic Data and "1" = Normal Traffic Data
X_train = []
Y_train = []

### ***************** LOADING DATASETS *******************

print('Finish Reading Data')

print('\nSpliting data')

import random

c = list(zip(X_train, Y_train))
random.shuffle(c)
X_train, Y_train = zip(*c)

VALIDATION_SPLIT = 0.1
n = int(len(X_train) * (1-VALIDATION_SPLIT))
X_valid = X_train[n:]
X_train = X_train[:n]
Y_valid = Y_train[n:]
Y_train = Y_train[:n]

from keras.utils import to_categorical

Y_train = np.array(Y_train)
Y_train = to_categorical(Y_train)

Y_valid = np.array(Y_valid)
Y_valid = to_categorical(Y_valid)

class MY_Generator(Sequence):

    def __init__(self, image_data, labels, batch_size):
        self.image_data, self.labels = image_data, labels
        self.batch_size = batch_size

    def __len__(self):
        return np.ceil(len(self.image_data) / float(self.batch_size))

    def __getitem__(self, idx):
        batch_x = self.image_data[idx * self.batch_size:(idx + 1) * self.batch_size]
        batch_y = self.labels[idx * self.batch_size:(idx + 1) * self.batch_size]

        arr_batch_x = np.array([cv2.resize(ity, (224, 224)).astype(np.float32) for ity in batch_x])

        for ix in range(0, arr_batch_x.shape[0]):
            im = arr_batch_x[ix]
            im[:, :, 0] = (im[:, :, 0] - 103.94) * 0.017
            im[:, :, 1] = (im[:, :, 1] - 116.78) * 0.017
            im[:, :, 2] = (im[:, :, 2] - 123.68) * 0.017
            arr_batch_x[ix] = im

        return arr_batch_x, batch_y

batch_size = 16

my_training_batch_generator = MY_Generator(X_train, Y_train, batch_size)
my_validation_batch_generator = MY_Generator(X_valid, Y_valid, batch_size)

del X_train
del Y_train
del X_valid
del Y_valid

print('Start Processing Test Data')

X_test = []
Y_test = []

for i3 in range(0,12):
    for filename in glob.glob('/home/shayan/PycharmProjects/Dataset/Normal/Test/' + str(i3) + '/*.png'):
        im = cv2.imread(filename)
        im = cv2.resize(im, (224, 224)).astype(np.float32)
        # Subtract mean pixel and multiple by scaling constant
        # Reference: https://github.com/shicai/DenseNet-Caffe
        im[:, :, 0] = (im[:, :, 0] - 103.94) * 0.017
        im[:, :, 1] = (im[:, :, 1] - 116.78) * 0.017
        im[:, :, 2] = (im[:, :, 2] - 123.68) * 0.017
        X_test.append(im)
        Y_test.append([0])

for i4 in range(0,11):
    for filename in glob.glob('/home/shayan/PycharmProjects/Dataset/Botnet/Test/' + str(i4) + '/*.png'):
        im = cv2.imread(filename)
        im = cv2.resize(im, (224, 224)).astype(np.float32)
        # Subtract mean pixel and multiple by scaling constant
        # Reference: https://github.com/shicai/DenseNet-Caffe
        im[:, :, 0] = (im[:, :, 0] - 103.94) * 0.017
        im[:, :, 1] = (im[:, :, 1] - 116.78) * 0.017
        im[:, :, 2] = (im[:, :, 2] - 123.68) * 0.017
        X_test.append(im)
        Y_test.append([1])

X_test = np.array(X_test)
Y_test = np.array(Y_test)
Y_test = to_categorical(Y_test)

NumNonTrainable = [75]

for ib in range(0,len(NumNonTrainable)):

    # Test pretrained model
    model = DenseNet(reduction=0.5, classes=classes, weights_path=weights_path, NumNonTrainable=NumNonTrainable[ib])

    # Learning rate is changed to 1e-3
    sgd = SGD(lr=1e-2, decay=1e-6, momentum=0.9, nesterov=True)

    model.compile(optimizer=sgd, loss='categorical_crossentropy', metrics=['accuracy'])

    start = time.time()
    model.fit_generator(generator=my_training_batch_generator,
                          epochs=5,
                          verbose=1,
                          shuffle=True,
                          validation_data=my_validation_batch_generator)
    end = time.time()

    train_time = end - start

    start = time.time()
    score = model.evaluate(X_test, Y_test, verbose=0)
    end = time.time()

    test_time = end - start

    f = open("/home/shayan/Codes/DenseNet-Keras-master/Stat_Results_NontTrainable_" + str(ib) + ".txt", "w")

    f.write(str(['Test loss: ', score[0]]))
    f.write('\n')

    f.write(str(['Test accuracy: ', score[1]]))
    f.write('\n')

    confusion = []
    precision = []
    recall = []
    f1s = []
    kappa = []
    auc = []
    roc = []

    scores = np.array([np.argmax(t) for t in np.asarray(model.predict(X_test))])
    predict = np.array([np.argmax(t) for t in np.round(np.asarray(model.predict(X_test)))])
    targ = np.array([np.argmax(t) for t in Y_test])

    auc.append(sklm.roc_auc_score(targ.flatten(), scores.flatten()))
    confusion.append(sklm.confusion_matrix(targ.flatten(), predict.flatten()))
    precision.append(sklm.precision_score(targ.flatten(), predict.flatten()))
    recall.append(sklm.recall_score(targ.flatten(), predict.flatten()))
    f1s.append(sklm.f1_score(targ.flatten(), predict.flatten()))
    kappa.append(sklm.cohen_kappa_score(targ.flatten(), predict.flatten()))

    confusion.append(sklm.confusion_matrix(targ.flatten(), predict.flatten()))


    f.write(str(['Area Under ROC Curve (AUC): ', auc]))
    f.write('\n')
    f.write('Confusion: ')
    f.write('\n')
    f.write(str(np.array(confusion)))
    f.write('\n')
    f.write(str(['Precision: ', precision]))
    f.write('\n')
    f.write(str(['Recall: ', recall]))
    f.write('\n')
    f.write(str(['F-1 Score: ', f1s]))
    f.write('\n')
    f.write(str(['Kappa: ', kappa]))
    f.write('\n')
    f.write(str(['Training Time: ', train_time]))
    f.write('\n')
    f.write(str(['Testing Time: ', test_time]))
    f.close()

    confusion = np.array(confusion)

    # Plot non-normalized confusion matrix
    fig1 = plt.figure()
    plot_confusion_matrix(confusion[0], classes=['Botnet', 'Normal'],
                          title='Confusion Matrix (Without Normalization)')

    fig1.savefig("/home/shayan/Codes/DenseNet-Keras-master/CM_NoNorm_NonTrainable_" + str(ib) + ".pdf")
    fig1.savefig("/home/shayan/Codes/DenseNet-Keras-master/CM_NoNorm_NonTrainable_" + str(ib) + ".eps")
    plt.close(fig1)

    # Plot normalized confusion matrix
    fig2 = plt.figure()
    plot_confusion_matrix(confusion[0], classes=['Botnet', 'Normal'], normalize=True,
                          title='Normalized Confusion Matrix')

    fig2.savefig("/home/shayan/Codes/DenseNet-Keras-master/CM_Norm_NonTrainable_" + str(ib) + ".pdf")
    fig2.savefig("/home/shayan/Codes/DenseNet-Keras-master/CM_Norm_NonTrainable_" + str(ib) + ".eps")

    del model
    del sgd
    del score
    del confusion
    del precision
    del recall
    del f1s
    del kappa
    del auc
    del roc
    del scores
    del predict
    del targ
