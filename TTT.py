"""Test ImageNet pretrained DenseNet"""

import os

os.environ["CUDA_VISIBLE_DEVICES"] = "2"

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
        X_train.append(im)
        Y_train.append([1])

for i2 in range(0,11):
    for filename in glob.glob('/home/shayan/PycharmProjects/Dataset/Botnet/Train/' + str(i2) + '/*.png'):
        im=cv2.imread(filename)
        X_train.append(im)
        Y_train.append([0])

print('Finish Reading Data')

XX_train = np.array(X_train)
YY_train = np.array(Y_train)

print(XX_train)
print(YY_train)

del X_train
del Y_train

print('Finish Reading Data')

batch_size = 16

Xs_train = np.empty([batch_size, XX_train.shape[1], XX_train.shape[2], XX_train.shape[3]])
Ys_train = np.empty([batch_size, YY_train.shape[1]])

iter = range(0,len(XX_train),batch_size)
np.random.shuffle(iter)

from keras.utils import to_categorical
YY_train = to_categorical(YY_train)

hist_acc = []
hist_loss = []

print('Start Processing Test Data')

X_test = []
Y_test = []

X_test = []
Y_test = []

for i3 in range(0,12):
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

for i4 in range(0,11):
    for filename in glob.glob('/home/shayan/PycharmProjects/Dataset/Botnet/Test/' + str(i4) + '/*.png'):
        im=cv2.imread(filename)
        im = cv2.resize(im, (224, 224)).astype(np.float32)
        # Subtract mean pixel and multiple by scaling constant
        # Reference: https://github.com/shicai/DenseNet-Caffe
        im[:, :, 0] = (im[:, :, 0] - 103.94) * 0.017
        im[:, :, 1] = (im[:, :, 1] - 116.78) * 0.017
        im[:, :, 2] = (im[:, :, 2] - 123.68) * 0.017
        X_test.append(im)
        Y_test.append([0])

X_test = np.array(X_test)
Y_test = np.array(Y_test)
Y_test = to_categorical(Y_test)

for it in range(0,11):

    print("epoch: " + str(it))

    for iv in iter:

        print("batch: " + str(iv))

        Xs_train = XX_train[iv:iv+batch_size-1]
        Ys_train = YY_train[iv:iv+batch_size-1]

        img_size = 32
        img_chan = 3
        n_classes = 2

        # print('Start Processing Train Data')

        X_Train = np.empty([len(Xs_train), 224, 224, 3])

        for i1 in range(len(Xs_train)):

            X_Train[i1] = cv2.resize(Xs_train[i1], (224, 224)).astype(np.float32)

            # Subtract mean pixel and multiple by scaling constant
            # Reference: https://github.com/shicai/DenseNet-Caffe
            X_Train[i1][:,:,0] = (X_Train[i1][:,:,0] - 103.94) * 0.017
            X_Train[i1][:,:,1] = (X_Train[i1][:,:,1] - 116.78) * 0.017
            X_Train[i1][:,:,2] = (X_Train[i1][:,:,2] - 123.68) * 0.017

        # Start Fine-tuning
        [loss, acc] = model.train_on_batch(X_Train, Ys_train)

        hist_acc.append(acc)
        hist_loss.append(loss)

    f = open("/home/shayan/PycharmProjects/new_proj/results/XStat_Results_epoch" + str(it) + ".txt", "w")

    f.write(str(['Mean of Accuracy in Training: ', np.mean(np.array(hist_acc))]))
    f.write('\n')

    f.write(str(['Mean of Loss in Training: ', np.mean(np.array(hist_loss))]))
    f.write('\n')

    # Make predictions
    score = model.evaluate(X_test, Y_test, verbose=0)

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

    scores = np.asarray(model.predict(X_test))
    predict = np.round(np.asarray(model.predict(X_test)))
    targ = Y_test

    auc.append(sklm.roc_auc_score(targ.flatten(), scores.flatten()))
    confusion.append(sklm.confusion_matrix(targ.flatten(), predict.flatten()))
    precision.append(sklm.precision_score(targ.flatten(), predict.flatten()))
    recall.append(sklm.recall_score(targ.flatten(), predict.flatten()))
    f1s.append(sklm.f1_score(targ.flatten(), predict.flatten()))
    kappa.append(sklm.cohen_kappa_score(targ.flatten(), predict.flatten()))


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
    f.close()

    confusion = np.array(confusion)

    # Plot non-normalized confusion matrix
    fig1 = plt.figure()
    plot_confusion_matrix(confusion[0], classes=['Botnet', 'Normal'],
                          title='Confusion Matrix (Without Normalization)')

    fig1.savefig("/home/shayan/PycharmProjects/new_proj/results/XCM_NoNorm_epoch" + str(it) + ".pdf")
    fig1.savefig("/home/shayan/PycharmProjects/new_proj/results/XCM_NoNorm_epoch" + str(it) + ".eps")
    plt.close(fig1)

    # Plot normalized confusion matrix
    fig2 = plt.figure()
    plot_confusion_matrix(confusion[0], classes=['Botnet', 'Normal'], normalize=True,
                          title='Normalized Confusion Matrix')

    fig2.savefig("/home/shayan/PycharmProjects/new_proj/results/XCM_Norm_epoch" + str(it) + ".pdf")
    fig2.savefig("/home/shayan/PycharmProjects/new_proj/results/XCM_Norm_epoch" + str(it) + ".eps")
    plt.close(fig2)
