"""Test ImageNet pretrained DenseNet"""

import os

os.environ["CUDA_VISIBLE_DEVICES"] = "2,3"

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
from sklearn import svm, datasets
from sklearn.metrics import accuracy_score, log_loss
from sklearn.svm import SVC, LinearSVC, NuSVC
from sklearn.linear_model import SGDClassifier

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

print('Start Reading Data')

# "0" = Botnet Traffic Data and "1" = Normal Traffic Data
X_train = []
Y_train = []

for i1 in range(0,5):
    for filename in glob.glob('/home/shayan/PycharmProjects/Dataset/Normal/Test/' + str(i1) + '/*.png'):
        im=cv2.imread(filename)
        data = im.reshape(1, -1)
        X_train.append(data)
        Y_train.append([1])

for i2 in range(0,5):
    for filename in glob.glob('/home/shayan/PycharmProjects/Dataset/Botnet/Test/' + str(i2) + '/*.png'):
        im=cv2.imread(filename)
        data = im.reshape(1, -1)
        X_train.append(data)
        Y_train.append([0])

print('Finish Reading Data')

import random

c = list(zip(X_train, Y_train))
random.shuffle(c)
X_train, Y_train = zip(*c)

print('Start Reading Test Data')

X_test = []
Y_test = []

for i3 in range(6,10):
    for filename in glob.glob('/home/shayan/PycharmProjects/Dataset/Normal/Train/' + str(i3) + '/*.png'):
        im = cv2.imread(filename)
        data = im.reshape(1, -1)
        X_test.append(data)
        Y_test.append([1])

for i4 in range(6,10):
    for filename in glob.glob('/home/shayan/PycharmProjects/Dataset/Botnet/Train/' + str(i4) + '/*.png'):
        im = cv2.imread(filename)
        data = im.reshape(1, -1)
        X_test.append(data)
        Y_test.append([0])

X_test = np.array(X_test)
Y_test = np.array(Y_test)

from keras.utils import to_categorical

X_train = np.array(X_train)
Y_train = np.array(Y_train)

for ity in range(0,5):

        # Create a linear SVM classifier
        clf = SGDClassifier(loss="hinge", penalty="l2")

        for iv in range(0,X_train.shape[0]):

            # Train classifier
            clf.partial_fit(X_train[iv], Y_train[iv], classes=[0,1])

        clf_predictions1 = []

        for iv in range(0,X_test.shape[0]):

            clf_predictions1.append(clf.predict(X_test[iv]))

        clf_predictions1 = np.array(clf_predictions1)

        acc = accuracy_score(Y_test, clf_predictions1)

        f = open("/home/shayan/PycharmProjects/DenseNet-Keras-master/results/Stat_Results_SVM" + str(ity) + ".txt", "w")

        f.write(str(['Test accuracy: ', acc]))
        f.write('\n')

        confusion = []
        precision = []
        recall = []
        f1s = []
        kappa = []
        auc = []
        roc = []

        scores = np.asarray(clf_predictions1)
        predict = np.round(np.asarray(clf_predictions1))
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

        fig1.savefig("/home/shayan/PycharmProjects/DenseNet-Keras-master/results/CM_NoNorm_SVM" + str(ity) + ".pdf")
        fig1.savefig("/home/shayan/PycharmProjects/DenseNet-Keras-master/results/CM_NoNorm_SVM" + str(ity) + ".eps")
        plt.close(fig1)

        # Plot normalized confusion matrix
        fig2 = plt.figure()
        plot_confusion_matrix(confusion[0], classes=['Botnet', 'Normal'], normalize=True,
                              title='Normalized Confusion Matrix')

        fig2.savefig("/home/shayan/PycharmProjects/DenseNet-Keras-master/results/CM_Norm_SVM" + str(ity) + ".pdf")
        fig2.savefig("/home/shayan/PycharmProjects/DenseNet-Keras-master/results/CM_Norm_SVM" + str(ity) + ".eps")

        del clf
        del clf_predictions1
        del acc
        del precision
        del recall
        del f1s
        del kappa
        del auc
        del roc
        del confusion
        del scores
        del predict
        del targ

for ity in range(0, 5):

    # Create a linear SVM classifier
    clf = SGDClassifier(loss="log", penalty="l2")

    for iv in range(0, X_train.shape[0]):
        # Train classifier
        clf.partial_fit(X_train[iv], Y_train[iv], classes=[0, 1])

    clf_predictions1 = []

    for iv in range(0, X_test.shape[0]):
        clf_predictions1.append(clf.predict(X_test[iv]))

    clf_predictions1 = np.array(clf_predictions1)

    acc = accuracy_score(Y_test, clf_predictions1)

    f = open("/home/shayan/PycharmProjects/DenseNet-Keras-master/results/Stat_Results_Logistic" + str(ity) + ".txt", "w")

    f.write(str(['Test accuracy: ', acc]))
    f.write('\n')

    confusion = []
    precision = []
    recall = []
    f1s = []
    kappa = []
    auc = []
    roc = []

    scores = np.asarray(clf_predictions1)
    predict = np.round(np.asarray(clf_predictions1))
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

    fig1.savefig("/home/shayan/PycharmProjects/DenseNet-Keras-master/results/CM_NoNorm_Logistic" + str(ity) + ".pdf")
    fig1.savefig("/home/shayan/PycharmProjects/DenseNet-Keras-master/results/CM_NoNorm_Logistic" + str(ity) + ".eps")
    plt.close(fig1)

    # Plot normalized confusion matrix
    fig2 = plt.figure()
    plot_confusion_matrix(confusion[0], classes=['Botnet', 'Normal'], normalize=True,
                          title='Normalized Confusion Matrix')

    fig2.savefig("/home/shayan/PycharmProjects/DenseNet-Keras-master/results/CM_Norm_Logistic" + str(ity) + ".pdf")
    fig2.savefig("/home/shayan/PycharmProjects/DenseNet-Keras-master/results/CM_Norm_Logistic" + str(ity) + ".eps")

    del clf
    del clf_predictions1
    del acc
    del precision
    del recall
    del f1s
    del kappa
    del auc
    del roc
    del confusion
    del scores
    del predict
    del targ
