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

im = cv2.resize(cv2.imread('resources/cat.jpg'), (224, 224)).astype(np.float32)
#im = cv2.resize(cv2.imread('resources/shark.jpg'), (224, 224)).astype(np.float32)

# Subtract mean pixel and multiple by scaling constant
# Reference: https://github.com/shicai/DenseNet-Caffe
im[:,:,0] = (im[:,:,0] - 103.94) * 0.017
im[:,:,1] = (im[:,:,1] - 116.78) * 0.017
im[:,:,2] = (im[:,:,2] - 123.68) * 0.017

if K.image_dim_ordering() == 'th':
  # Transpose image dimensions (Theano uses the channels as the 1st dimension)
  im = im.transpose((2,0,1))

  # Use pre-trained weights for Theano backend
  weights_path = '/home/shayan/PycharmProjects/DenseNet-Keras-master/DenseNet-Keras-master/imagenet_models/densenet169_weights_th.h5'
else:
  # Use pre-trained weights for Tensorflow backend
  weights_path = '/home/shayan/PycharmProjects/DenseNet-Keras-master/DenseNet-Keras-master/imagenet_models/densenet169_weights_tf.h5'

# Insert a new dimension for the batch_size
im = np.expand_dims(im, axis=0)


# Test pretrained model
model = DenseNet(reduction=0.5, classes=classes, weights_path=weights_path, NumNonTrainable=10)

# Learning rate is changed to 1e-3
sgd = SGD(lr=1e-2, momentum=0.9, nesterov=True)
model.compile(optimizer=sgd, loss='categorical_crossentropy', metrics=['accuracy'])

'''
out = model.predict(im)

# Load ImageNet classes file
classes = []
with open('resources/classes.txt', 'r') as list_:
    for line in list_:
        classes.append(line.rstrip('\n'))

print 'Prediction: '+str(classes[np.argmax(out)])
'''
#from keras.datasets import cifar10
#(X_train, Y_train), (X_test, Y_test) = cifar10.load_data()

print('Start Reading Data')

X_test = []
Y_test = []

### ***************** LOADING DATASETS *******************

from keras.utils import to_categorical
X_test = np.array(X_test)
Y_test = np.array(Y_test)
Y_test = to_categorical(Y_test)

X_Test = np.empty([len(X_test), 224, 224, 3])

for i5 in range(len(X_test)):
    X_Test[i5] = cv2.resize(X_test[i5], (224, 224)).astype(np.float32)

    # Subtract mean pixel and multiple by scaling constant
    # Reference: https://github.com/shicai/DenseNet-Caffe
    X_Test[i5][:, :, 0] = (X_Test[i5][:, :, 0] - 103.94) * 0.017
    X_Test[i5][:, :, 1] = (X_Test[i5][:, :, 1] - 116.78) * 0.017
    X_Test[i5][:, :, 2] = (X_Test[i5][:, :, 2] - 123.68) * 0.017

batch_size = 16

# Start Fine-tuning
model.fit(X_Test, Y_test,
          batch_size=batch_size,
          epochs=5,
          shuffle=True,
          verbose=1,
)

for ixy in range(0,11):

    if ixy >= 1:
        model.load_weights('/home/shayan/PycharmProjects/DenseNet-Keras-master/DenseNet-Keras-master/imagenet_models/temp_weights.h5')

    # "0" = Botnet Traffic Data and "1" = Normal Traffic Data
    loop = 0
    for i1 in range(ixy+0,ixy+1):
        for filename in glob.glob('/home/shayan/PycharmProjects/Dataset/Normal/Train/' + str(i1) + '/*.png'):
            loop = loop + 1

    for i2 in range(ixy+0,ixy+1):
        for filename in glob.glob('/home/shayan/PycharmProjects/Dataset/Botnet/Train/' + str(i2) + '/*.png'):
            loop = loop + 1

    if loop > 20000:
        X_train = np.empty([20000,224,224,3])
        Y_train = np.empty([20000,1])
    else:
        X_train = np.empty([loop, 224, 224, 3])
        Y_train = np.empty([loop, 1])

    loop = 0
    for i1 in range(ixy+0,ixy+1):
        if loop >= 20000: break
        for filename in glob.glob('/home/shayan/PycharmProjects/Dataset/Normal/Train/' + str(i1) + '/*.png'):
            if loop >= 20000: break
            im=cv2.imread(filename)

            im = cv2.resize(im, (224, 224)).astype(np.float32)

            # Subtract mean pixel and multiple by scaling constant
            # Reference: https://github.com/shicai/DenseNet-Caffe

            im[:, :, 0] = (im[:, :, 0] - 103.94) * 0.017
            im[:, :, 1] = (im[:, :, 1] - 116.78) * 0.017
            im[:, :, 2] = (im[:, :, 2] - 123.68) * 0.017

            X_train[loop] = im
            Y_train[loop] = 1

            loop = loop + 1

    print('Done_1')

    for i2 in range(ixy+0,ixy+1):
        if loop >= 20000: break
        for filename in glob.glob('/home/shayan/PycharmProjects/Dataset/Botnet/Train/' + str(i2) + '/*.png'):
            if loop >= 20000: break
            im=cv2.imread(filename)

            im = cv2.resize(im, (224, 224)).astype(np.float32)

            # Subtract mean pixel and multiple by scaling constant
            # Reference: https://github.com/shicai/DenseNet-Caffe

            im[:, :, 0] = (im[:, :, 0] - 103.94) * 0.017
            im[:, :, 1] = (im[:, :, 1] - 116.78) * 0.017
            im[:, :, 2] = (im[:, :, 2] - 123.68) * 0.017

            X_train[loop] = im
            Y_train[loop] = 0

            loop = loop + 1

    print('Done_2')

    for i3 in range(ixy+0,ixy+1):
        if loop >= 20000: break
        for filename in glob.glob('/home/shayan/PycharmProjects/Dataset/Normal/Test/' + str(i3) + '/*.png'):
            if loop >= 20000: break
            im=cv2.imread(filename)

            im = cv2.resize(im, (224, 224)).astype(np.float32)

            # Subtract mean pixel and multiple by scaling constant
            # Reference: https://github.com/shicai/DenseNet-Caffe

            im[:, :, 0] = (im[:, :, 0] - 103.94) * 0.017
            im[:, :, 1] = (im[:, :, 1] - 116.78) * 0.017
            im[:, :, 2] = (im[:, :, 2] - 123.68) * 0.017

            X_train[loop] = im
            Y_train[loop] = 1

            loop = loop + 1

    print('Done_3')

    for i4 in range(ixy+0,ixy+1):
        if loop >= 20000: break
        for filename in glob.glob('/home/shayan/PycharmProjects/Dataset/Botnet/Test/' + str(i4) + '/*.png'):
            if loop >= 20000: break
            im=cv2.imread(filename)

            im = cv2.resize(im, (224, 224)).astype(np.float32)

            # Subtract mean pixel and multiple by scaling constant
            # Reference: https://github.com/shicai/DenseNet-Caffe

            im[:, :, 0] = (im[:, :, 0] - 103.94) * 0.017
            im[:, :, 1] = (im[:, :, 1] - 116.78) * 0.017
            im[:, :, 2] = (im[:, :, 2] - 123.68) * 0.017

            X_train[loop] = im
            Y_train[loop] = 0

            loop = loop + 1

    print('Done_4')

    print('Finish Reading Data')

    img_size = 32
    img_chan = 3
    n_classes = 2

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
    hist = model.fit(X_train, Y_train,
                     batch_size=batch_size,
                     epochs=1,
                     shuffle=True,
                     verbose=1,
                     validation_data=(X_valid, Y_valid)
    )

    model.save_weights('/home/shayan/PycharmProjects/DenseNet-Keras-master/DenseNet-Keras-master/imagenet_models/temp_weights.h5')

    # Make predictions
    score = model.evaluate(X_Test, Y_test, verbose=0)
    print('Test loss:', score[0])
    print('Test accuracy:', score[1])

    confusion = []
    precision = []
    recall = []
    f1s = []
    kappa = []
    auc = []
    roc = []

    scores = np.asarray(model.predict(X_Test))
    predict = np.round(np.asarray(model.predict(X_Test)))
    targ = Y_test

    auc.append(sklm.roc_auc_score(targ.flatten(), scores.flatten()))
    confusion.append(sklm.confusion_matrix(targ.flatten(), predict.flatten()))
    precision.append(sklm.precision_score(targ.flatten(), predict.flatten()))
    recall.append(sklm.recall_score(targ.flatten(), predict.flatten()))
    f1s.append(sklm.f1_score(targ.flatten(), predict.flatten()))
    kappa.append(sklm.cohen_kappa_score(targ.flatten(), predict.flatten()))

    f = open("/home/shayan/PycharmProjects/DenseNet-Keras-master/results/Stat_Results.txt","w")
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
    plot_confusion_matrix(confusion[0], classes=['Botnet','Normal'],
                          title='Confusion Matrix (Without Normalization)')
    fig1.savefig("/home/shayan/PycharmProjects/DenseNet-Keras-master/results/CM_NoNorm.pdf")
    fig1.savefig("/home/shayan/PycharmProjects/DenseNet-Keras-master/results/CM_NoNorm.eps")
    plt.close(fig1)

    # Plot normalized confusion matrix
    fig2 = plt.figure()
    plot_confusion_matrix(confusion[0], classes=['Botnet','Normal'], normalize=True,
                          title='Normalized Confusion Matrix')

    fig2.savefig("/home/shayan/PycharmProjects/DenseNet-Keras-master/results/CM_Norm.pdf")
    fig2.savefig("/home/shayan/PycharmProjects/DenseNet-Keras-master/results/CM_Norm.eps")
    plt.close(fig2)
