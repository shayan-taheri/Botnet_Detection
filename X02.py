
classes = 2

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
n = int(len(X_train) * (1 - VALIDATION_SPLIT))
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

for i3 in range(0, 12):
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

for i4 in range(0, 11):
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

# Test pretrained model
model, logits = DenseNet(reduction=0.5, classes=classes, weights_path=weights_path)

# Learning rate is changed to 1e-3
sgd = SGD(lr=1e-2, decay=1e-6, momentum=0.9, nesterov=True)

model.compile(optimizer=sgd, loss='categorical_crossentropy', metrics=['accuracy'])


class Dummy:
    pass


env = Dummy()


def make_fgsm(sess, env, X_data, epochs=1, eps=0.01, batch_size=128):
    print('\nMaking adversarials via FGSM')

    n_sample = X_data.shape[0]
    n_batch = int((n_sample + batch_size - 1) / batch_size)
    X_adv = np.empty_like(X_data)

    for batch in range(n_batch):
        print(' batch {0}/{1}'.format(batch + 1, n_batch))
        print('\r')
        start = batch * batch_size
        end = min(n_sample, start + batch_size)
        feed_dict = {env.x: X_data[start:end], env.adv_eps: eps,
                     env.adv_epochs: epochs}
        adv = sess.run(env.x_fgsm, feed_dict=feed_dict)
        X_adv[start:end] = adv

    return X_adv


def make_jsma(sess, env, X_data, epochs=0.2, eps=1.0, batch_size=128):
    print('\nMaking adversarials via JSMA')

    n_sample = X_data.shape[0]
    n_batch = int((n_sample + batch_size - 1) / batch_size)
    X_adv = np.empty_like(X_data)

    for batch in range(n_batch):
        print(' batch {0}/{1}'.format(batch + 1, n_batch))
        print('\r')
        start = batch * batch_size
        end = min(n_sample, start + batch_size)
        feed_dict = {
            env.x: X_data[start:end],
            env.adv_y: np.random.choice(n_classes),
            env.adv_epochs: epochs,
            env.adv_eps: eps}
        adv = sess.run(env.x_jsma, feed_dict=feed_dict)
        X_adv[start:end] = adv

    return X_adv


def make_deepfool(sess, env, X_data, epochs=1, batch_size=1, noise=True, D=0.1, batch=True):
    print('\nMaking adversarials via DeepFool')

    n_sample = X_data.shape[0]
    n_batch = int((n_sample + batch_size - 1) / batch_size)
    X_adv = np.empty_like(X_data)

    for batch in range(n_batch):
        print(' batch {0}/{1}'.format(batch + 1, n_batch))
        print('\r')
        start = batch * batch_size
        end = min(n_sample, start + batch_size)
        feed_dict = {env.x: X_data[start:end], env.adv_epochs: epochs, env.adv_D: D}
        adv = sess.run(env.x_deepfool, feed_dict=feed_dict)
        X_adv[start:end] = adv

    return X_adv


def make_cw(env, X_data, epochs=1, eps=0.1, batch_size=1):
    """
    Generate adversarial via CW optimization.
    """
    print('\nMaking adversarials via CW')
    n_sample = X_data.shape[0]
    n_batch = int((n_sample + batch_size - 1) / batch_size)
    X_adv = np.empty_like(X_data)
    for batch in range(n_batch):
        with Timer('Batch {0}/{1}   '.format(batch + 1, n_batch)):
            end = min(n_sample, (batch + 1) * batch_size)
            start = end - batch_size
            feed_dict = {
                env.x_fixed: X_data[start:end],
                env.adv_eps: eps,
                env.adv_y: np.random.choice(n_classes)}
            # reset the noise before every iteration
            env.sess.run(env.noise.initializer)
            for epoch in range(epochs):
                env.sess.run(env.adv_train_op, feed_dict=feed_dict)
            xadv = env.sess.run(env.xadv, feed_dict=feed_dict)
            X_adv[start:end] = xadv
    return X_adv


with tf.variable_scope('model', reuse=tf.AUTO_REUSE):
    env.x = tf.placeholder(tf.float32, (None, img_size, img_size, img_chan),
                           name='x')
    env.y = tf.placeholder(tf.float32, (None, n_classes), name='y')

    env.adv_eps = tf.placeholder(tf.float32, (), name='adv_eps')
    env.adv_D = tf.placeholder(tf.float32, (), name='adv_D')
    env.adv_epochs = tf.placeholder(tf.int32, (), name='adv_epochs')
    env.x_fixed = tf.placeholder(
        tf.float32, (batch_size, img_size, img_size, img_chan),
        name='x_fixed')
    env.adv_y = tf.placeholder(tf.int32, (), name='adv_y')
    optimizer = tf.train.AdamOptimizer()

    xent = tf.nn.softmax_cross_entropy_with_logits(labels=env.y,
                                                   logits=logits)
    env.loss = tf.reduce_mean(xent, name='loss')
    env.train_op = optimizer.minimize(env.loss)

    env.saver = tf.train.Saver()

    env.x_fgsm = fgm(model, env.x, epochs=env.adv_epochs, eps=env.adv_eps)

    env.x_deepfool = deepfool(model, env.x, epochs=env.adv_epochs, batch=True)
    env.x_jsma = jsma(model, env.x, env.adv_y, eps=env.adv_eps, epochs=env.adv_epochs)

    env.adv_train_op, env.x_cw, env.noise = cw(model, env.x_fixed,
                                               y=env.adv_y, eps=env.adv_eps,
                                               optimizer=optimizer)

print('\nInitializing graph')

sess = tf.InteractiveSession()
sess.run(tf.global_variables_initializer())
sess.run(tf.local_variables_initializer())

X_test = np.array(X_test)
Y_test = np.array(Y_test)
Y_test = to_categorical(Y_test)

model.fit_generator(generator=my_training_batch_generator,
                    epochs=1,
                    verbose=1,
                    shuffle=True,
                    validation_data=my_validation_batch_generator)

score = model.evaluate(X_test, Y_test, verbose=0)

X_adv_fgsm = X_test
X_adv_jsma = X_test
X_adv_deepfool = X_test
X_adv_cw = X_test

for i in range(len(X_test)):
    xorg, y0 = X_test[i], Y_test[i]

    xorg = np.expand_dims(xorg, axis=0)

    xadvs = [make_fgsm(sess, env, xorg, eps=0, epochs=1),
             make_jsma(sess, env, xorg, eps=0, epochs=1),
             make_deepfool(sess, env, xorg, D=0.1, noise=True, epochs=1, batch=True),
             make_cw(env, xorg, eps=0, epochs=1)]

    X_adv_fgsm[i] = xadvs[0]
    X_adv_jsma[i] = xadvs[1]
    X_adv_deepfool[i] = xadvs[2]
    X_adv_cw[i] = xadvs[3]

    print('\nEvaluating on Single FGSM adversarial data')

    model.evaluate(xadvs[0], Y_test)

    print('\nEvaluating on Single JSMA adversarial data')

    model.evaluate(xadvs[1], Y_test)

    print('\nEvaluating on Single DeepFool adversarial data')

    model.evaluate(xadvs[2], Y_test)

    xorg = np.squeeze(xorg, axis=0)
    #   xadvs = [xorg] + xadvs
    xadvs = [np.squeeze(e) for e in xadvs]

print('\nEvaluating on FGSM adversarial data')

model.evaluate(X_adv_fgsm, Y_test)

print('\nEvaluating on JSMA adversarial data')

model.evaluate(X_adv_jsma, Y_test)

print('\nEvaluating on DeepFool adversarial data')

model.evaluate(X_adv_deepfool, Y_test)
