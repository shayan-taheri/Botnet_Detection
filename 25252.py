
print('\nLoading Biometric')

X_train = []
Y_train = []

### ***************** LOADING DATASETS *******************

print('Finish Reading Data')

y_train = Y_train

X_train0 = []

for ix in range(0,len(X_train)):
    im = X_train[ix]
    im = cv2.resize(im, (img_size, img_size))
    X_train0.append(im)

X_train = np.array(X_train0)

X_train = X_train.astype(np.float32) / 255

to_categorical = tf.keras.utils.to_categorical

y_train = to_categorical(y_train)

print('\nConstruction graph')

def model(x, logits=False, training=False):

    with tf.variable_scope('conv0'):
        z = tf.layers.conv2d(x, filters=64, kernel_size=[3, 3],
                             padding='same', activation=tf.nn.relu)

    with tf.variable_scope('conv1'):
        z = tf.layers.conv2d(z, filters=64, kernel_size=[3, 3],
                             padding='same', activation=tf.nn.relu)

    z = tf.layers.max_pooling2d(z, pool_size=[2, 2], strides=2)

    with tf.variable_scope('conv2'):
        z = tf.layers.conv2d(z, filters=128, kernel_size=[3, 3],
                             padding='same', activation=tf.nn.relu)

    with tf.variable_scope('conv3'):
        z = tf.layers.conv2d(z, filters=128, kernel_size=[3, 3],
                                 padding='same', activation=tf.nn.relu)

    z = tf.layers.max_pooling2d(z, pool_size=[2, 2], strides=2)

    with tf.variable_scope('flatten'):
        shape = z.get_shape().as_list()
        z = tf.reshape(z, [-1, np.prod(shape[1:])])

    with tf.variable_scope("fc1"):
        z = tf.contrib.layers.fully_connected(z, num_outputs=256, activation_fn=tf.nn.relu)

    with tf.variable_scope("fc2"):
        z = tf.contrib.layers.fully_connected(z, num_outputs=256, activation_fn=tf.nn.relu)

    logitsO = tf.layers.dense(z, units=4, name='logits')
    y = tf.nn.softmax(logitsO, name='ybar')

    if logits:
        return y, logitsO
    return y


class Dummy:
    pass


env = Dummy()


with tf.variable_scope('model'):
    env.x = tf.placeholder(tf.float32, (None, img_size, img_size, img_chan),
                           name='x')

    env.y = tf.placeholder(tf.float32, (None, n_classes), name='y')

    env.training = tf.placeholder_with_default(False, (), name='mode')

    env.ybar, logitsO = model(env.x, logits=True, training=env.training)

    env.xs = tf.Variable(np.zeros((1, 128, 128, 3), dtype=np.float32),
                                    name='modifier')

    env.orig_xs = tf.placeholder(tf.float32, [None, 128, 128, 3])

    env.ys = tf.placeholder(tf.int32, [None])

    with tf.variable_scope('acc'):
        count = tf.equal(tf.argmax(env.y, axis=1), tf.argmax(env.ybar, axis=1))
        env.acc = tf.reduce_mean(tf.cast(count, tf.float32), name='acc')

    with tf.variable_scope('loss'):
        xent = tf.nn.softmax_cross_entropy_with_logits(labels=env.y,
                                                       logits=logitsO)
        env.loss = tf.reduce_mean(xent, name='loss')

    with tf.variable_scope('train_op'):

        optimizer = tf.train.AdamOptimizer()
        env.train_op = optimizer.minimize(env.loss)

    env.x_fixed = tf.placeholder(
        tf.float32, (batch_size, img_size, img_size, img_chan),
        name='x_fixed')

    env.saver = tf.train.Saver()

with tf.variable_scope('model', reuse=tf.AUTO_REUSE):
    env.adv_eps = tf.placeholder(tf.float32, (), name='adv_eps')
    env.adv_eta = tf.placeholder(tf.float32, (), name='adv_eta')
    env.adv_epochs = tf.placeholder(tf.int32, (), name='adv_epochs')
    env.adv_y = tf.placeholder(tf.int32, (), name='adv_y')

    env.x_fgsm = fgm(model, env.x, epochs=env.adv_epochs, eps=env.adv_eps)
    env.x_deepfool = deepfool(model, env.x, epochs=env.adv_epochs, batch=True)
    env.x_jsma = jsma(model, env.x, env.adv_y, eps=env.adv_eps,
                      epochs=env.adv_epochs)

    env.cw_train_op, env.x_cw, env.cw_noise = cw(model, env.x_fixed,
                                               y=env.adv_y, eps=env.adv_eps,
                                               optimizer=optimizer)

print('\nInitializing graph')

sess = tf.InteractiveSession()
sess.run(tf.global_variables_initializer())
sess.run(tf.local_variables_initializer())


def evaluate(sess, env, X_data, y_data, batch_size=128):
    """
    Evaluate TF model by running env.loss and env.acc.
    """
    print('\nEvaluating')

    n_sample = X_data.shape[0]
    n_batch = int((n_sample+batch_size-1) / batch_size)
    loss, acc = 0, 0

    for batch in range(n_batch):
        print(' batch {0}/{1}'.format(batch + 1, n_batch))
        print('\r')
        start = batch * batch_size
        end = min(n_sample, start + batch_size)
        cnt = end - start
        batch_loss, batch_acc = sess.run(
            [env.loss, env.acc],
            feed_dict={env.x: X_data[start:end],
                       env.y: y_data[start:end]})
        loss += batch_loss * cnt
        acc += batch_acc * cnt
    loss /= n_sample
    acc /= n_sample

    print(' loss: {0:.4f} acc: {1:.4f}'.format(loss, acc))
    return loss, acc


def train(sess, env, X_data, y_data, X_valid=None, y_valid=None, epochs=1,
          load=False, shuffle=True, batch_size=128, name='model'):
    """
    Train a TF model by running env.train_op.
    """
    if load:
        if not hasattr(env, 'saver'):
            print('\nError: cannot find saver op')
            return
        print('\nLoading saved model')
        return env.saver.restore(sess, 'model/{}'.format(name))

    print('\nTrain model')
    n_sample = X_data.shape[0]
    n_batch = int((n_sample+batch_size-1) / batch_size)
    for epoch in range(epochs):
        print('\nEpoch {0}/{1}'.format(epoch + 1, epochs))

        if shuffle:
            print('\nShuffling data')
            ind = np.arange(n_sample)
            np.random.shuffle(ind)
            X_data = X_data[ind]
            y_data = y_data[ind]

        for batch in range(n_batch):
            print(' batch {0}/{1}'.format(batch + 1, n_batch))
            print('\r')
            start = batch * batch_size
            end = min(n_sample, start + batch_size)
            sess.run(env.train_op, feed_dict={env.x: X_data[start:end],
                                              env.y: y_data[start:end],
                                              env.training: True})
        if X_valid is not None:
            evaluate(sess, env, X_valid, y_valid)

    if hasattr(env, 'saver'):
        print('\n Saving model')
        env.saver.save(sess, 'model/{}'.format(name))


def predict(sess, env, X_data, batch_size=128):
    """
    Do inference by running env.ybar.
    """
    print('\nPredicting')
    n_classes = env.ybar.get_shape().as_list()[1]

    n_sample = X_data.shape[0]
    n_batch = int((n_sample+batch_size-1) / batch_size)
    yval = np.empty((n_sample, n_classes))

    for batch in range(n_batch):
        print(' batch {0}/{1}'.format(batch + 1, n_batch))
        print('\r')
        start = batch * batch_size
        end = min(n_sample, start + batch_size)
        y_batch = sess.run(env.ybar, feed_dict={env.x: X_data[start:end]})
        yval[start:end] = y_batch
    print()
    return yval

def make_fgsm(sess, env, X_data, epochs=2000, eps=5000, batch_size=128):
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

def make_jsma(sess, env, X_data, epochs=2000, eps=5000, batch_size=128):
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


def make_deepfool(sess, env, X_data, epochs=4, batch_size=1, noise=True, batch=True, eta=5000):
    print('\nMaking adversarials via DeepFool')

    n_sample = X_data.shape[0]
    n_batch = int((n_sample + batch_size - 1) / batch_size)
    X_adv = np.empty_like(X_data)

    for batch in range(n_batch):
        print(' batch {0}/{1}'.format(batch + 1, n_batch))
        print('\r')
        start = batch * batch_size
        end = min(n_sample, start + batch_size)
        feed_dict = {env.x: X_data[start:end], env.adv_epochs: epochs, env.adv_eta: eta}
        adv = sess.run(env.x_deepfool, feed_dict=feed_dict)
        X_adv[start:end] = adv

    return X_adv

def make_cw(sess, env, X_data, epochs=2000, eps=5000, batch_size=1):
    """
    Generate adversarial via CW optimization.
    """
    print('\nMaking adversarials via CW')
    n_sample = X_data.shape[0]
    n_batch = int((n_sample + batch_size - 1) / batch_size)
    X_adv = np.empty_like(X_data)
    for batch in range(n_batch):
        with Timer('Batch {0}/{1}   '.format(batch + 1, n_batch)):
            end = min(n_sample, (batch+1) * batch_size)
            start = end - batch_size
            feed_dict = {
                env.x_fixed: X_data[start:end],
                env.adv_eps: eps,
                env.adv_y: np.random.choice(n_classes)}
            # reset the noise before every iteration
            sess.run(env.cw_noise.initializer)
            for epoch in range(epochs):
                sess.run(env.cw_train_op, feed_dict=feed_dict)
            xadv = sess.run(env.x_cw, feed_dict=feed_dict)
            X_adv[start:end] = xadv

    return X_adv

def pgd_func(image, eps=5000, epochs=2000):

    print('\nMaking adversarials via PGD')
    backup = image
    perturbed = image
    perturbed = perturbed + np.random.uniform(-eps, eps, backup.shape)
    perturbed = np.clip(perturbed, 0, 255)
    for _ in range(epochs):
        gradient = np.gradient(perturbed,axis=2)
        perturbed += 0.0001 * np.sign(np.array(gradient))
        perturbed = np.clip(perturbed, backup - eps, backup + eps)
        perturbed = np.clip(perturbed, 0, 255)

    return perturbed

print('\nTraining')

train(sess, env, X_train, y_train, load=False, epochs=20,
      name='biometric')

X_adv_fgsm = np.zeros(X_train.shape)
X_adv_jsma = np.zeros(X_train.shape)
X_adv_deepfool = np.zeros(X_train.shape)
X_adv_cw = np.zeros(X_train.shape)
X_adv_pgd = np.zeros(X_train.shape)

for i in range(len(X_train)):

    xorg_ini, y0 = X_train[i], y_train[i]

    xorg = np.expand_dims(xorg_ini, axis=0)

    xadvs = [make_fgsm(sess, env, xorg, eps=5000, epochs=2000),
             make_jsma(sess, env, xorg, eps=5000, epochs=2000),
             make_deepfool(sess, env, xorg, noise=True, epochs=4),
             make_cw(sess, env, xorg, eps=5000, epochs=2000)]

    X_adv_fgsm[i] = xadvs[0]
    X_adv_jsma[i] = xadvs[1]
    X_adv_deepfool[i] = xadvs[2]
    X_adv_cw[i] = xadvs[3]
    X_adv_pgd[i] = np.expand_dims(pgd_func(image=xorg_ini, eps=5000, epochs=2000), axis=0)

    print('\nEvaluating on Single FGSM adversarial data')

    fgsm_loss, fgsm_acc = evaluate(sess, env, xadvs[0], y_train)

    print('\nEvaluating on Single JSMA adversarial data')

    jsma_loss, jsma_acc = evaluate(sess, env, xadvs[1], y_train)

    print('\nEvaluating on Single DeepFool adversarial data')

    deep_loss, deep_acc = evaluate(sess, env, xadvs[2], y_train)

    print('\nEvaluating on Single CW adversarial data')

    cw_loss, cw_acc = evaluate(sess, env, xadvs[3], y_train)

    print('\nEvaluating on Single PGD adversarial data')

    pgd_loss, pgd_acc = evaluate(sess, env, np.expand_dims(pgd_func(image=xorg_ini, eps=5000, epochs=2000), axis=0), y_train)

    xorg = np.squeeze(xorg, axis=0)
    xadvs = [xorg] + xadvs
    xadvs = [np.squeeze(e) for e in xadvs]

    print('\nSaving figure')

    fig = plt.figure()
    plt.imshow(xadvs[0])
    plt.savefig('/home/shayan/Biometric_Attacked/xadvs_fgsm_' + str(i) + '_' + str(fgsm_acc) + '.png')
    plt.close(fig)

    fig = plt.figure()
    plt.imshow(xadvs[1])
    plt.savefig('/home/shayan/Biometric_Attacked/xadvs_jsma_' + str(i) + '_' + str(jsma_acc) + '.png')
    plt.close(fig)

    fig = plt.figure()
    plt.imshow(xadvs[2])
    plt.savefig('/home/shayan/Biometric_Attacked/xadvs_deepfool_' + str(i) + '_' + str(deep_acc) + '.png')
    plt.close(fig)

    fig = plt.figure()
    plt.imshow(xadvs[3])
    plt.savefig('/home/shayan/Biometric_Attacked/xadvs_cw_' + str(i) + '_' + str(cw_acc) + '.png')
    plt.close(fig)

    fig = plt.figure()
    plt.imshow(xadvs[4])
    plt.savefig('/home/shayan/Biometric_Attacked/xadvs_pgd_' + str(i) + '_' + str(pgd_acc) + '.png')
    plt.close(fig)

