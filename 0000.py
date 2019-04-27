

if K.image_dim_ordering() == 'th':
  # Transpose image dimensions (Theano uses the channels as the 1st dimension)
  im = im.transpose((2,0,1))

  # Use pre-trained weights for Theano backend
  weights_path = 'imagenet_models/densenet121_weights_th.h5'
else:
  # Use pre-trained weights for Tensorflow backend
  weights_path = 'imagenet_models/densenet121_weights_tf.h5'

# Insert a new dimension for the batch_size
im = np.expand_dims(im, axis=0)

# Test pretrained model
model = DenseNet(reduction=0.5, classes=1000, weights_path=weights_path)

sgd = SGD(lr=1e-2, decay=1e-6, momentum=0.9, nesterov=True)
model.compile(optimizer=sgd, loss='categorical_crossentropy', metrics=['accuracy'])

out = model.predict(im)

# Load ImageNet classes file
classes = []
with open('resources/classes.txt', 'r') as list_:
    for line in list_:
        classes.append(line.rstrip('\n'))

print 'Prediction: '+str(classes[np.argmax(out)])