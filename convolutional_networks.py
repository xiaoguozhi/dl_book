import numpy as np
#np.random.seed(100)
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Flatten
from keras.layers import Convolution2D, MaxPooling2D
from keras.utils import np_utils
from keras.datasets import mnist
(train_samples, train_labels), (test_samples, test_labels) = mnist.load_data()
train_samples = train_samples.reshape(train_samples.shape[0], 28, 28, 1)
test_samples = test_samples.reshape(test_samples.shape[0], 28, 28, 1)
train_samples = train_samples.astype('float32')
test_samples = test_samples.astype('float32')
train_samples = train_samples/255
test_samples = test_samples/255
c_train_labels = np_utils.to_categorical(train_labels, 10)
c_test_labels = np_utils.to_categorical(test_labels, 10)
convnet = Sequential()
convnet.add(Convolution2D(32, 4, 4, activation='relu', input_shape=(28,28,1)))
convnet.add(MaxPooling2D(pool_size=(2,2)))
convnet.add(Convolution2D(32, 3, 3, activation='relu'))
convnet.add(MaxPooling2D(pool_size=(2,2)))
convnet.add(Dropout(0.3))
convnet.add(Flatten())
convnet.add(Dense(10, activation='softmax'))
convnet.compile(loss='mean_squared_error', optimizer='sgd', metrics=['accuracy'])
convnet.fit(train_samples, c_train_labels, batch_size=32, nb_epoch=20, verbose=1)
metrics = convnet.evaluate(test_samples, c_test_labels, verbose=1)
print()
print("\n%s: %.2f%%\n" % (convnet.metrics_names[1], metrics[1]*100))
predictions = convnet.predict(test_samples)#put here unseen_samples