from keras.layers import Input, Dense
from keras.models import Model
from keras.datasets import mnist
import numpy as np
(x_train, _), (x_test, _) = mnist.load_data()
x_train = x_train.astype('float32') / 255.0 
x_test = x_test.astype('float32') / 255.0
noise_factor = 0.05
x_train_noisy = x_train + noise_factor * np.random.normal(loc=0.0, scale=1.0, size=x_train.shape) 
x_test_noisy = x_test + noise_factor * np.random.normal(loc=0.0, scale=1.0, size=x_test.shape) 
x_train_noisy = np.clip(x_train_noisy, 0.0, 1.0)
x_test_noisy = np.clip(x_test_noisy, 0.0, 1.0) 
x_train = x_train.reshape((len(x_train), np.prod(x_train.shape[1:])))
x_test = x_test.reshape((len(x_test), np.prod(x_test.shape[1:])))
x_train_noisy = x_train_noisy.reshape((len(x_train_noisy), np.prod(x_train_noisy.shape[1:]))) 
x_test_noisy = x_test_noisy.reshape((len(x_test_noisy), np.prod(x_test_noisy.shape[1:])))
assert x_train_noisy.shape[1] == x_test_noisy.shape[1]
inputs = Input(shape=(x_train_noisy.shape[1],))
encode1 = Dense(128, activation='relu', use_bias=False)(inputs)
encode2 = Dense(64, activation='tanh', use_bias=False)(encode1)
encode3 = Dense(32, activation='relu', use_bias=False)(encode2)
decode3 = Dense(64, activation='relu', use_bias=False)(encode3)
decode2 = Dense(128, activation='sigmoid', use_bias=False)(decode3)
decode1 = Dense(x_train_noisy.shape[1], activation='relu', use_bias=False)(decode2)
autoencoder = Model(inputs, decode1)
autoencoder.compile(optimizer='sgd', loss='mean_squared_error',metrics=['accuracy'])
autoencoder.fit(x_train, x_train,
                epochs=5,
                batch_size=256,
                shuffle=True)
metrics = autoencoder.evaluate(x_test_noisy, x_test, verbose=1)
print("\n%s: %.2f%%\n" % (autoencoder.metrics_names[1], metrics[1]*100))
results = autoencoder.predict(x_test)
print(results.shape)
all_AE_weights_shapes = [x.shape for x in autoencoder.get_weights()]
print(all_AE_weights_shapes)
ww=len(all_AE_weights_shapes)
deeply_encoded_MNIST_weight_matrix = autoencoder.get_weights()[int((ww/2))]
print(deeply_encoded_MNIST_weight_matrix.shape)
autoencoder.save_weights("all_AE_weights.h5")