import tensorflow_datasets as tfds
import numpy as np

def load_data(validation_size = 10000):
    mnist_train, mnist_test = tfds.load('mnist', split=['train', 'test'], shuffle_files=True , as_supervised=True)
    mnist_validation = mnist_train.take(validation_size)
    mnist_train = mnist_train.skip(10000)

    mnist_train = tfds.as_numpy(mnist_train)
    mnist_validation = tfds.as_numpy(mnist_validation)
    mnist_test = tfds.as_numpy(mnist_test)
    
    return (mnist_train,mnist_validation,mnist_test)

def load_preprocess_data():
    tr_d, va_d, te_d = load_data()
    training_data = [(np.reshape(image, (784, 1))/255,vectorized_output(int(label))) for image,label in tr_d ]
    validation_data = [(np.reshape(image, (784, 1))/255,label) for image,label in va_d ]
    test_data = [(np.reshape(image, (784, 1))/255,label) for image,label in te_d ]
    return (training_data, validation_data, test_data)

def vectorized_output(label):
    expected_output = np.zeros ((10 , 1))
    expected_output[label] = 1.0
    return expected_output

if __name__ == "__main__":
    load_preprocess_data()
    


