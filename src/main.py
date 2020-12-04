import network
import load_mnist

training_data ,validation_data , test_data =  load_mnist.load_preprocess_data()
net = network.Network([784,100,10])

net.SGD(training_data, 30,1000,0.02,test_data=test_data)