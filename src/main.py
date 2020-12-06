import network
import load_mnist

training_data ,validation_data , test_data =  load_mnist.load_preprocess_data()
net = network.Network([784,30,10])

net.SGD(training_data, 15,10,3.0,test_data=test_data)

import model_to_json
model_to_json.export(net)
