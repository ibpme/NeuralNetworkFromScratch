import numpy as np
import random


def sigmoid(z):
    "Sigmoid Function"
    return 1.0/(1.0+np.exp(-z))

def sigmoid_prime(z):
    " Derivative of Sigmoid Function"
    return sigmoid(z)*(1-sigmoid(z))

class Network(object):

    def __init__(self,sizez:list):
        """
        Sizez is an list of the number of neuron in each layer

        Example: [748,15,15,10] is a Network with 4 Layers with 748 in the input layer, 10 in 
        the output layer and 2 hidden layers with 15 neurons

        Args:
            sizez (list)
        """
        self.num_layers = len(sizez)
        self.sizez = sizez

        self.generate_biases()
        self.generate_weights()

    def generate_biases(self):
        """Generates the biases of each layer with standard normal distribution and shape of y*1
        """
        self.biases = [np.random.randn(y,1) for y in self.sizez[1:]]
        return self.biases

    def generate_weights(self):
        """Generates the weights of each layer with standard normal distribution and shape of y*x
        """
        self.weights = [np.random.randn(y,x) for x, y in zip(self.sizez[:-1],self.sizez[1:])]
        return self.weights

    def feed_forward(self,data):
        """Method feeds the input through each layer till the end of the network

        Args:
            data (ndarray): it is assumed that the input data is a (n,1) ndarray
            where n is the corresponding length of the input data 

        Returns:
            data(ndarray): also returns a (n,1)
        """
        for bias , weight in zip(self.biases,self.weights):
            data = sigmoid(np.dot(weight,data)+bias)
        return data

    def SGD(self,training_data,epochs,mini_batch_size, eta,test_data=None):
        """Train the neural network using mini-batch stochastic gradient descent

        Args:
            training_data ([tuple]): list of tuples "(x, y)" representing the training inputs and the desired outputs
            epochs (int): number of epochs
            mini_batch_size (int): size of the mini batch in each SGD step
            eta (float): learning rate
            test_data ([tuple], optional): "test_data" is provided then the network will be evaluated
            against the test data after each epoch , and partial progress printed out.
            This is useful for tracking progress , but slows things down substantially. Defaults to None.
        """
        if test_data:
            num_of_test = len(test_data)
            num_of_training= len(training_data)

        for epoch in range(epochs):
            random.shuffle(training_data)
            mini_batches = [training_data[k:k+mini_batch_size] for k in range(0,num_of_training,mini_batch_size)]
            for mini_batch in mini_batches:
                self.update_mini_batch(mini_batch,eta)

            if test_data :
                print("Epoch {0}: {1} / {2}, Test_Accuracy : {3} %".format(epoch, self.evaluate(test_data), num_of_test,self.evaluate(test_data)/num_of_test))
            else:
                print("Epoc {0} complete".format(epoch))
        
    def update_mini_batch(self,mini_batch,eta):
        """Update the network’s weights and biases by applying
            gradient descent using backpropagation to a single mini batch.
        Args:
            mini_batch ([tuples]): list of tuples (x, y) representing the training inputs and the desired outputs
            eta (float): learning rate.
        """
        #Nabla is the gradient of the loss function we are trying to optimize
        nabla_b = [np.zeros_like(bias) for bias in self.biases]
        nabla_w = [np.zeros_like(weight) for weight in self.weights]
        for x,y in mini_batch:
            #The delta_nabla is the new difference between the old and the new nabla generated form backprop ??
            delta_nabla_b , delta_nabla_w = self.backprop(x,y)
            nabla_b = [nb +dnb for nb, dnb in zip(nabla_b,delta_nabla_b)]
            nabla_w = [nw +dnw for nw, dnw in zip(nabla_w,delta_nabla_w)]
        self.weights = [w-eta/len(mini_batch)*nw for w,nw in zip(self.weights,nabla_w)]
        self.biases = [b-eta/len(mini_batch)*nb for b,nb in zip(self.biases,nabla_b)]

    def backprop(self,x,y):
        """Return a tuple (delta_nabla_b , delta_nabla_b) representing the
            gradient for the cost function C_x. nabla_b and 
            nabla_w  are layer -by-layer lists of numpy arrays , similar
            to self.biases and self.weights.

        Args:
            x (ndarray): training inputs
            y (ndarray): training outputs
        """
        delta_nabla_b = [np.zeros_like(bias) for bias in self.biases]
        delta_nabla_w = [np.zeros_like(weight) for weight in self.weights]
        #Same as the feed foward method , but here we will save each activation and the weigthed input z = w*a+b
        activation = x
        activations = [x] # list to store all the activations , layer by layer
        zs = [] # list to store all the z vectors , layer by layer
        
        for bias , weight in zip(self.biases,self.weights):
            z = np.dot(weight,activation)+bias
            zs.append(z)
            activation = sigmoid(z)
            activations.append(activation)
        delta = self.cost_derivative(activations[-1], y)*sigmoid_prime(zs[-1])
        delta_nabla_b[-1] = delta
        delta_nabla_w[-1] = np.dot(delta , activations[-2].T)


        for layer_index in range(2,self.num_layers):
            z = zs[-layer_index]
            delta = np.dot(self.weights[-layer_index+1].T, delta)*sigmoid_prime(z)
            delta_nabla_b[-layer_index] = delta
            delta_nabla_w[-layer_index] = np.dot(delta , activations[-layer_index-1].T)

        return (delta_nabla_b,delta_nabla_w)

    def evaluate(self,test_data):
        """Return the number of test inputs for which the neural
        network outputs the correct result. Note that the neural
        network’s output is assumed to be the index of whichever
        neuron in the final layer has the highest activation."""
        test_results = [(np.argmax(self.feed_forward (x)), y) for (x, y) in test_data ]
        return sum(int(x == y) for (x, y) in test_results )

    def cost_derivative(self,output_activations,y):
        """Return the vector of partial derivatives \partial C_x / \partial a for the output activations."""
        return (output_activations-y)

if __name__ == "__main__":
    net = Network([2,3,4])
    print("Biases",net.biases)
    print("weights",net.weights)
    print(net.feed_foward(np.array([3,2])))
        