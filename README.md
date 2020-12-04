# NeuralNetworkFromScratch

A repository where I attempt to make a Simple Convolutional Neural Network from scratch with no framework for classifying the MNSIT dataset.
All the code and algorithm to build the network is inside the network.py file under src directory, alongside load_mnist.py to load the dataset

### Run

| Run:         | Requirements:                     |
| ------------ | --------------------------------- |
| `py main.py` | `pip install -r requirements.txt` |

#### Running the script manually

Inside the main.py file you will find the code to customize the Neural Network
You can run it manually using any python shell

```
import network
import load_mnist
training_data ,validation_data , test_data = load_mnist.load_preprocess_data()
net = network.Network([784,100,10])
```

### Datasets

- The current datasets are taken from tensorflow_datasets module.
- Currently looking for another dataset that are easy to unpack

## Curent accuracy

Epoch 0: 856 / 10000, Test_Accuracy : 0.0856 %
Epoch 1: 827 / 10000, Test_Accuracy : 0.0827 %
Epoch 2: 814 / 10000, Test_Accuracy : 0.0814 %

## Things to improve

- The current accuracy of the model is very low and requires thousands of epoch to even reach a decent number.
- The cost function is not an optimal one.
- The algorithms used are very slow
