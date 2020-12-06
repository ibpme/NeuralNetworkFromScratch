# NeuralNetworkFromScratch

A repository where I attempt to make a Simple Neural Network from scratch with no framework for classifying the MNSIT dataset.
All the code and algorithm to build the network is inside the network.py file under src directory, alongside load_mnist.py to load the dataset

### Run

| Run:         | Requirements:                     |
| ------------ | --------------------------------- |
| `py main.py` | `pip install -r requirements.txt` |

#### Running the script manually

Inside the main.py file you will find the code to customize the Neural Network
You can run it manually using any python shell

```python
import network
import load_mnist
training_data ,validation_data , test_data = load_mnist.load_preprocess_data()
net = network.Network([784,100,10])
```

### Datasets

- The current datasets are taken from tensorflow_datasets module.
- Input data is a **748 x 1** flattened array from a is a 28x28 grayscaled image (numbers-white, background-black)
- Currently looking for another dataset that are easy to unpack

## Curent accuracy

```s
Epoch 0: 8986 / 10000, Test_Accuracy : 89.86 %
Epoch 1: 9235 / 10000, Test_Accuracy : 92.35 %
Epoch 2: 9259 / 10000, Test_Accuracy : 92.59 %
Epoch 3: 9331 / 10000, Test_Accuracy : 93.31 %
.
.
.
Epoch 12: 9432 / 10000, Test_Accuracy : 94.32 %
Epoch 13: 9449 / 10000, Test_Accuracy : 94.49 %
Epoch 14: 9421 / 10000, Test_Accuracy : 94.21 %
```

## Curent Model

1. **Hidden Layer** : 1 (Width:30)
2. **Activation Function** : Sigmoid on every Layer
3. **Learning Rate Scheduler** : None
4. **Optimizer** : Stochastic Gradient Descent
5. **Cost Function** : Quadratic Loss
6. **Other information** :
   Constant Learning Rate : 3/Epoch
   Mini Batch : 100

## Exporting Model

The model can now be exported to a JSON file where it contains the weights and biases and other information about the model . You can use the **model_to_json** module to do so inside the main.py file.

## Things to improve

- The cost function is not an optimal one.
- The algorithms used are very slow
- Currently doesn't validate the data so overfitting is expected
- Learning rate is not optimized
