# ScratchNetwork

A basic neural network comparison built using only the python numpy library.

## Installation

Use the package manager [pip](https://pip.pypa.io/en/stable/) to install numpy.

```bash
        git clone https://github.com/redmagnu5/ScratchNetwork.git
```
```bash
        pip3 install numpy
```

## Usage
```bash
        cd ScratchNetwork
```
```bash
        python3 main.py
```

## Results
Currently there are two networks created in Networks.py:

```python3
        perceptron_network = Networks.Perceptron()
```
and,
```python3
        neural_network = Networks.NeuralNetwork()
```

Which are a basic 1 layer (for just the input) perceptron network and a
neural network with one hidden layer respectively.

The training set used for both networks are shown here:
```python3
        training_inputs = np.array([[0,0,1],
                                    [1,1,1],
                                    [1,0,1],
                                    [0,1,1]])

        training_outputs = np.array([[0,1,1,0]]).T
```

Where the training inputs are a 4x3 matrix vector, and the training outputs are
a 4x1 matrix vector. As you can see the first element of the row in the training
data corresponds to the same value in the output. A 0 in the first element, gives a 0
in the output, and a 1 in the first element gives a 1 in the output and so on.

The goal of the project is to see how well the models predict this pattern.

Results for a 2 different scenarios are shown below, with the first having an
expected value of 1, and the second with the value of 0. Different training
iterations are also considered.

1. Data set with expected value 1:
```python3
        np.array([1, 0, 0])
```
* 3 iterations:
```bash
        Perceptron prediction:
        [0.658589]
        Neural network (1 layer) prediction:
        [0.99895181]
```
* 50 iterations:
```bash
        Perceptron prediction:
        [0.97689888]
        Neural network (1 layer) prediction:
        [0.99895181]
```
* 1000 iterations:
```bash
        Perceptron prediction:
        [0.99929937]
        Neural network (1 layer) prediction:
        [0.99895181]
```


2. Data set with expected value 0:
```python3
        np.array([0, 1, 1])
```
* 3 iterations:
```bash
        Perceptron prediction:
        [0.46955296]
        Neural network (1 layer) prediction:
        [0.00234966]
```
* 50 iterations:
```bash
        Perceptron prediction:
        [0.13804798]
        Neural network (1 layer) prediction:
        [0.00234966]
```
* 1000 iterations:
```bash
        Perceptron prediction:
        [0.02575143]
        Neural network (1 layer) prediction:
        [0.00234966]
```

## Conclusion

Its clear that the neural network with 1 hidden layer got much more accurate
much faster, which is to be expected. The hidden layer adds a lot more weights
and biases to be considered for adjustment which corresponds to more possible
features of the data set.





## License
[MIT](https://choosealicense.com/licenses/mit/)
