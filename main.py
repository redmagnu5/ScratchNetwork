import numpy as np
import Networks

if __name__ == "__main__":

    # Initialize the single neuron neural network
    perceptron_network = Networks.Perceptron()
    neural_network = Networks.NeuralNetwork()

    print("Perceptron weights: ")
    print(perceptron_network.synaptic_weights)

    print("Neural network (1 layer) input weights: ")
    print(neural_network.wij)
    print("Neural network (1 layer) hidden layer weights: ")
    print(neural_network.wjk)


    training_inputs = np.array([[0,0,1],
                                [1,1,1],
                                [1,0,1],
                                [0,1,1]])

    training_outputs = np.array([[0,1,1,0]]).T

    # Train
    iterations = int(input("Train networks how many times? "))
    perceptron_network.train(training_inputs, training_outputs, iterations)
    neural_network.train(training_inputs, training_outputs, iterations)

    #Give new input
    print("new inputs to test: ")
    A = str(input("Input 1: "))
    B = str(input("Input 2: "))
    C = str(input("Input 3: "))

    userInput = np.array([A, B, C])

    #display output
    print("Perceptron prediction: ")
    print(perceptron_network.think(userInput))
    print("Neural network (1 layer) prediction: ")
    print(neural_network.think(userInput))
