import numpy as np

class Network(object):

    def __init__(self, layers, learn_rate=0.25):
        self.layers = layers
        self.learn_rate = learn_rate


    def full_fwd_prop(self, training_data):
        init_layer = self.layers[0]
        self.layers[0].earlier_a = training_data
        init_layer.fwd_prop(training_data)
        for i in range(1,len(self.layers)):
            prev_layer, layer = self.layers[i-1], self.layers[i]
            self.layers[i].earlier_a = prev_layer.a
            layer.fwd_prop(prev_layer.a)
    
    def full_back_prop(self, training_label):
        last_layer = self.layers[-1]
        self.layers[-1].output_delta(training_label, last_layer.earlier_a)
        for i in range(2,len(self.layers)+1):
            self.layers[-i].back_prop(self.layers[-i+1].weights,self.layers[-i+1].delta,self.layers[-i].earlier_a)
        for i in range(1,len(self.layers)+1):
            current_layer = self.layers[-i]
            self.layers[-i].weights = current_layer.weights-self.learn_rate*(current_layer.nabla_w).T
            self.layers[-i].biases = current_layer.biases-self.learn_rate*current_layer.nabla_b


class FullyConnected(object):

    def __init__(self, neurons_in, neurons_out):
        
        self.neurons_in = neurons_in
        self.neurons_out = neurons_out
        self.weights = np.asarray(np.random.normal(loc=0.0,scale=1.0,size=(neurons_in,neurons_out)))
        self.biases = np.asarray(np.random.normal(loc=0.0,scale=1.0,size=(neurons_out,1)))
    
    def fwd_prop(self, prev_output):
        self.z = (np.dot(self.weights.T, prev_output)+self.biases)
        self.a = sigmoid(self.z)
    
    def output_delta(self, correct, earl_a):
        self.a_prime = sigmoid_prime(self.z)
        self.delta = (self.a-correct) * self.a_prime
        self.nabla_b = self.delta
        self.nabla_w = (np.dot(self.delta, earl_a.T))
    
    def back_prop(self, prev_weights, prev_delta, earl_a):
        self.a_prime = sigmoid_prime(self.z)
        self.delta = np.dot(prev_weights,prev_delta) * self.a_prime
        self.nabla_b = self.delta
        self.nabla_w = (np.dot(self.delta, earl_a.T))


def sigmoid(z):
    return 1.0/(1.0+np.exp(-z))

def sigmoid_prime(z):
    return sigmoid(z)*(1-sigmoid(z))