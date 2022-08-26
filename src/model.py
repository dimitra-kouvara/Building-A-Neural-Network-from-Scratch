"""
ITC6230:    ADVANCED MACHINE LEARNING
Final Project: Regression – Implementation of a Neural Network  
               with Adam optimizer from scratch

Part: Model

@author: Dimitra Kouvara
"""
# Importing the necessary libraries
import pandas as pd
import time
import numpy as np
import config
import activation
import initialization
import optimizers
import data_preprocessing
# Graphical representations
import matplotlib.pyplot as plt
from scipy import stats
import seaborn as sns

# Initializing variables
dropped_neurons_dict = dict()
problem = config.problem
loss_type = config.loss_type
preds_type = config.preds_type    
learning_rate = config.learning_rate    
parameters = config.parameters
layers = parameters['layers']
grid_search = config.grid_search
round_decimal = config.round_decimal
validation_flag = data_preprocessing.validation_flag
# L2 regularization
l2_flag = config.l2_flag
if l2_flag:
    l2_lambda = config.l2_lambda
# Dropout
dropout_flag = config.dropout_flag
if dropout_flag:
    dropoutThreshold = 1 - config.dropoutThreshold
# Adam optimizer
optimizer = config.optimizer
if optimizer == 'Adam':
    beta1 = config.beta1
    beta2 = config.beta2
    epsilon = config.epsilon
        
    # Initializing m and v as two python dictionaries with:
    # - keys: "W1", "db1", ..., "WL", "bL", where l = 1, 2, ... , L 
    # - values: zeros
    m, v = dict(), dict()
    for i in layers.keys():
        m[f'W{i}'] = 0
        m[f'b{i}'] = 0
        v[f'W{i}'] = 0
        v[f'b{i}'] = 0


def feed_forward(X, nodes, ini_type, act, flag, idx, cache):
    '''
    Implementation of the feed forward propagation

    Parameters
    ----------
    X : numpy ndarray
        The inputs
    nodes : int
        The number of nodes for each layer
    ini_type : str
        The initialization method for the weight matrices and bias vectors
    act : str
        the type of the activation function
    flag : int
        It activates (1) / disactivates (0) the initialization step of weights and biases
    idx : int
        It indicates each layer
    cache : python dictionary
        It contains the updated parameters of the model

    Returns
    -------
    A : numpy array
        The predicted output (the result of the activation function on z)
    W: numpy array
        The weights
    Z : numpy array
        The product sum of weights, inputs plus bias
    X : numpy array
        The inputs
    b : numpy array
        The biases 
    dropped_neurons: python dictionary
        A dictionary containing the dropped neurons (denoted as True) and the rest neurons (as False)

    '''
    global dropout_flag, dropoutThreshold
    dropped_neurons = {}
    X = np.array(X)
    input_nodes, nodes = X.shape[-1], nodes
    if flag == 1:
        if ini_type == "xavier":
            W, b = initialization.initialize_params_xa(input_nodes, nodes, act)
        elif ini_type == "xavier uniform":
            W, b = initialization.initialize_params_xaUni(input_nodes, nodes, act)
        elif ini_type == "he":
            W, b = initialization.initialize_params_he(input_nodes, nodes, act)
        elif ini_type == "random":
            W, b = initialization.initialize_params_rnd(input_nodes, nodes, act)
        elif ini_type == "zeros":
            W, b = initialization.initialize_params_zeros(input_nodes, nodes, act)
        Z = np.dot(X, W) + b
        A = activation.activation_fn(Z, act, derivative = False)
        if dropout_flag:
            # Step 1: initialize matrix d = np.random.rand(..., ...)
            # Step 2: convert entries of D to 0 or 1 (using dropout threshold)
            d = np.random.rand(*A.shape) < dropoutThreshold
            dropped_neurons[idx] = d
            # Step 3: shut down some neurons of A and scale the value of neurons that haven't been shut down
            A *= d
            A /= dropoutThreshold  
        return A, W, Z, X, b, dropped_neurons
    else:
        Z = np.dot(X, cache[f'W{idx}']) + cache[f'B{idx}']
        A = activation.activation_fn(Z, act, derivative = False)
        if dropout_flag:
            last_layer_idx = max(layers.keys())
            if idx != last_layer_idx:
                # Step 1: initialize matrix d = np.random.rand(..., ...)
                # Step 2: convert entries of D to 0 or 1 (using dropout threshold)
                d = np.random.rand(*A.shape) < dropoutThreshold
                dropped_neurons[idx] = d
                # Step 3: shut down some neurons of A and scale the value of neurons that haven't been shut down
                A *= d
                A /= dropoutThreshold  
        return A, cache[f'W{idx}'], Z, X, cache[f'B{idx}'], dropped_neurons


def backward(y, grads, cache, activations, l2_flag, dropped_neurons_dict):
    '''
    Implementation of the backpropagation

    Parameters
    ----------
    y : numpy ndarray
        The true labels of the dataset
    grads : python dictionary
        It contains the gradients for each parameter (weight and bias for every node)
    cache : python dictionary
        It contains the updated parameters of the model
    activations : python dictionary
        The type of the activation function for the neural network layers
    l2_flag: bool
        It activates (True) / disactivates (False) L2 regularization
    dropped_neurons_dict: python dictionary
        The dictionary of dropped neurons

    Returns
    -------
    None.

    '''
    global l2_lambda, dropout_flag, dropoutThreshold
    m = y.shape[0] # Number of values; used for averaging
    if problem == 'binary classification': # or problem == 'multi-class classification':
        if l2_flag == False and dropout_flag == False:
            last_layer_idx = max(layers.keys())
            # back prop through all dZs 
            for idx in reversed(range(1, last_layer_idx+1)):
                if idx == last_layer_idx:
                    grads[f'dZ{idx}'] = cache[f'A{idx}'] - y
                else:
                    # dZn = dZ(n+1) dot W(n+1) * inverse derivative of activation function of Layer n, with Zn as input 
                    grads[f'dZ{idx}'] = np.dot(grads[f'dZ{idx+1}'], cache[f'W{idx+1}'].T) *\
                                            activation.activation_fn(cache[f'Z{idx}'], activations[idx], derivative=True)
                grads[f'dW{idx}'] = 1./m * np.dot(cache[f'X{idx}'].T, grads[f'dZ{idx}']) 
                grads[f'db{idx}'] = 1./m * np.sum(grads[f'dZ{idx}'], axis=0, keepdims=True)
    
        if l2_flag:
            last_layer_idx = max(layers.keys())
            # back prop through all dZs 
            for idx in reversed(range(1, last_layer_idx+1)):
                if idx == last_layer_idx:
                    grads[f'dZ{idx}'] = cache[f'A{idx}'] - y
                else:
                    # dZn = dZ(n+1) dot W(n+1) * inverse derivative of activation function of Layer n, with Zn as input 
                    grads[f'dZ{idx}'] = np.dot(grads[f'dZ{idx+1}'], cache[f'W{idx+1}'].T) *\
                                            activation.activation_fn(cache[f'Z{idx}'], activations[idx], derivative=True)
                grads[f'dW{idx}'] = 1./m * np.dot(cache[f'X{idx}'].T, grads[f'dZ{idx}']) + (l2_lambda/m) * cache[f'W{idx}'] 
                grads[f'db{idx}'] = 1./m * np.sum(grads[f'dZ{idx}'], axis=0, keepdims=True)
              
        # Currently not supported due to "division by zero encountered in log" error in the computation of the loss function
        # if dropout_flag:
        #     last_layer_idx = max(layers.keys())
        #     first_layer_idx = min(layers.keys())
        #     # back prop through all dZs 
        #     for idx in reversed(range(1, last_layer_idx+1)):
        #         if idx == last_layer_idx:
        #             grads[f'dZ{idx}'] = cache[f'A{idx}'] - y
        #             grads[f'dW{idx}'] = 1./m * np.dot(cache[f'A{idx-1}'].T, grads[f'dZ{idx}'])
        #             grads[f'db{idx}'] = 1./m * np.sum(grads[f'dZ{idx}'], axis=0, keepdims=True)
        #             grads[f'dA{idx-1}'] = np.dot(grads[f'dZ{idx}'], cache[f'W{idx}'].T)
        #         else:
        #             # dZn = dZ(n+1) dot W(n+1) * inverse derivative of activation function of Layer n, with Zn as input 
        #             grads[f'dA{idx}'] *= dropped_neurons_dict[idx]
        #             grads[f'dA{idx}'] /= dropoutThreshold
                    
        #             grads[f'dZ{idx}'] = np.multiply(np.int64(cache[f'A{idx}'] > 0), grads[f'dA{idx}']) *\
        #                                     activation.activation_fn(cache[f'Z{idx}'], activations[idx], derivative=True)
        #             if idx != first_layer_idx:
        #                 grads[f'dW{idx}'] = 1./m * np.dot(cache[f'A{idx-1}'].T, grads[f'dZ{idx}'])
        #                 grads[f'dA{idx-1}'] = np.dot(grads[f'dZ{idx}'], cache[f'W{idx}'].T)
        #             else:
        #                 grads[f'dW{idx}'] = 1./m * np.dot(cache[f'X{idx}'].T, grads[f'dZ{idx}'])
        #             grads[f'db{idx}'] = 1./m * np.sum(grads[f'dZ{idx}'], axis=0, keepdims=True)
    
    if problem == 'regression':
        if loss_type == 'mse':
            if l2_flag == False and dropout_flag == False:
                last_layer_idx = max(layers.keys())
                # back prop through all dZs 
                for idx in reversed(range(1, last_layer_idx+1)):
                    if idx == last_layer_idx:
                        grads[f'dZ{idx}'] = (cache[f'A{idx}'] - y)
                    else:
                        # dZn = dZ(n+1) dot W(n+1) * inverse derivative of activation function of Layer n, with Zn as input 
                        grads[f'dZ{idx}'] = np.dot(grads[f'dZ{idx+1}'], cache[f'W{idx+1}'].T) *\
                                                activation.activation_fn(cache[f'Z{idx}'], activations[idx], derivative=True)
                    grads[f'dW{idx}'] = 2./m * np.dot(cache[f'X{idx}'].T, grads[f'dZ{idx}']) 
                    grads[f'db{idx}'] = 2./m * np.sum(grads[f'dZ{idx}'], axis=0, keepdims=True)
        
            if l2_flag:
                last_layer_idx = max(layers.keys())
                # back prop through all dZs 
                for idx in reversed(range(1, last_layer_idx+1)):
                    if idx == last_layer_idx:
                        grads[f'dZ{idx}'] = (cache[f'A{idx}'] - y)
                    else:
                        # dZn = dZ(n+1) dot W(n+1) * inverse derivative of activation function of Layer n, with Zn as input 
                        grads[f'dZ{idx}'] = np.dot(grads[f'dZ{idx+1}'], cache[f'W{idx+1}'].T) *\
                                                activation.activation_fn(cache[f'Z{idx}'], activations[idx], derivative=True)
                    grads[f'dW{idx}'] = 2./m * np.dot(cache[f'X{idx}'].T, grads[f'dZ{idx}']) + (l2_lambda/m) * cache[f'W{idx}'] 
                    grads[f'db{idx}'] = 2./m * np.sum(grads[f'dZ{idx}'], axis=0, keepdims=True)
                    
            if dropout_flag:
                last_layer_idx = max(layers.keys())
                first_layer_idx = min(layers.keys())
                # back prop through all dZs 
                for idx in reversed(range(1, last_layer_idx+1)):
                    if idx == last_layer_idx:
                        grads[f'dZ{idx}'] = (cache[f'A{idx}'] - y)
                        grads[f'dW{idx}'] = 2./m * np.dot(cache[f'A{idx-1}'].T, grads[f'dZ{idx}'])
                        grads[f'db{idx}'] = 2./m * np.sum(grads[f'dZ{idx}'], axis=0, keepdims=True)
                        grads[f'dA{idx-1}'] = np.dot(grads[f'dZ{idx}'], cache[f'W{idx}'].T)
                    else:
                        # dZn = dZ(n+1) dot W(n+1) * inverse derivative of activation function of Layer n, with Zn as input 
                        grads[f'dA{idx}'] *= dropped_neurons_dict[idx]
                        grads[f'dA{idx}'] /= dropoutThreshold
                        
                        grads[f'dZ{idx}'] = np.multiply(np.int64(cache[f'A{idx}'] > 0), grads[f'dA{idx}']) *\
                                                activation.activation_fn(cache[f'Z{idx}'], activations[idx], derivative=True)
                        if idx != first_layer_idx:
                            grads[f'dW{idx}'] = 2./m * np.dot(cache[f'A{idx-1}'].T, grads[f'dZ{idx}'])
                            grads[f'dA{idx-1}'] = np.dot(grads[f'dZ{idx}'], cache[f'W{idx}'].T)
                        else:
                            grads[f'dW{idx}'] = 2./m * np.dot(cache[f'X{idx}'].T, grads[f'dZ{idx}'])
                        grads[f'db{idx}'] = 2./m * np.sum(grads[f'dZ{idx}'], axis=0, keepdims=True)
    
        if loss_type == 'rmse':
            if l2_flag == False and dropout_flag == False:
                last_layer_idx = max(layers.keys())
                # back prop through all dZs 
                for idx in reversed(range(1, last_layer_idx+1)):
                    if idx == last_layer_idx:
                        grads[f'dZ{idx}'] = (cache[f'A{idx}'] - y) / np.abs(cache[f'A{idx}'] - y)
                    else:
                        # dZn = dZ(n+1) dot W(n+1) * inverse derivative of activation function of Layer n, with Zn as input 
                        grads[f'dZ{idx}'] = np.dot(grads[f'dZ{idx+1}'], cache[f'W{idx+1}'].T) *\
                                                activation.activation_fn(cache[f'Z{idx}'], activations[idx], derivative=True)
                    grads[f'dW{idx}'] = 1./np.sqrt(m) * np.dot(cache[f'X{idx}'].T, grads[f'dZ{idx}']) 
                    grads[f'db{idx}'] = 1./np.sqrt(m) * np.sum(grads[f'dZ{idx}'], axis=0, keepdims=True)
        
            if l2_flag:
                last_layer_idx = max(layers.keys())
                # back prop through all dZs 
                for idx in reversed(range(1, last_layer_idx+1)):
                    if idx == last_layer_idx:
                        grads[f'dZ{idx}'] = (cache[f'A{idx}'] - y) / np.abs(cache[f'A{idx}'] - y)
                    else:
                        # dZn = dZ(n+1) dot W(n+1) * inverse derivative of activation function of Layer n, with Zn as input 
                        grads[f'dZ{idx}'] = np.dot(grads[f'dZ{idx+1}'], cache[f'W{idx+1}'].T) *\
                                                activation.activation_fn(cache[f'Z{idx}'], activations[idx], derivative=True)
                    grads[f'dW{idx}'] = 1./np.sqrt(m) * np.dot(cache[f'X{idx}'].T, grads[f'dZ{idx}']) + (l2_lambda/m) * cache[f'W{idx}'] 
                    grads[f'db{idx}'] = 1./np.sqrt(m) * np.sum(grads[f'dZ{idx}'], axis=0, keepdims=True)
                    
            if dropout_flag:
                last_layer_idx = max(layers.keys())
                first_layer_idx = min(layers.keys())
                # back prop through all dZs 
                for idx in reversed(range(1, last_layer_idx+1)):
                    if idx == last_layer_idx:
                        grads[f'dZ{idx}'] = (cache[f'A{idx}'] - y) / np.abs(cache[f'A{idx}'] - y)
                        grads[f'dW{idx}'] = 1./np.sqrt(m) * np.dot(cache[f'A{idx-1}'].T, grads[f'dZ{idx}'])
                        grads[f'db{idx}'] = 1./np.sqrt(m) * np.sum(grads[f'dZ{idx}'], axis=0, keepdims=True)
                        grads[f'dA{idx-1}'] = np.dot(grads[f'dZ{idx}'], cache[f'W{idx}'].T)
                    else:
                        # dZn = dZ(n+1) dot W(n+1) * inverse derivative of activation function of Layer n, with Zn as input 
                        grads[f'dA{idx}'] *= dropped_neurons_dict[idx]
                        grads[f'dA{idx}'] /= dropoutThreshold
                        
                        grads[f'dZ{idx}'] = np.multiply(np.int64(cache[f'A{idx}'] > 0), grads[f'dA{idx}']) *\
                                                activation.activation_fn(cache[f'Z{idx}'], activations[idx], derivative=True)
                        if idx != first_layer_idx:
                            grads[f'dW{idx}'] = 1./np.sqrt(m) * np.dot(cache[f'A{idx-1}'].T, grads[f'dZ{idx}'])
                            grads[f'dA{idx-1}'] = np.dot(grads[f'dZ{idx}'], cache[f'W{idx}'].T)
                        else:
                            grads[f'dW{idx}'] = 1./np.sqrt(m) * np.dot(cache[f'X{idx}'].T, grads[f'dZ{idx}'])
                        grads[f'db{idx}'] = 1./np.sqrt(m) * np.sum(grads[f'dZ{idx}'], axis=0, keepdims=True)
        
        if loss_type == 'mae':
            if l2_flag == False and dropout_flag == False:
                last_layer_idx = max(layers.keys())
                # back prop through all dZs 
                for idx in reversed(range(1, last_layer_idx+1)):
                    if idx == last_layer_idx:
                        grads[f'dZ{idx}'] = (cache[f'A{idx}'] - y) / np.abs(cache[f'A{idx}'] - y)
                    else:
                        # dZn = dZ(n+1) dot W(n+1) * inverse derivative of activation function of Layer n, with Zn as input 
                        grads[f'dZ{idx}'] = np.dot(grads[f'dZ{idx+1}'], cache[f'W{idx+1}'].T) *\
                                                activation.activation_fn(cache[f'Z{idx}'], activations[idx], derivative=True)
                    grads[f'dW{idx}'] = 1./m * np.dot(cache[f'X{idx}'].T, grads[f'dZ{idx}']) 
                    grads[f'db{idx}'] = 1./m * np.sum(grads[f'dZ{idx}'], axis=0, keepdims=True)
        
            if l2_flag:
                last_layer_idx = max(layers.keys())
                # back prop through all dZs 
                for idx in reversed(range(1, last_layer_idx+1)):
                    if idx == last_layer_idx:
                        grads[f'dZ{idx}'] = (cache[f'A{idx}'] - y) / np.abs(cache[f'A{idx}'] - y)
                    else:
                        # dZn = dZ(n+1) dot W(n+1) * inverse derivative of activation function of Layer n, with Zn as input 
                        grads[f'dZ{idx}'] = np.dot(grads[f'dZ{idx+1}'], cache[f'W{idx+1}'].T) *\
                                                activation.activation_fn(cache[f'Z{idx}'], activations[idx], derivative=True)
                    grads[f'dW{idx}'] = 1./m * np.dot(cache[f'X{idx}'].T, grads[f'dZ{idx}']) + (l2_lambda/m) * cache[f'W{idx}'] 
                    grads[f'db{idx}'] = 1./m * np.sum(grads[f'dZ{idx}'], axis=0, keepdims=True)
                    
            if dropout_flag:
                last_layer_idx = max(layers.keys())
                first_layer_idx = min(layers.keys())
                # back prop through all dZs 
                for idx in reversed(range(1, last_layer_idx+1)):
                    if idx == last_layer_idx:
                        grads[f'dZ{idx}'] = (cache[f'A{idx}'] - y) / np.abs(cache[f'A{idx}'] - y)
                        grads[f'dW{idx}'] = 1./m * np.dot(cache[f'A{idx-1}'].T, grads[f'dZ{idx}'])
                        grads[f'db{idx}'] = 1./m * np.sum(grads[f'dZ{idx}'], axis=0, keepdims=True)
                        grads[f'dA{idx-1}'] = np.dot(grads[f'dZ{idx}'], cache[f'W{idx}'].T)
                    else:
                        # dZn = dZ(n+1) dot W(n+1) * inverse derivative of activation function of Layer n, with Zn as input 
                        grads[f'dA{idx}'] *= dropped_neurons_dict[idx]
                        grads[f'dA{idx}'] /= dropoutThreshold
                        
                        grads[f'dZ{idx}'] = np.multiply(np.int64(cache[f'A{idx}'] > 0), grads[f'dA{idx}']) *\
                                                activation.activation_fn(cache[f'Z{idx}'], activations[idx], derivative=True)
                        if idx != first_layer_idx:
                            grads[f'dW{idx}'] = 1./m * np.dot(cache[f'A{idx-1}'].T, grads[f'dZ{idx}'])
                            grads[f'dA{idx-1}'] = np.dot(grads[f'dZ{idx}'], cache[f'W{idx}'].T)
                        else:
                            grads[f'dW{idx}'] = 1./m * np.dot(cache[f'X{idx}'].T, grads[f'dZ{idx}'])
                        grads[f'db{idx}'] = 1./m * np.sum(grads[f'dZ{idx}'], axis=0, keepdims=True)
        
        if loss_type == 'rae':
            if l2_flag == False and dropout_flag == False:
                last_layer_idx = max(layers.keys())
                # back prop through all dZs 
                for idx in reversed(range(1, last_layer_idx+1)):
                    if idx == last_layer_idx:
                        grads[f'dZ{idx}'] = (cache[f'A{idx}'] - y) / (np.abs(cache[f'A{idx}'] - y))*(np.abs(y-np.mean(y)))
                    else:
                        # dZn = dZ(n+1) dot W(n+1) * inverse derivative of activation function of Layer n, with Zn as input 
                        grads[f'dZ{idx}'] = np.dot(grads[f'dZ{idx+1}'], cache[f'W{idx+1}'].T) *\
                                                activation.activation_fn(cache[f'Z{idx}'], activations[idx], derivative=True)
                    grads[f'dW{idx}'] = 1./m * np.dot(cache[f'X{idx}'].T, grads[f'dZ{idx}']) 
                    grads[f'db{idx}'] = 1./m * np.sum(grads[f'dZ{idx}'], axis=0, keepdims=True)
        
            if l2_flag:
                last_layer_idx = max(layers.keys())
                # back prop through all dZs 
                for idx in reversed(range(1, last_layer_idx+1)):
                    if idx == last_layer_idx:
                        grads[f'dZ{idx}'] = (cache[f'A{idx}'] - y) / (np.abs(cache[f'A{idx}'] - y))*(np.abs(y-np.mean(y)))
                    else:
                        # dZn = dZ(n+1) dot W(n+1) * inverse derivative of activation function of Layer n, with Zn as input 
                        grads[f'dZ{idx}'] = np.dot(grads[f'dZ{idx+1}'], cache[f'W{idx+1}'].T) *\
                                                activation.activation_fn(cache[f'Z{idx}'], activations[idx], derivative=True)
                    grads[f'dW{idx}'] = 1./m * np.dot(cache[f'X{idx}'].T, grads[f'dZ{idx}']) + (l2_lambda/m) * cache[f'W{idx}'] 
                    grads[f'db{idx}'] = 1./m * np.sum(grads[f'dZ{idx}'], axis=0, keepdims=True)
                    
            if dropout_flag:
                last_layer_idx = max(layers.keys())
                first_layer_idx = min(layers.keys())
                # back prop through all dZs 
                for idx in reversed(range(1, last_layer_idx+1)):
                    if idx == last_layer_idx:
                        grads[f'dZ{idx}'] = (cache[f'A{idx}'] - y) / (np.abs(cache[f'A{idx}'] - y))*(np.abs(y-np.mean(y)))
                        grads[f'dW{idx}'] = 1./m * np.dot(cache[f'A{idx-1}'].T, grads[f'dZ{idx}'])
                        grads[f'db{idx}'] = 1./m * np.sum(grads[f'dZ{idx}'], axis=0, keepdims=True)
                        grads[f'dA{idx-1}'] = np.dot(grads[f'dZ{idx}'], cache[f'W{idx}'].T)
                    else:
                        # dZn = dZ(n+1) dot W(n+1) * inverse derivative of activation function of Layer n, with Zn as input 
                        grads[f'dA{idx}'] *= dropped_neurons_dict[idx]
                        grads[f'dA{idx}'] /= dropoutThreshold
                        
                        grads[f'dZ{idx}'] = np.multiply(np.int64(cache[f'A{idx}'] > 0), grads[f'dA{idx}']) *\
                                                activation.activation_fn(cache[f'Z{idx}'], activations[idx], derivative=True)
                        if idx != first_layer_idx:
                            grads[f'dW{idx}'] = 1./m * np.dot(cache[f'A{idx-1}'].T, grads[f'dZ{idx}'])
                            grads[f'dA{idx-1}'] = np.dot(grads[f'dZ{idx}'], cache[f'W{idx}'].T)
                        else:
                            grads[f'dW{idx}'] = 1./m * np.dot(cache[f'X{idx}'].T, grads[f'dZ{idx}'])
                        grads[f'db{idx}'] = 1./m * np.sum(grads[f'dZ{idx}'], axis=0, keepdims=True)
    
    

        
def create_batches(x, y, batch_size):
    '''
    Extract batches of data for training the neural network

    Parameters
    ----------
    x : numpy ndarray
        The features of the dataset
    y : numpy ndarray
        The true labels of the dataset
    batch_size : int
        The number of samples to work through before updating the internal model parameters

    Returns
    -------
    batches : python list
        The batches of data (combining x and y arrays)

    '''
    m = x.shape[0]
    num_batches = m / batch_size
    batches = []
    for i in range(int(num_batches+1)):
        batch_x = x[i*batch_size:(i+1)*batch_size]
        batch_y = y[i*batch_size:(i+1)*batch_size]
        batches.append((batch_x, batch_y))
            
    # without this, batch sizes that are perfectly divisible will create an 
    # empty array at index -1
    if m % batch_size == 0:
        batches.pop(-1)

    return batches


def compute_loss(y, y_hat, l2_flag, cache):
    '''
    Compute the loss function for a vector of predictions
    
    Parameters
    ----------
    y : numpy array
        An array of the y labels
    y_hat : numpy array
        An array of the yhat predictions

    Returns
    -------
    L : float
        The result of the loss function

    '''
    global loss_type, epoch_loss, problem, l2_lambda
    m = y.shape[0]
    if problem == 'multi-class classification':
        if loss_type == 'categorical_cross-entropy':
            L = -1./m * np.sum(y * np.log(y_hat))
        # elif loss_type == 'kullback':
        #     L = y * np.log(y / y_hat)
    elif problem == 'binary classification':
        if loss_type == 'binary_cross-entropy':
            L = -1./m * np.sum(y * np.log(y_hat) + (1 - y) * np.log(1 - y_hat))
        # elif loss_type == 'hinge':
        #     L = np.maximum(0, 1 - y * y_hat)
    # Regression problem        
    else:                             
        if loss_type == 'mse':
            L = 1./m * np.sum((y - y_hat)**2)
        elif loss_type == 'rmse':
            L = np.sqrt(1./m * np.sum((y - y_hat)**2))
        elif loss_type == 'mae':
            L = 1./m * np.sum(np.abs(y - y_hat))    
        elif loss_type == 'rae':
            y_mean = np.mean(y)
            L = 1./m * np.sum(np.abs(y - y_hat)) / np.sum(np.abs(y - y_mean))
    if l2_flag: 
        weights_squaredSum = 0
        for key, value in cache.items():   # iter on both keys and values
            if key.startswith('W'):
                weights_squaredSum += np.sum(np.square(value))
        L_regularization = l2_lambda / (m * 2) * weights_squaredSum
        L = L + L_regularization
    return L

    


def cache_feedforward(x, layers, ini_type, activations, cache, flag):
    '''
    Cache values for weights, biases, inputs, outputs and dropped neurons in dictionaries

    Parameters
    ----------
    x : numpy ndarray
        The inputs
    layers : python dictionary
        It contains information about the NN architecture: the different layers (keys) along with their nodes (values)
    ini_type : str
        The initialization method for the weight matrices and bias vectors
    activations : python dictionary
        The type of the activation function for the neural network layers
    cache : python dictionary
        It contains the updated parameters of the model
    flag : int
        It activates (1) / disactivates (0) the initialization step of weights and biases

    Returns
    -------
    x : numpy ndarray
        The predicted output (the result of the activation function on z)
    dropped_neurons_dict: python dictionary
         The dictionary of dropped neurons
    
    '''
    global dropped_neurons_dict
    for idx, layer in layers.items():
        activation = activations[idx]
        x, W, Z, X, b, dropped_neurons = feed_forward(x, layer, ini_type, activation, flag, idx, cache)
        cache[f'W{idx}'] = W
        cache[f'Z{idx}'] = Z
        cache[f'A{idx}'] = x
        cache[f'X{idx}'] = X
        cache[f'B{idx}'] = b
        dropped_neurons_dict.update(dropped_neurons)
    return cache[f'A{idx}'], dropped_neurons_dict


def update_parameters(steps, grads, cache, epochs):
    '''
    Update parameters using either the gradient descent or the Adam optimizer

    Parameters
    ----------
    steps : int
        It indicates the timestep (iteration)
    grads : python dictionary
        It contains the gradients for each parameter (weight and bias for every node)
    cache : python dictionary
        It contains the updated parameters of the model
    epochs : int
        The number times that the learning algorithm will work through the entire training dataset

    Returns
    -------
    None.

    '''
    global optimizer, learning_rate, layers
    for idx in layers.keys():
        if optimizer is None:
            optimizers.gradient_descent(idx, cache, grads, learning_rate)
        else:
            global m, v, beta1, beta2, epsilon
            optimizers.ADAM_optimize(idx, grads, steps, cache, m, v, learning_rate, beta1, beta2, epsilon)    


def evaluate_performance(x, y, layers, ini_type, activations, cache, flag):
    '''
    Evaluate the classifier performance with various metrics

    Parameters
    ----------
    x : numpy ndarray
        The features of the dataset
    y : numpy ndarray
        The true labels of the dataset
    layers : python dictionary
        It contains information about the NN architecture: the different layers (keys) along with their nodes (values)
    ini_type : str
        The initialization method for the weight matrices and bias vectors
    activations : python dictionary
        The type of the activation function for the neural network layers
    cache : python dictionary
        It contains the updated parameters of the model
    flag : int
        It activates (1) / disactivates (0) the initialization step of weights and biases

    Returns
    -------
    float
        The value of evaluation metrics at each epoch

    '''
    global preds_type, y_pred, problem
    preds_x, dropped_neurons_dict = cache_feedforward(x, layers, ini_type, activations, cache, flag)

    if problem == 'multi-class classification':
        y_values = y.flatten()
        preds_x = np.argmax(preds_x, axis=1)
        preds_y = np.argmax(y, axis=1)        
        accu = np.mean(np.equal(preds_x, preds_y))
        return accu        
    elif problem == 'binary classification':
        y_values = y.flatten()
        preds_x = preds_x.round(decimals=0)
        accu = (preds_x == y).all(axis=(1)).mean()
        return accu
    # Regression Problem
    else:
        y_values = y.flatten()
        y_pred = preds_x.flatten()
        if preds_type == 'RMSE':
            # Computing the RMSE
            rmse = np.sqrt(np.mean((y - preds_x)**2)) 
            return rmse
        elif preds_type == 'R^2':
            # Computing the R squared error
            correlation_matrix = np.corrcoef(y_pred, y_values)
            correlation_xy = correlation_matrix[0, 1]
            r_squared = correlation_xy**2
            return r_squared
        elif preds_type == 'MAE':
            mae = np.mean(np.abs(y - preds_x)) 
            return mae


def fit(x_train, y_train, x_test, y_test, epochs, batch_size, layers, ini_type, activations, cache, grads): 
    '''
    Train the model for a fixed number of epochs 

    Parameters
    ----------
    x_train : numpy ndarray
        The features of the training data
    y_train : numpy ndarray
        The true labels of the training data
    x_test : numpy ndarray
        The features of the test data
    y_test : numpy ndarray
        The true labels of the test data
    epochs : int
        The number times that the learning algorithm will work through the entire training dataset
    batch_size : int
        The number of samples to work through before updating the internal model parameters
    layers : python dictionary
        It contains information about the NN architecture: the different layers (keys) along with their nodes (values)
    ini_type : str
        The initialization method for the weight matrices and bias vectors
    activations : python dictionary
        The type of the activation function for the neural network layers
    cache : python dictionary
        It contains the updated parameters of the model
    grads : python dictionary
        It contains the gradients for each parameter (weight and bias for every node)

    Returns
    -------
    None.
        
    '''
    global train_accuracy, test_accuracy, losses, grid_search, y_pred
    # Start the time for fitting
    start_time = time.time()
    
    train_accuracy = []
    test_accuracy = []
    losses = []
    flag = 1
    for epoch in range(1, epochs+1):
        print()
        print(f'Epoch {epoch}')
        # Creating mini-batches 
        batches = create_batches(x_train, y_train, batch_size)
        epoch_loss = []
        steps = 0
        for x, y in batches:
            steps += 1
            # forward_propagation
            predicted_output, dropped_neurons_dict = cache_feedforward(x, layers, ini_type, activations, cache, flag)
            flag = 0
            # loss
            loss = compute_loss(y, predicted_output, l2_flag, cache)
            epoch_loss.append(loss) 
            # back propagation
            backward(y, grads, cache, activations, l2_flag, dropped_neurons_dict)
            # update parameters
            update_parameters(steps, grads, cache, epochs)   
        loss = sum(epoch_loss) / len(epoch_loss)
        losses.append(loss)
        
        # Calculating accuracy on the training set
        train_acc = evaluate_performance(x_train, y_train, layers, ini_type, activations, cache, flag)
        train_accuracy.append(train_acc)

        # Calculating accuracy on the test set
        test_acc = evaluate_performance(x_test, y_test, layers, ini_type, activations, cache, flag)
        test_accuracy.append(test_acc)
        
        print(f'Loss: {loss} Train {preds_type}: {train_acc} Test {preds_type}: {test_acc}')
    
    # End the time
    exec_time = time.time() - start_time
    print(f'--- Execution time: {exec_time} seconds ---')
    # Saving the architecture of the model along with the results in a dictionary
    model_res = {'layers': len(layers), 'epochs': epochs, 'batch size': batch_size, 'losses': losses, 'train accuracy': train_accuracy, 'test_accuracy': test_accuracy, 'execution time': exec_time, 'cache': cache}
    # Appending the previous info on grid_search list
    grid_search.append(model_res)
    
    # Statistics for the regression problem
    if problem == 'regression':
        y_testFlatten = y_test.flatten()
        # round values and convert to integers
        y_pred = np.round(y_pred, round_decimal)
        res = stats.linregress(y_testFlatten, y_pred)
        print(res)
        print(f"R-squared: {res.rvalue**2:.6f}")
    
def predict(y_hat, y, cache, test_flag):
    '''
    Predict the y labels and gather all info: y_hat (the predicted output), 
    y (the true labels) and metric for the evaluation of model's performance in a dataframe
        
    Parameters
    ----------
    y_hat : numpy ndarray
        The predicted outputs of the test/validation (real-world) data 
    y : numpy ndarray
        The true labels of the test/validation (real-world) data
    cache : python dictionary
        It contains the updated parameters of the model
    test_flag : bool
        It activates the testing (True) / activates the validation (real-world) data case (False)

    Returns
    -------
    predict_df: pandas.core.frame.DataFrame
        A dataframe containing the predicted outputs along with the true 
        labels and their difference

    '''
    global preds_type, problem
    if problem == 'multi-class classification' or problem == 'binary classification':
        if test_flag:
            # Get the last layer's label
            last_layer_label = str(list(layers)[-1])
            cache_key = ''.join(['A', last_layer_label])
            preds_x = cache[cache_key]
            if problem == 'multi-class classification':
                y_pred = np.argmax(preds_x, axis=1)
                y_values = np.argmax(y, axis=1)
                metric = np.equal(y_pred, y_values)
                # Getting the absolute difference between y_hat and y
                predict_dict = {'y': y_values, 'y_hat': y_pred, 'prediction': metric}
                predict_df = pd.DataFrame(predict_dict)
                print('Percentage of correct and wrong predictions: ')
                print(predict_df['prediction'].value_counts(normalize=True)*100)
            if problem == 'binary classification':
                y_values = y.flatten()
                preds_x = preds_x.flatten()
                y_pred = preds_x.round(decimals=0)
                metric = np.equal(y_pred, y_values)
                # Getting the absolute difference between y_hat and y
                predict_dict = {'y': y_values, 'y_hat': y_pred, 'prediction': metric}
                predict_df = pd.DataFrame(predict_dict)
                # converting 'y_hat' from float to int
                predict_df['y_hat'] = predict_df['y_hat'].astype(int)
                print('Percentage of correct and wrong predictions: ')
                print(predict_df['prediction'].value_counts(normalize=True)*100)
                # Create the confusion matrix
                create_confusion_matrix(predict_df)
            print(predict_df)
        else:
            preds_x = y_hat
        if validation_flag:
            if problem == 'multi-class classification':
                y_pred = np.argmax(preds_x, axis=1)
                y_values = np.argmax(y, axis=1)
                metric = np.equal(y_pred, y_values)
                # Getting the absolute difference between y_hat and y
                predict_dict = {'y': y_values, 'y_hat': y_pred, 'prediction': metric}
                predict_df = pd.DataFrame(predict_dict)
                print('Percentage of correct and wrong predictions: ')
                print(predict_df['prediction'].value_counts(normalize=True)*100)
            if problem == 'binary classification':
                y_values = y.flatten()
                preds_x = preds_x.flatten()
                y_pred = preds_x.round(decimals=0)
                metric = np.equal(y_pred, y_values)
                # Getting the absolute difference between y_hat and y
                predict_dict = {'y': y_values, 'y_hat': y_pred, 'prediction': metric}
                predict_df = pd.DataFrame(predict_dict)
                # converting 'y_hat' from float to int
                predict_df['y_hat'] = predict_df['y_hat'].astype(int)
                print('Percentage of correct and wrong predictions: ')
                print(predict_df['prediction'].value_counts(normalize=True)*100)
                create_confusion_matrix(predict_df)
    # Regression Problem
    else:
        if test_flag:
            # Get the last layer's label
            last_layer_label = str(list(layers)[-1])
            cache_key = ''.join(['A', last_layer_label])
            # Round the y labels
            preds_x = cache[cache_key].round(decimals=round_decimal)
            y_pred = preds_x.flatten()
            y_values = y.flatten()
            # Getting the absolute difference between y_hat and y
            predict_dict = {'y': y_values, 'y_hat': y_pred}
            predict_df = pd.DataFrame(predict_dict) 
            predict_df['abs_diff'] = (predict_df['y'] - predict_df['y_hat']).abs()
            condition = (predict_df['y'] == predict_df['y_hat'])
            predict_df['prediction'] = np.where(condition, True, False)
            print('Percentage of correct and wrong predictions: ')
            print(predict_df['prediction'].value_counts(normalize=True)*100)
            # if two class problem (could also be treated as binary classification problem)
            if predict_df['y'].nunique() == 2: 
                create_confusion_matrix(predict_df)
            print(predict_df)
        else:
            preds_x = y_hat.round(decimals=round_decimal)
        if validation_flag:
            y_pred = preds_x.flatten()
            y_values = y.flatten()
            # Getting the absolute difference between y_hat and y
            predict_dict = {'y': y_values, 'y_hat': y_pred}
            predict_df = pd.DataFrame(predict_dict) 
            predict_df['abs_diff'] = (predict_df['y'] - predict_df['y_hat']).abs()
            condition = (predict_df['y'] == predict_df['y_hat'])
            predict_df['prediction'] = np.where(condition, True, False)
            print('Percentage of correct and wrong predictions: ')
            print(predict_df['prediction'].value_counts(normalize=True)*100)
            # if two class problem (could also be treated as binary classification problem)
            if predict_df['y'].nunique() == 2: 
                create_confusion_matrix(predict_df)
            print(predict_df)
    # Save the labels of the validation data in csv file
    if test_flag != False:
        validation_labels = {'true labels': y_values, 'predicted labels': y_pred}
        validation_df = pd.DataFrame(validation_labels)
        validation_df['predicted labels'] = validation_df['predicted labels'].astype(int)
        validation_df.to_csv('validation_data_Predictedlabels.csv') 
        print(validation_df)

 
def create_confusion_matrix(predict_df):
    '''
    Create a confusion matrix with TP, TN, FP & FN (in case of two classes) 
    
    Parameters
    ----------
    predict_df : pandas.core.frame.DataFrame
        A dataframe containing the predicted outputs along with the true 
        labels and their difference

    Returns
    -------
    None.

    '''
    TP = predict_df.apply(lambda x : True
                if x['prediction'] == True and x['y'] == 1 else False, axis = 1)
    # TP Count number of True in the series
    TP_counts = len(TP[TP == True].index)
    
    TN = predict_df.apply(lambda x : True
        if x['prediction'] == True and x['y'] == 0 else False, axis = 1)
    # TN Count number of True in the series
    TN_counts = len(TN[TN == True].index)
    
    FP = predict_df.apply(lambda x : True
                if x['prediction'] == False and x['y'] == 1 else False, axis = 1)
    # FP Count number of True in the series
    FP_counts = len(FP[FP == True].index)
    
    FN = predict_df.apply(lambda x : True
        if x['prediction'] == False and x['y'] == 0 else False, axis = 1)
    # FN Count number of True in the series
    FN_counts = len(FN[FN == True].index)

    # Get the confusion matrix
    cf_matrix = np.array([[TP_counts, FP_counts], [FN_counts, TN_counts]])        
    # Heatmap
    plt.figure(figsize=(5.5, 4))
    sns.heatmap(cf_matrix, annot=True, cmap="YlGnBu")
    plt.title('\nConfusion matrix')
    plt.ylabel('Actual class')
    plt.xlabel('Predicted class')
    plt.show()    

   
def treat_validation_data(x, y, cache, ini_type, activations):
    '''
    Compute the predicted outputs of the validation data by applying feed forward
    propagation. Note that the weights and biases used for the computation are the 
    result of the model fitting on the training data.

    Parameters
    ----------
    x : numpy ndarray
        The features of the validation data
    y : numpy ndarray
        The true labels of the validation data
    cache : python dictionary
        It contains the updated parameters of the model
    ini_type : str
        The initialization method for the weight matrices and bias vectors
    activations : python dictionary
        The type of the activation function for the neural network layers
        
    Returns
    -------
    python function
        The output of predict: A dataframe containing the predicted outputs 
        along with the true labels and their difference

    '''
    global preds_type, y_pred, problem
    # Keep the weights and biases and run the feed forward propagation
    y_hat, dropped_neurons_dict = cache_feedforward(x, layers, ini_type, activations, cache, flag=0)
    predict(y_hat, y, cache, test_flag = False)

    
def best_model(): 
    '''
    Returns
    -------
    best_model : python dictionary
        -- Contains the characteristics of the best model according to the 
        selected evaluation metrics(layers, epochs, batch size, loss and 
        evaluation metrics values for the training and the test data)  
    '''
    global train_accuracy, test_accuracy, losses, grid_search, problem
    # Sorting the list based on the testing accuracy
    if problem == 'multi-class classification' or problem == 'binary classification':
         best_model = sorted(grid_search, key=lambda d: d['test_accuracy'][-1])[-1]
    # Regression Problem
    else:
        if preds_type == 'RMSE' or preds_type == 'MAE':
            best_model = sorted(grid_search, key=lambda d: d['test_accuracy'][-1])[0]
        elif preds_type == 'R^2':
            best_model = sorted(grid_search, key=lambda d: d['test_accuracy'][-1])[-1]  
    return best_model
    

def graphs(best_parameters):
    '''
    Graphical representations of Loss vs. epoch and Evaluation metrics vs. epoch 
    for the model which obtained the best result (according to the selected evaluation metrics)
    '''
    global preds_type
    losses = best_parameters['losses']
    train_accuracy = best_parameters['train accuracy']
    test_accuracy = best_parameters['test_accuracy'] 
    
    # Loss vs epochs
    plt.plot(losses, 'g', label='Training loss')
    plt.title('Loss = f(Epochs)')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.show()
    
    # Accuracy vs epochs
    plt.plot(train_accuracy, 'g', label='Training '+ preds_type) 
    plt.plot(test_accuracy, 'b', label='Test ' + preds_type)
    plt.title(preds_type + ' = f(Epochs)')
    plt.xlabel('Epochs')
    plt.ylabel(preds_type)
    plt.legend()
    plt.show()