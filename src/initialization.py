"""
ITC6230:    ADVANCED MACHINE LEARNING
Final Project: Regression – Implementation of a Neural Network  
               with Adam optimizer from scratch

Part: Initialization of weights and bias

@author: Dimitra

Source: See page 317 from Goodfellow book
================================================================== 
We set the biases for each unit to heuristically chosen constants, 
and initialize only the weights randomly.
"""
# Importing the necessary libraries
import numpy as np
import config

# Initializing variables
initialization_reluFlag = config.initialization_reluFlag

# Weights initialization
def initialize_params_zeros(input_nodes, nodes, act):
    '''
    Implement initialization for an L-layer Neural Network -- zeros

    Parameters
    ----------
    input_nodes : int
        The dimensions (number of nodes) of the previous layer in the network
    nodes : int
        The dimensions (number of nodes) of the current layer in the network
    act : str
        the type of the activation function
        
    Returns
    -------
    W : numpy array of shape (size of current layer, size of previous layer)
        The weights matrix
    b : numpy array of shape (size of the current layer, 1)
        The bias vector

    '''
    # set seed for reproducibility 
    np.random.seed(42)
    W = np.zeros((input_nodes, nodes))
    if act == 'relu' and initialization_reluFlag == True:
        b = np.full((1, nodes), 0.1)
    else:
        b = np.zeros((1, nodes))
    return W, b

def initialize_params_rnd(input_nodes, nodes, act):
    '''
    Implement initialization for an L-layer Neural Network -- randomly

    Parameters
    ----------
    input_nodes : int
        The dimensions (number of nodes) of the previous layer in the network
    nodes : int
        The dimensions (number of nodes) of the current layer in the network
    act : str
        the type of the activation function

    Returns
    -------
    W : numpy array of shape (size of current layer, size of previous layer)
        The weights matrix
    b : numpy array of shape (size of the current layer, 1)
        The bias vector

    '''
    # set seed for reproducibility 
    np.random.seed(42)
    W = np.random.rand(input_nodes, nodes)
    if act == 'relu' and initialization_reluFlag == True:
        b = np.full((1, nodes), 0.1)
    else:
        b = np.zeros((1, nodes))
    return W, b
    
def initialize_params_he(input_nodes, nodes, act):
    '''
    Implement initialization for an L-layer Neural Network -- He

    Parameters
    ----------
    input_nodes : int
        The dimensions (number of nodes) of the previous layer in the network
    nodes : int
        The dimensions (number of nodes) of the current layer in the network
    act : str
        the type of the activation function
        
    Returns
    -------
    W : numpy array of shape (size of current layer, size of previous layer)
        The weights matrix
    b : numpy array of shape (size of the current layer, 1)
        The bias vector

    '''
    # set seed for reproducibility 
    np.random.seed(42)
    W = np.random.normal(0, np.sqrt(2.0/input_nodes),(input_nodes, nodes))
    # W = np.random.randn(input_nodes, nodes) * np.sqrt(2/input_nodes) 
    if act == 'relu' and initialization_reluFlag == True:
        b = np.full((1, nodes), 0.1)
    else:
        b = np.zeros((1, nodes))
    return W, b


def initialize_params_xa(input_nodes, nodes, act):
    '''
    Implement initialization for an L-layer Neural Network -- Xavier

    Parameters
    ----------
    input_nodes : int
        The dimensions (number of nodes) of the previous layer in the network
    nodes : int
        The dimensions (number of nodes) of the current layer in the network
    act : str
        the type of the activation function
        
    Returns
    -------
    W : numpy array of shape (size of current layer, size of previous layer)
        The weights matrix
    b : numpy array of shape (size of the current layer, 1)
        The bias vector

    '''
    # set seed for reproducibility 
    np.random.seed(42)
    W = np.random.normal(0, np.sqrt(1.0/input_nodes),(input_nodes, nodes)) 
    # W = np.random.randn(input_nodes, nodes) * np.sqrt(1/input_nodes) 
    if act == 'relu' and initialization_reluFlag == True:
        b = np.full((1, nodes), 0.1)
    else:
        b = np.zeros((1, nodes))
    return W, b


def initialize_params_xaUni(input_nodes, nodes, act):  
    '''
    Implement initialization for an L-layer Neural Network -- Xavier Uniform

    Parameters
    ----------
    input_nodes : int
        The dimensions (number of nodes) of the previous layer in the network
    nodes : int
        The dimensions (number of nodes) of the current layer in the network
    act : str
        the type of the activation function
        
    Returns
    -------
    W : numpy array of shape (size of current layer, size of previous layer)
        The weights matrix
    b : numpy array of shape (size of the current layer, 1)
        The bias vector

    '''
    # set seed for reproducibility 
    np.random.seed(42)
    W = np.random.uniform(-(np.sqrt(6.0/(nodes+input_nodes))), (np.sqrt(6.0/(nodes+input_nodes))), (nodes, input_nodes))
    if act == 'relu' and initialization_reluFlag == True:
        b = np.full((1, nodes), 0.1)
    else:
        b = np.zeros((1, nodes))
    return W, b