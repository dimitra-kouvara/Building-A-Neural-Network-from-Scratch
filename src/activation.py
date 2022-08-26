"""
ITC6230:    ADVANCED MACHINE LEARNING
Final Project: Regression – Implementation of a Neural Network  
               with Adam optimizer from scratch

Part: Activation Functions

@author: Dimitra Kouvara
"""
# Import the necessary libraries
import numpy as np

def activation_fn(z, activation, derivative=False):
    '''
    Calculate neuron activation for input z
    
    Parameters
    ----------
    z : numpy array
        The product sum of weights and input array plus bias

    Returns
    -------
    numpy array
        the element wise linear function, same shape as Z

    '''
    if activation == 'linear':
        if derivative:
            return dlinear(z)
        return linear(z)
    if activation == 'relu':
        if derivative:
            return drelu(z)
        return relu(z)
    if activation == 'sigmoid':
        if derivative:
            return dsigmoid(z)
        return sigmoid(z)
    if activation == 'softmax':
        if derivative: 
            return dsoftmax(z)
        return softmax(z)
    if activation == 'leaky relu':
        if derivative:
            return dl_relu(z)
        return l_relu(z)
    if activation == 'tanh':
        if derivative:
            return dtanh(z)
        return tanh(z)


def linear(z):
    '''
    Implement the linear activation in numpy
    
    Parameters
    ----------
    z : numpy array
        The product sum of weights and input array plus bias

    Returns
    -------
    numpy array
        the element wise linear function, same shape as Z

    '''
    return z

def dlinear(z):
    '''
    Implement the derivative of the linear activation in numpy
    
    Parameters
    ----------
    z : numpy array
        The product sum of weights and input array plus bias

    Returns
    -------
    numpy array
        the derivative of the linear function, same shape as Z

    '''
    return 1

def sigmoid(z):
    '''
    Implement the sigmoid activation in numpy
    
    Parameters
    ----------
    z : numpy array
        The product sum of weights and input array plus bias

    Returns
    -------
    numpy array
        the element wise sigmoid function, same shape as Z

    '''
    return 1 / (1 + np.exp(-z))

def dsigmoid(z):
    '''
    Implement the derivative of the sigmoid activation in numpy
    
    Parameters
    ----------
    z : numpy array
        The product sum of weights and input array plus bias

    Returns
    -------
    numpy array
        the derivative of the sigmoid function, same shape as Z

    '''
    return sigmoid(z) * (1 - sigmoid(z))

def relu(z):
    '''
    Implement the ReLU activation in numpy
    
    Parameters
    ----------
    z : numpy array
        The product sum of weights and input array plus bias

    Returns
    -------
    numpy array
        the element wise ReLu function, same shape as Z

    '''
    return np.maximum(0, z)

def drelu(z) :
    '''
    Implement the derivative of the ReLU activation in numpy
    
    Parameters
    ----------
    z : numpy array
        The product sum of weights and input array plus bias

    Returns
    -------
    numpy array
        the derivative of the ReLU function, same shape as Z

    '''
    return 1 * (z >= 0)

def l_relu(z) :
    '''
    Implement the leaky ReLU activation in numpy
    
    Parameters
    ----------
    z : numpy array
        The product sum of weights and input array plus bias

    Returns
    -------
    numpy array
        the element wise leaky ReLu function, same shape as Z

    '''
    return np.maximum(z / 100, z)

def dl_relu(z) :
    '''
    Implement the derivative of the leaky ReLU activation in numpy
    
    Parameters
    ----------
    z : numpy array
        The product sum of weights and input array plus bias

    Returns
    -------
    numpy array
        the derivative of the leaky ReLU function, same shape as Z

    '''
    z = 1*(z >= 0)
    z[z == 0] = 1 / 100
    return z

def softmax(x):
    '''
    Implement the softmax activation in numpy
    
    Parameters
    ----------
    z : numpy array
        The product sum of weights and input array plus bias

    Returns
    -------
    numpy array
        the element wise softmax function, same shape as Z

    '''
    exp = np.exp(x - np.max(x, axis = 1, keepdims = True))
    return exp / np.sum(exp, axis = 1, keepdims = True)

def dsoftmax(x):
    '''
    Implement the derivative of the softmax activation in numpy
    
    Parameters
    ----------
    z : numpy array
        The product sum of weights and input array plus bias

    Returns
    -------
    numpy array
        the derivative of the softmax function, same shape as Z

    '''
    exp = np.exp(x - np.max(x, axis = 1, keepdims = True)) 
    return exp / np.sum(exp, axis = 0) * (1 - exp / np.sum(exp, axis = 0))

def tanh(z) :
    '''
    Implement the hyperbolic tangent (tanh) activation in numpy
    
    Parameters
    ----------
    z : numpy array
        The product sum of weights and input array plus bias

    Returns
    -------
    numpy array
        the element wise tanh function, same shape as Z

    '''
    return np.tanh(z)

def dtanh(z) : 
    '''
    Implement the derivative of the hyperbolic tangent (tanh) activation in numpy
    
    Parameters
    ----------
    z : numpy array
        The product sum of weights and input array plus bias

    Returns
    -------
    numpy array
        the derivative of the tanh function, same shape as Z

    '''
    return (1 - tanh(z)**2)