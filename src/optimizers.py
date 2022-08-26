"""
ITC6230:    ADVANCED MACHINE LEARNING
Final Project: Regression – Implementation of a Neural Network  
               with Adam optimizer from scratch

Part: Optimizers

@author: Dimitra Kouvara
"""
# Importing the necessary libraries
import numpy as np
    
def ADAM_optimize(idx, grads, steps, cache, m, v, learning_rate, beta1, beta2, epsilon):
    '''
    Update parameters using Adam optimizer

    Parameters
    ----------
    idx : int
        It indicates each layer
    grads : python dictionary
        It contains the gradients for each parameter (weight and bias for every node)
    steps : int
        It indicates the timestep (iteration)
    cache : python dictionary
        It contains the updated parameters of the model
    m : python dictionary
        It contains the exponentially weighted average of the gradient
    v : python dictionary
        It contains the exponentially weighted average of the squared gradient
    learning_rate : float
        Stepsize: Larger values results in faster initial learning before the rate is updated, 
        while smaller values slow learning right down during training 
    beta1 : float
        Exponential decay hyperparameter for the first moment estimates 
    beta2 : float
        Exponential decay hyperparameter for the second moment estimates 
    epsilon : float
        hyperparameter preventing division by zero in Adam updates

    Returns
    -------
    cache -- python dictionary containing the updated parameters 

    '''
    # Get gradients at timestep t
    dW = grads[f'dW{idx}']
    db = grads[f'db{idx}']

    # weights
    # Update first moment estimate
    m[f'W{idx}'] = beta1 * m[f'W{idx}'] + (1 - beta1) * dW
    # Update second raw moment estimate
    v[f'W{idx}'] = beta2 * v[f'W{idx}'] + (1 - beta2) * dW ** 2 
        
    # biases
    # Update first moment estimate
    m[f'b{idx}'] = beta1 * m[f'b{idx}'] + (1 - beta1) * db
    # Update second raw moment estimate
    v[f'b{idx}'] = beta2 * v[f'b{idx}'] + (1 - beta2) * db ** 2 

    # take timestep into account
    # Compute corrected first moment estimate
    mt_w  = m[f'W{idx}'] / (1 - beta1 ** steps)
    # Compute corrected second raw moment estimate
    vt_w = v[f'W{idx}'] / (1 - beta2 ** steps)

    mt_b  = m[f'b{idx}'] / (1 - beta1 ** steps)
    vt_b = v[f'b{idx}'] / (1 - beta2 ** steps)
    
    # Update parameters
    w_update = - learning_rate * mt_w / (np.sqrt(vt_w) + epsilon)
    b_update = - learning_rate * mt_b / (np.sqrt(vt_b) + epsilon)

    # Storing the updated weights and biases
    cache[f'W{idx}'] += w_update 
    cache[f'B{idx}'] += b_update 
    

def gradient_descent(idx, cache, grads, learning_rate):
    '''
    Update parameters using gradient descent on 
    every W[l] and b[l] for l = 1, 2, ..., L
    
    W[l] = W[l] - α dW[l]
    b[l] = b[l] - α db[l]
    where α is the learning rate

    Parameters
    ----------
    idx : int
        It indicates each layer

    Returns
    -------
    cache : python dictionary 
          It contains the updated parameters of the model

    '''
    # Vanilla minibatch gradient descent
    cache[f'W{idx}'] -= learning_rate * grads[f'dW{idx}'] 
    cache[f'B{idx}'] -= learning_rate * grads[f'db{idx}'] 
    

