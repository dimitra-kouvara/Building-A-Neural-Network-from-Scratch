"""
ITC6230:    ADVANCED MACHINE LEARNING
Final Project: Regression – Implementation of a Neural Network  
               with Adam optimizer from scratch

Part: Main - Run script

@author: Dimitra
"""
# Importing the necessary libraries
import model
import config
import data_preprocessing
import pandas as pd

# Intializing variables
cache = dict()
grads = dict()

# Loading the hyperparameters for the model
problem = config.problem
no_epochs = config.no_epochs
no_batch_sizes = config.no_batch_sizes
ini_type = config.ini_type  
parameters = config.parameters
layers = parameters['layers']
activations = parameters['activation']
    
if __name__ == "__main__":
    # Loading the data
    x_train, y_train = data_preprocessing.x_train, data_preprocessing.y_train
    x_test, y_test = data_preprocessing.x_test, data_preprocessing.y_test
    x_validation, y_validation = data_preprocessing.x_validation, data_preprocessing.y_validation
    # Testing on different epochs
    for epochs in no_epochs:
        # Testing on different batch sizes
        for batch_size in no_batch_sizes:
            model.fit(x_train, y_train, x_test, y_test, epochs, batch_size, layers, ini_type, activations, cache, grads)
    # Presenting the best model based on the validation accuracy
    best_parameters = model.best_model()
    best_parameters_keys = best_parameters.keys()
    for key in best_parameters_keys:
        if key != 'cache':
            print(key, best_parameters[key])
    # Graphical Representation
    model.graphs(best_parameters)
    # Predict the y labels of the test set
    cache = best_parameters['cache']
    model.predict(cache, y_test, cache, test_flag=True)
    # Test the model on validation data
    model.treat_validation_data(x_validation, y_validation, cache, ini_type, activations)