"""
ITC6230:    ADVANCED MACHINE LEARNING
Final Project: Regression – Implementation of a Neural Network  
               with Adam optimizer from scratch

Part: Configuration file - User input

@author: Dimitra Kouvara
"""
csv_file = 'OPTION2_train_data.csv'
label_column = 'Considering_electric_or_hybrid_vehicle_next_purchase'
validation_data = 'OPTION2_test_data_no_labels.csv'
# Options: None (if no validation true labels are provided) / the name of the csv file containing the true labels for the validation dataset
y_validation = None
# Rounding decimal for the y labels for the prediction phase for regression problems
round_decimal = 0

#                                   PRE-PROCESSING PHASE

# Undersampling: used when the dataset is rather large
# Options: True (to activate undersampling) / False (to disactivate undersampling)
undersampling_flag = False
# Construct a dictionary consisting of the samples of each class
# example for the provided dataset
proportion_dictionary = {0: 5000, 1: 5000}
# Highly correlated features flag
high_correlated_features = False
# feature correlation threshold to drop 
corr_threshold = 0.99
# drop highly correlated features
drop_hc_features = False
# Handle outliers
outliers_flag = False
# Replace outliers
outlier_replacement = False
# Eliminate outliers
outlier_elimination = False
# The percentage to which the original data will split to training and test data
split_percentage = 0.75
# adding Gaussian noise to data at the pre-processing phase (before normalization)
# Options:
#          True (to activate gaussian noise addition) / False (to disactivate gaussian noise addition)
noise = False
# If noise=True: Setting the mean and standard deviation
mu = 0
sigma = 0.1

# --------------------------------------------------------------------------------------------------------

#                                   NEURAL NETWORK ARCHITECTURE

#                                   HYPERPARAMETERS SELECTION
grid_search = list()
# Select the model's hyperparameters 
# The number of samples to work through before updating the internal model parameters
no_batch_sizes = [40] # [20, 30, 40]

# The number times that the learning algorithm will work through the entire training dataset
no_epochs = [200] # [70, 100, 200]

# Controls how much to change the model in response to the estimated error each time the model weights are updated
# Larger values results in faster initial learning before the rate is updated, 
# while smaller values slow learning right down during training
learning_rate = 1e-3


#                                   OPTIMIZER PARAMETERS SELECTION

# Select the optimizer
# Options: 
#          None -- using gradient descent & 'Adam'
optimizer = 'Adam'

# Adam Optimizer parameters

# Suggestions:
#              The Adam paper suggests: 
#              Good default settings for the tested machine learning problems are learning rate=0.001, beta1=0.9, beta2=0.999 and epsilon=10−8
#              The TensorFlow documentation suggests some tuning of epsilon:
#              The default value of 1e-8 for epsilon might not be a good default in general.

# The popular deep learning libraries generally use the default parameters
# TensorFlow: learning_rate=0.001, beta1=0.9, beta2=0.999, epsilon=1e-08.
# Keras: lr=0.001, beta_1=0.9, beta_2=0.999, epsilon=1e-08, decay=0.0.
# Blocks: learning_rate=0.002, beta1=0.9, beta2=0.999, epsilon=1e-08, decay_factor=1.
# Lasagne: learning_rate=0.001, beta1=0.9, beta2=0.999, epsilon=1e-08
# Caffe: learning_rate=0.001, beta1=0.9, beta2=0.999, epsilon=1e-08
# MxNet: learning_rate=0.001, beta1=0.9, beta2=0.999, epsilon=1e-8
# Torch: learning_rate=0.001, beta1=0.9, beta2=0.999, epsilon=1e-8

if optimizer == 'Adam':
    # The exponential decay rate for the first moment estimates
    beta1 = 0.9
    # The exponential decay rate for the second-moment estimates.
    beta2 = 0.999
    # Is a very small number to prevent any division by zero in the implementation
    epsilon = 1e-8

#                                   REGULARIZATION TECHNIQUES SELECTION

# L2 Regularization
# Options: True (to activate L2 regularization addition), False (to disactivate L2 regularization addition)
l2_flag = True 
l2_lambda = 0.1

# Dropout
# Options:
#          True (to activate dropout addition) / False (to disactivate dropout addition)
dropout_flag = False
# If dropout_flag=True: Setting the dropout threshold (probability of keeping a neuron active during drop-out, scalar)
dropoutThreshold = 0.5
# Meaning that at every iteration we shut down each neurons of layers with x% probability

#                                   INITIALIZATION OF WEIGHTS AND BIASES SELECTION

# Select the initialization method for the weight matrices and bias vectors
# Options: 
#          'he': He Normal
#          'xavier': Xavier Normal / 'xavier uniform': Xavier Normal
#          'random': Random
#          'zeros': Zeros

# Suggestions:
#              ReLU function for hidden layers: “He Normal” weight initialization
#              Sigmoid function for hidden layers: “Xavier Normal” / “Xavier Uniform” weight initialization
#              TanH function for hidden layers: “Xavier Normal” or “Xavier Uniform” weight initialization
ini_type = 'he'

# Options: True (activate the intialization of biases to 0.1 instead of 0, for ReLU to avoid too 
#          much saturation at initialization) / False (disactivate the intialization of biases to 0.1)
initialization_reluFlag = True 


#                                   LOSS FUNCTION SELECTION

# Select the loss function
# Options: 'mse': Mean Squared Error (MSE)
#          'rmse': Relative Mean Error (RMSE)
#          'mae': Mean Absolute Error (MAE)
#          'rae': Relative Absolute Error (RAE)
#          'categorical_cross-entropy': Multi-class cross-entropy
#          'binary_cross-entropy': Binary cross-entropy
    
# Suggestions: 
#              Regression Loss Functions:
#                                         MSE, RMSE, MAE, RAE
#              Binary Classification Loss Functions:
#                                         Binary Cross-Entropy
#              Multi-Class Classification Loss Functions:
#                                         Multi-Class Cross-Entropy
loss_type = 'binary_cross-entropy'


#                                   EVALUATION METRICS SELECTION

# Select the evaluation metrics
# Options:  
#          'RMSE': Mean Square Error
#          'R^2': R Square
#          'MAE': Mean Absolute Error
#          'accu': Accuracy
    
# Suggestions:
#              For Classification problems (either binary or multi-class): Accuracy
#              For Regression problems: RMSE, R^2, MAE      
preds_type = 'Accuracy'


#                                   LAYERS, NODES AND ACTIVATION FUNCTIONS SELECTION

# Setting the type of problem (currently regression is supported)
# Options:
#          'multi-class classification', 'binary classification', 'regression' 
problem = 'binary classification'

# Neural Network Architecture consisting of two nested dictionaries
# layers, where the number of layers (keys) along with their corresponding number of nodes (values) are defined
# activation, where the number of layers (keys) along with their activation function (values) are defined

# Options:
#          'relu', 'linear', 'sigmoid', 'tanh', 'softmax', 'l_relu'      

# Suggestions:
#              For the hidden layer
#              Multilayer Perceptron (MLP): ReLU activation function.
#              Convolutional Neural Network (CNN): ReLU activation function.
#              Recurrent Neural Network: Tanh and/or Sigmoid activation function.    

#              For the output layer    
#              Regression: One node, linear activation
#              Binary Classification: One node, sigmoid activation.
#              Multiclass Classification: One node per class, softmax activation.
#              Multilabel Classification: One node per class, sigmoid activation.
'''
parameters =  {'layers': {1: 425, 2: 268, 3: 100, 4: 58, 5: 1}, 
               'activation': {1: 'linear', 2: 'relu', 3: 'relu', 4: 'relu', 5: 'linear'}}
'''
parameters = {'layers': {1: 11, 2: 90, 3: 90, 4: 1}, 
              'activation': {1: 'relu', 2: 'relu', 3: 'relu', 4: 'sigmoid'}}


