"""
ITC6230:    ADVANCED MACHINE LEARNING
Final Project: Regression – Implementation of a Neural Network  
               with Adam optimizer from scratch

Part: Data pre-processing

@author: Dimitra
"""
# Import the necessary libraries
import pandas as pd
import config
import numpy as np
import matplotlib.pyplot as plt 
import imblearn.under_sampling as under
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeRegressor
# Catching warnings
import warnings
warnings.filterwarnings("error")

def NormalizeData1(data):
    '''
    Inplement the normalization of data, that is transforming the data to appear 
    on the same scale across all the records, more specifically from [0, 1].
    (with min - max)
    
    Parameters
    ----------
    data : numpy array
        input data

    Returns
    -------
    numpy array
        normalized array

    '''
    # bring each column to 0 mean and 1 variance
    return (data - data.mean(axis=0)) / data.std(axis=0)

def NormalizeData2(data):
    '''
    Inplement the normalization of data, that is transforming the data to appear 
    on the same scale across all the records, more specifically from [0, 1].
    (with mean - std)
    
    Parameters
    ----------
    data : numpy array
        input data

    Returns
    -------
    numpy array
        normalized array

    '''
    return (data - np.min(data)) / (np.max(data) - np.min(data))

def unflatten(Y):
    '''
    The inverse procedure of flattening of an array, 
    so that the data are fed to the neural network

    Parameters
    ----------
    Y : numpy ndarray
        The labels for training and test data

    Returns
    -------
    numpy array
        The labels for training and test data into a different format

    '''
    new_Y = []
    for label in Y:
        new_Y.append([label])
    return np.array(new_Y) 


def one_hot(array):
    '''
    Convert labels (categorical values) to one-hot encodings

    Parameters
    ----------
    array : numpy ndarray
        The labels (y_true values) of the dataset

    Returns
    -------
    onehot : numpy ndarray
        The labels (y_true values) of the dataset expressed as an array of 0s and 1s

    '''
    unique, inverse = np.unique(array, return_inverse=True)
    onehot = np.eye(unique.shape[0])[inverse]
    return onehot

# Initializing variables
noise = config.noise
undersampling_flag = config.undersampling_flag
high_correlated_features = config.high_correlated_features
corr_threshold = config.corr_threshold
drop_hc_features = config.drop_hc_features
outliers_flag = config.outliers_flag
outlier_replacement = config.outlier_replacement
outlier_elimination = config.outlier_elimination

# Suggestion on the selection of the hidden layers' number of nodes 
# (mainly applied for non Deep Neural Networks)
# Options: True / False
noNodes = False 

problem = config.problem
split_percentage = config.split_percentage
csv_file = config.csv_file
label_column = config.label_column
validation_data = config.validation_data
y_validation = config.y_validation
if y_validation != None:
    validation_flag = True
else:
    validation_flag = False

# Load data on dataframe
df = pd.read_csv(csv_file, sep=',')
# Info on the data
print(df.info())
# Statistics
print(df.describe())


# Finding how many rows have at least one missing value
df[df.isnull().any(axis = 1)]

missing_data = pd.DataFrame({'percent_missing': df.isnull().sum() * 100 / len(df)})
print("\nColumns with most missing values percentage: ")
with pd.option_context("display.max_rows", None):
    print(missing_data.sort_values('percent_missing', ascending = False))

# Dropping whole columns if they have more than 80% missing values
all_columns = list(df.columns)
for var in all_columns:
    if (missing_data.loc[var, 'percent_missing'] > 80):
        del df[var]
        
print("\nColumns with missing values: ") 
print(df[df.isnull().any(axis = 1)])

# Finding the percentage of the labels
print(df[label_column].value_counts(normalize=True)*100)
# A question of whether the dataset is balanced or not arises!

# Graphical representation
print("\nGraphs")
class_column = 'name of class column'

# attributes = data.select_dtypes(include = ['float64'])
column_size = df.shape[1]
attributes = df.iloc[:,:column_size]

print("\nNumerical attributes: ", attributes.columns.values)

plt.figure()
ds_attributes = attributes.hist(figsize = (22,20))
plt.show()
 
# Loading the validation data
validation = pd.read_csv(validation_data, sep=',')
if validation_flag:    
    y_validation = pd.read_csv(y_validation, sep=',')

# Handling high correlated features   
if high_correlated_features:
    # Correlation Matrix of the attributes    
    plt.figure(figsize = (10,7))
    c = df.corr()
    mask = np.triu(np.ones_like(c, dtype = bool))
    sns.heatmap(c, mask=mask, annot = False, cmap = 'coolwarm', linecolor = 'white', linewidths=0.1) 
    plt.show()
    print("Finding highly correlated features...")
    array_keys = list(df.columns)


    # Create loop to test each feature as a dependent variable
    df_feature_score = pd.DataFrame(data = None, index = None, columns = ['Score'])
    for var in array_keys:
        print("Check correlated feature:", var)
        new_data = df.drop([var], axis = 1) 
    
        # Create feature Series (Vector)
        new_feature = pd.DataFrame(df.loc[:, var])  
        x_train, x_test, y_train, y_test = train_test_split(new_data, new_feature, test_size = 0.25, random_state = 42) 
        clf_DT = DecisionTreeRegressor(random_state = None)
        # Fit
        clf_DT.fit(x_train, y_train) 
        score = clf_DT.score(x_test, y_test)
        df_feature_score.loc[var] = score  
        
    # Select features with correlation score greater than threshold
    hc_columns = df_feature_score[df_feature_score['Score'] > corr_threshold].index
    print("\nHigh correlated columns over threshold " + str(corr_threshold) + " : ", hc_columns)
    df_feature_score = df_feature_score.sort_values(by = ['Score'], ascending = False)
    print(df_feature_score) 
    
    if drop_hc_features:
        print("The features before dropping")
        print(df.columns)
        print("Dropping highly correlated features...")
        df = df.drop(hc_columns, axis = 1)
        print("The remaining features")
        print(df.columns)

# Handling outliers        
if outliers_flag: 
    """
    ON histograms, an outlier will appear outside the overall pattern of distribution.
    """
    attributes = df.select_dtypes(include = ['int64','float64'])
    print("\nNumerical attributes distribution: \n" )
    
    plt.figure()
    num_plot = attributes.hist(figsize=(32,30))
    plt.show()
    
    q1 = 0.25 
    q3 = 0.75
    if outlier_replacement:    
        print("Handling Outliers")
        print("\nReplace outliers with median: ") 
        dep_vars = list(df.columns)
        
        for var in dep_vars:
            fithQ = round(df[var].quantile(float(q1)), 4)
            medQ = round(df[var].quantile(0.50), 4)
            nintfifQ = round(df[var].quantile(float(q3)), 4)
            
            print("\nValue " + var + " of " + str(q1) +": ", str(fithQ) ) 
            print("\nValue " + var + " of 50%: ", str(medQ)) 
            print("\nValue " + var + " of " + str(q3) +": ", str(nintfifQ))
            
            df[var] = np.where(df[var] < fithQ, medQ, df[var])
            df[var] = np.where(df[var] > nintfifQ, medQ, df[var])
        print(df.shape)
        print(df.describe())
    
    if outlier_elimination: 
        Q1 = df.quantile(q1)
        Q3 = df.quantile(q3)
        IQR = Q3 - Q1 
        print("IQR: ")
        print(IQR)
        
        df = df[~((df < (Q1 - 1.5 * IQR)) |(df > (Q3 + 1.5 * IQR))).any(axis=1)]
        print(df.shape)
        print(df.describe())

if undersampling_flag:
    proportion_dictionary = config.proportion_dictionary
    UnderSampling = under.ClusterCentroids(sampling_strategy=proportion_dictionary, random_state=83, voting='hard')
    x_resampled, y_resampled = UnderSampling.fit_resample(df, df[label_column])
    
    df = x_resampled
    print(df[label_column].value_counts(normalize=True)*100)

# Convert the labels of the binary classification problem to 0 and 1, if necessary
if problem == 'binary classification' and df[label_column].dtypes == 'object':
    labels = df[label_column].unique()
    df[label_column] = np.where(df[label_column].eq(labels[0]), 1, 0) 

# Creating a dataframe with values of original dataframe
train = df.sample(frac = split_percentage, random_state=200) # random state is a seed value
# Creating dataframe with rest of the values
test = df.drop(train.index)
x_train = train.drop(label_column, axis=1)
x_test = test.drop(label_column, axis=1)
y_train = train[label_column]
y_test = test[label_column]

# Convert dataframes to numpy arrays
x_train = x_train.to_numpy()
x_test = x_test.to_numpy()

y_train = y_train.to_numpy()
y_test = y_test.to_numpy()

x_validation = validation.to_numpy()
if validation_flag:
    y_validation = y_validation.to_numpy()

# Adding noise to data before normalization
if noise:
    mu = config.mu
    sigma = config.sigma
    # add noise i.e., N(μ=0, σ^2=0.01)
    noise_train = np.random.normal(mu, sigma, [x_train.shape[0], x_train.shape[1]])
    noise_test = np.random.normal(mu, sigma, [x_test.shape[0], x_test.shape[1]])
    
    x_train = x_train + noise_train
    x_test = x_test + noise_test
    
# Handle warnings as errors    
try:
    # Normalize the data in the range of 0 to 1 using mean and std value
    x_train = NormalizeData1(x_train)
    x_test = NormalizeData1(x_test)
    x_validation = NormalizeData1(x_validation)
except RuntimeWarning:
    # Normalize the data in the range of 0 to 1 using min and max value
    x_train = NormalizeData2(x_train)
    x_test = NormalizeData2(x_test)
    x_validation = NormalizeData2(x_validation)


if noNodes:
    # Calculating the number of hidden neurons from formula:
    # Nh = Ns / (α ∗ (Ni + No))
    # where
    # N_s = number of samples in training data set
    N_s = x_train.shape[0]
    # Ni = number of input neurons. / number of features
    N_i = x_train.shape[1]
    # No = number of output neurons.
    def count_unique(array):
        (unique, counts) = np.unique(array, return_counts=True)
        frequencies = np.asarray((unique, counts)).T
        return frequencies
    
    N_o = count_unique(y_train).shape[0]
    # Ns = number of samples in training data set.
    # α = an arbitrary scaling factor usually 2-10.
    a = 2
    N_h = int(N_s / (a * (N_i + N_o)))
    print(N_h)


if problem == 'regression' or problem == 'binary classification':
    y_train = unflatten(y_train)
    y_test = unflatten(y_test)
    
elif problem == 'multi-class classification':
    y_train = y_train.astype(int)
    y_test = y_test.astype(int)
    
    # Converting categorical data via one hot encoding
    y_train = one_hot(y_train)
    y_test = one_hot(y_test) 
    
    if validation_flag:
        y_validation = y_validation.astype(int)
        y_validation = one_hot(y_validation)