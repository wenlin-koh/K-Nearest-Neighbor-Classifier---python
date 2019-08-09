#==============================================================================
# title           : knn.py
# author          : Koh Wen Lin
# date            : 25/05/2019
# description     : This contain implementations of knn algorithm.
# python_version  : 3.7.3  
#==============================================================================
import numpy as np

def loadData(filename):
  # load data from filename into X
  X=[]
  count = 0
  
  text_file = open(filename, "r")
  lines = text_file.readlines()
    
  for line in lines:
    X.append([])
    words = line.split(",")
    # convert value of first attribute into float  
    for word in words:
      if (word=='M'):
        word = 0.333
      if (word=='F'):
        word = 0.666
      if (word=='I'):
        word = 1
      X[count].append(float(word))
    count += 1
  
  return np.asarray(X)

def testNorm(X_norm):
  xMerged = np.copy(X_norm[0])
  #merge datasets
  for i in range(len(X_norm)-1):
    xMerged = np.concatenate((xMerged,X_norm[i+1]))
  print (np.mean(xMerged,axis=0))
  print (np.sum(xMerged,axis=0))

# this is an example main for KNN with train-and-test + euclidean
def knnMain(filename, percentTrain, k):
 
  #data load
  X = loadData(filename)
  #normalization
  X_norm = dataNorm(X)
  #data split: train-and-test
  X_split = splitTT(X_norm, percentTrain)
  #KNN: euclidean
  accuracy = knn(X_split[0],X_split[1],k)
  print(accuracy)
  return accuracy

#==============================================================================
# Brief : Function normalizes the input data set and returns a normalized
#         data set.
#
# Parameter :
#    x (2D Array) : Input data set to be normalized.
# 
# Returns :
#    A 2D Array containing the normalized data set.
#==============================================================================
def dataNorm(x):
  iLen = x.shape[0] # Get row size
  jLen = x.shape[1] # Get col size
  
  max = np.full((jLen), -float("inf")) # stores max of all attributes
  min = np.full((jLen), float("inf")) # stores min of all attributes
  x_norm = np.zeros((iLen, jLen)) # stores result of normalized data

  # find the min and max values of all attributes by row
  for i in range(iLen):
    min = np.minimum(x[i], min)
    max = np.maximum(x[i], max)
  
  # Normalized the data by row using (data - min) / (max - min)
  for i in range(iLen):
    x_norm[i] = (x[i] - min) / (max - min)
    x_norm[i][jLen - 1] = x[i][jLen - 1]

  # return normalized data
  return x_norm

#==============================================================================
# Brief : Function splits 2D array into 2 separate 2D array and return a list
#         containing the 2 separate 2D array. Uses Train-and-Test splitting
#         strategy.
#
# Parameter :
#    X_norm (2D Array) : Input data set to be split.
#    percentTrain (float) : Percentage to split for 1st Array and remainder for
#                           second array.
#
# Returns :
#    A list containing two 2D Array. One for training and one for testing
#==============================================================================
def splitTT(X_norm, percentTrain):
  # Compute the separation pivot
  separationPivot = int(percentTrain * X_norm.shape[0])

  # Slice the array into two set
  X_split = [X_norm[:separationPivot], X_norm[separationPivot:]]

  # return the split list
  return X_split

#==============================================================================
# Brief : Function splits 2D array into k-folds of data set and return a list
#         containing k-number of 2D array. Uses k-fold splitting strategy.
#
# Parameter :
#    X_norm (2D Array) : Input data set to be split.
#    k (int) : Number of folds.
#
# Returns :
#    A list containing k-number of 2D Array of data set. 
#==============================================================================
def splitCV(X_norm, k):
  # Shuffle the data set
  np.random.shuffle(X_norm)

  # Compute number of samples per fold
  samplesPerFold = int(X_norm.shape[0] / k)

  # Split the data set into k-folds
  X_split = []
  curr = 0
  for i in range(k - 1):
    X_split += [X_norm[curr:curr + samplesPerFold]]
    curr += samplesPerFold
  
  X_split += [X_norm[curr:]]

  # return the split list
  return X_split

#==============================================================================
# Brief : Function runs the knn algorithm using euclidean distance formula.
#
# Parameter :
#    X_train (List of 2D Array) : List containing training data set.
#    X_test (List of 2D Array) : List containing testing data set.
#    k (int) : Number of nearest neighbour for knn algorithm.
#
# Returns :
#    Accuracy of the test data set. 
#==============================================================================
def knn(X_train, X_test, k):
  return knnImpl(X_train, X_test, k, euclideanDist)

def knnManhattan(X_train, X_test, k):
  return knnImpl(X_train, X_test, k, manhattanDist)

def knnMinkow(X_train, X_test, k):
  return knnImpl(X_train, X_test, k, minkowskiDist)

def knnImpl(X_train, X_test, k, distFn):
  testLen = X_test.shape[0]
  trainLen = X_train.shape[0]

  # stores test values of test set
  # this stores a list containing index of trainSet and distance pair
  neighbourListSet = []
  prediction = []

  # loop all training data and compute distance to test data
  # form k nearest neighbour and predict the output attribute
  for i in range(testLen):
    distPair = []
    for j in range(trainLen):
      distPair.append([int(j), distFn(X_test[i][:-1], X_train[j][:-1])])
    
    # sort the distPair list by distance
    distPair = sorted(distPair, key = lambda x : x[1])

    # retrieve the k nearest neighbour
    distPair = distPair[:k]
    neighbourSet = []
    for pair in distPair:
      neighbourSet.append(X_train[int(pair[0])])
    
    # Compile votes
    votes = {}
    for neighbour in neighbourSet:
      values = neighbour[-1]

      if values in votes:
        votes[values] += 1
      else:
        votes[values] = 1

    # get predicted value from the highest voted value in neighbour set
    prediction.append(max(votes, key = lambda x : votes[x]))

  # Compute number of correct prediction of value
  hit = 0.0
  for i in range(testLen):
    if prediction[i] == X_test[i][-1]:
      hit += 1.0
  
  # return the accuracy of the knn algorithm
  return (hit / testLen) * 100

#==============================================================================
# Brief : Function computes euclidean distance between two n-dimension points.
#
# Parameter :
#    a (1D Array) : Array containing first data.
#    b (1D Array) : Array containing second data.
#
# Returns :
#    Euclidean distance between two points. 
#==============================================================================
def euclideanDist(a, b, k=2):
  return minkowskiDist(a, b, 2)

#==============================================================================
# Brief : Function computes manhattan distance between two n-dimension points.
#
# Parameter :
#    a (1D Array) : Array containing first data.
#    b (1D Array) : Array containing second data.
#
# Returns :
#    Manhattan distance between two points. 
#==============================================================================
def manhattanDist(a, b, k=1):
  return minkowskiDist(a, b, 1)

#==============================================================================
# Brief : Function computes minkowski distance between two n-dimension points.
#
# Parameter :
#    a (1D Array) : Array containing first data.
#    b (1D Array) : Array containing second data.
#    k (int)    : Power value for minkowski distance function.
#
# Returns :
#    Minkowski distance between two points. 
#==============================================================================
def minkowskiDist(a, b, k=3):
  return np.sum(np.abs(b - a)**k)**(1/k)
