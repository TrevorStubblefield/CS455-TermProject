from pyspark import SparkContext
import numpy as np
import random
import network as nn
import unicodedata
import sys
import os
import datetime as datetime

def mapper(line, parameters, target):
  pMap = {
    'time': 1,
    'global_active_power': 2,
    'global_reactive_power': 3,
    'voltage': 4,
    'global_intensity': 5,
    'sub_metering_1': 6,
    'sub_metering_2': 7,
    'sub_metering_3': 8
  }

  parts = line.split(';')
  inputs = []

  if len(parts) >= 9:
    for x in range(1, 9):
      if x > 1:
        try:
          parts[x] = float(parts[x])
        except (ValueError, TypeError):
          parts[x] = 0.0
      else:
        try:
          temp = parts[x].split(":")
          parts[x] = int(temp[0]) * 60 + int(temp[1])
        except (ValueError, TypeError):
          parts[x] = 0


    for param in parameters:
      inputs.append(parts[pMap[param]])

    try:
      newTarget = float(parts[pMap[target]])
    except (ValueError, TypeError):
      newTarget = 0.0

    assignment = random.randint(1, 10)

    return assignment, [np.array(inputs), np.array([newTarget])]
  else:
    assignment = random.randint(1, 10)

    return assignment, [np.array([-1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]), np.array([0.0])]

def simpleReducer(a, b):
  return len()

def reducer(a, b):
  return [np.vstack((a[0], b[0])), np.vstack((a[1], b[1]))]

## Method to train on training dataset.
## Training set is matrix of inputs X minus the output/target columns and output/target columns expressed as target mat
## or target column and output matrix 
## Takes arguments X, T and a 2 element neural network parameters list 'parameters'
## 1st element of 'parameters' is another positional list of hidden layer specs i.e # of units in each hidden layer 
## 2nd element of 'parameters' is number of iterations to perform in gradient descent using scaled conjugate gradient d
## Returns the trained neural network as model that can be used to predict on test set or validation set

def trainNetwork(X,T,parameters): #X,T,[[10,10], 200]

  ## NeworkModeler object init with numInputAttributes, list of hidden layer specs
  nnet = nn.NetworkModeler(X.shape[1], parameters[0], 1 )

  ## trainBySCG on the network modeler object passing the number of iterations todo in SCG
  nnet.trainBySCG(X, T, nIterations=parameters[1], verbose=False)
  return {'neuralnetwork':nnet}

# Method to evaluate the error (RMSE) of predictions on test set or validation set or any other dataset.
# Takes as arguments the test set X and T matrices along with the model object which is really the constructed neural n
# Model is used to use to predict Y matrix of outputs for input matrix X
# The predicted Y matrix of outputs is compared against recorded T matrix of outputs for the input dataset
# RMSE is calculated for Y‐T diff and returned

def evaluateNetwork(model,X,T):
  Y=model['neuralnetwork'].predict(X)
  return np.sqrt(np.mean((Y-T)**2))


## This method does following pseudo code
##for each of the K test folds:
##     for each K‐1 validation folds for this test fold:
##         instantiate neural network using supplied hidden layers parameters 
##         run SCG for supplied nIterations to train the network
##         evaluate error of prediction for this validation fold by predicting using the trained network
##         if this validation fold's prediction error is < minimum error across this test fold's validation folds:
##             update new best Model i.e. the best network to be this iteration's trained network
##             update new best error to be this validation fold's prediction error
##     Now use the best network across all validation folds of this test fold to predict on this test fold
##     Evaulate this test fold's prediction error and note down the error
##     note down this testfold's chosen trained network (the best across this test fold's validation folds)
##     
## Return the list of dictionaries for each test fold
## dictionary for each fold has foldNumber, best network, fold's error, minimum of fold's validation fold errors 
## 
def trainValidateTestKFolds(trainf,evaluatef,X,T,parameters,nFolds, shuffle=False,verbose=False):

  # first get rid of bad rows with indices for which X[:,0] is -1 i.e. time a.k.a minute of the day = -1.0
  goodRowsBooleanMask = X[:,0] != -1
  X = X[goodRowsBooleanMask]
  T = T[goodRowsBooleanMask]
  # Randomly arrange row indices
  rowIndices = np.arange(X.shape[0])
  if shuffle:
    np.random.shuffle(rowIndices)
  # Calculate number of samples in each of the nFolds folds
  nSamples = X.shape[0]
  nEach = int(nSamples / nFolds)
  if nEach == 0:
    raise ValueError("partitionKFolds: Number of samples in each fold is 0.")
  # Calculate the starting and stopping row index for each fold.
  # Store in startsStops as list of (start,stop) pairs
  starts = np.arange(0,nEach*nFolds,nEach)
  stops = starts + nEach
  stops[-1] = nSamples
  startsStops = list(zip(starts,stops))
  # Repeat with testFold taking each single fold, one at a time
  results = []
  minimumValidationError = None
  #bestModel = None
  # Iterations  over all test folds
  for testFold in range(nFolds):
    # Find best set of parameter values
    #bestParms = []
    validationFoldErrors = []

    # iterations over validation folds and train, evaluate
    for validateFold in range(nFolds):
      if testFold == validateFold:
        continue
      # trainFolds are all remaining folds, after selecting test and validate folds
      trainFolds = np.setdiff1d(range(nFolds), [testFold,validateFold])
      # Construct Xtrain and Ttrain by collecting rows for all trainFolds
      rows = []
      for tf in trainFolds:
        a,b = startsStops[tf]
        rows += rowIndices[a:b].tolist()
      Xtrain = X[rows,:]
      Ttrain = T[rows,:]

      # Construct Xvalidate and Tvalidate
      a,b = startsStops[validateFold]
      rows = rowIndices[a:b]
      Xvalidate = X[rows,:]
      Tvalidate = T[rows,:]

      # Construct Xtest and Ttest
      a,b = startsStops[testFold]
      rows = rowIndices[a:b]
      Xtest = X[rows,:]
      Ttest = T[rows,:]

      # now train and evaluate for this validation fold
      model=trainf(Xtrain,Ttrain,parameters)
      thisValidationFoldError=evaluatef(model,Xvalidate,Tvalidate)
      if minimumValidationError == None:
        minimumValidationError = thisValidationFoldError
        bestModel = model
      else:
        if thisValidationFoldError < minimumValidationError :
          minimumValidationError = thisValidationFoldError
          bestModel = model

    # End of iterations over validation folds and train, evaluate

    ## Now check test errors with this best model obtained across the validations folds

    a2,b2 = startsStops[testFold]
    testRows = rowIndices[a2:b2]
    NewXtest = X[testRows,:]
    NewTtest = T[testRows,:]

    testFoldError=evaluatef(bestModel,NewXtest,NewTtest)

    results.append({'Number of samples': nSamples, 'testFoldNumber': testFold,'bestNetworkForTheFold':bestModel,
                    'minValidationError': minimumValidationError,'testFoldError':testFoldError })

  # End of Iterations  over all test folds
  return results


if __name__ == "__main__":
  # input: <file>

  # The parameters we will be testing on
  parameters = ['time', 'global_reactive_power', 'voltage', 'global_intensity', 'sub_metering_1', 'sub_metering_2', 'sub_metering_3']
  target = 'global_active_power'

  sc = SparkContext(appName="NNTraining")
  sc.addPyFile("network.py")
  sc.addPyFile("scg.py")

  print('DefaultParallelism=', sc.defaultParallelism)

  try:
    #lines = sc.textFile('hdfs:///cs455/hw4fulldata/alldata.txt', 10)
    lines = sc.textFile('hdfs:///cs455/hw4minidata/x01-1', 10)

    header = lines.first()
    lines = lines.filter(lambda line: line != header)

    map_results = lines.map(lambda line: mapper(line, parameters, target), True)

  except Exception as e:
    print(e.message)

  
  reduce_results = map_results.reduceByKey(reducer).cache()
  map_results.unpersist()

  print('After reduce by buckets', str(datetime.datetime.now()))

  train_map_results = reduce_results.map(lambda a: trainValidateTestKFolds(trainNetwork, evaluateNetwork, a[1][0], a[1][1], [[10, 2,10], 100], nFolds=5, shuffle=False))

  train_map_results.cache()
  reduce_results.unpersist()

  train_map_results.collect()

  print('After collect', str(datetime.datetime.now()))

  with open(sys.argv[1], "w") as text_file:
    i=0
    for x in train_map_results.toLocalIterator():
      print("Results from partition {}:\n {}".format(i,x), file=text_file) 
      i += 1
    #[print("Results from partition:\n {}".format(x), file=text_file) for x in train_map_results.toLocalIterator()]


  #train_map_results.saveAsTextFile('hdfs:///cs455/hw4minidata-spark-out')

  print('After save', str(datetime.datetime.now()))

  sc.stop()
