from pyspark import SparkContext
import numpy as np
import random
import network as nn
import unicodedata
import sys
import os

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

  #if '?' in line:
  #  return

  parts = line.split(';')

  assignment = random.randint(1, 5)
  inputs = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
  newTarget = 0.0

  if len(parts) < 9:
    return "P" + str(assignment), [np.array(inputs), np.array([newTarget])]

  inputs = []

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

  return "P" + str(assignment), [np.array(inputs), np.array([newTarget])]

def simpleReducer(a, b):
  return len()

def reducer(a, b):
  return [np.vstack((a[0], b[0])), np.vstack((a[1], b[1]))]

def trainNetwork(X,T,parameters): #X,T,[[10,10], 200]

  ## NeworkModeler object init with numInputAttributes, list of hidden layer specs
  nnet = nn.NetworkModeler(X.shape[1], parameters[0], 1 )

  ## trainBySCG on the network modeler object passing the number of iterations todo in SCG
  nnet.trainBySCG(X, T, nIterations=parameters[1], verbose=False)
  return {'neuralnetwork':nnet}


def evaluateNetwork(model,X,T):
  Y=model['neuralnetwork'].predict(X)
  return np.sqrt(np.mean((Y-T)**2))



def trainValidateTestKFolds(trainf,evaluatef,X,T,parameters,nFolds, shuffle=False,verbose=False):
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

    results.append({'testFoldNumber': testFold,'bestNetworkForTheFold':bestModel,
                    'minValidationError': minimumValidationError,'testFoldError':testFoldError })

  # End of Iterations  over all test folds
  return results

def train(a):
  #print(a[1][0].shape, a[1][1].shape)
  return trainValidateTestKFolds(trainNetwork, evaluateNetwork, a[1][0], a[1][1], [[10, 2,10], 100], nFolds=5, shuffle=False)
  
if __name__ == "__main__":
  # input: <file>
  
  # The parameters we will be testing on
  parameters = ['time', 'global_reactive_power', 'voltage', 'global_intensity', 'sub_metering_1', 'sub_metering_2', 'sub_metering_3']
  target = 'global_active_power'
  
  sc = SparkContext(appName="NNTraining")
  sc.addPyFile("network.py")
  sc.addPyFile("scg.py")

  # The output of the mapping
  results = []

  try:
    lines = sc.textFile('hdfs:///test/data.txt', 1)
    
    header = lines.first()
    lines = lines.filter(lambda line: line != header or "?" in line or any(char.isdigit() for char in line))
    
    result = lines.map(lambda line: mapper(line, parameters, target))
    
    results.append(result.cache())
  
  except Exception as e:
    print(e.message)
    
  # RDD to hold the output of all of our mapping
  map_results = sc.emptyRDD()
  
  map_results = sc.union(results)
  map_results = map_results.reduceByKey(reducer).cache()

  map_results.foreach(train)

  #map_results = map_results.map(lambda a: train(a))

  map_results.collect()
  map_results.saveAsTextFile('hdfs:///spark-out')
  
  sc.stop()


