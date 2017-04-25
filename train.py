from pyspark import SparkContext
import unicodedata
import sys
import os

def mapper(line, parameters, layers, nodes):
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
  
  parts = line.encode("ascii", "ignore").split(';')
  inputs = []

  for param in parameters:
    inputs.append(parts[pMap[param]])

  return (0, 0), [inputs]
  

def reducer(a, b):
  return a + b
  
if __name__ == "__main__":
  # input: <file>
  
  # The parameters we will be testing on
  parameters = ['time', 'global_active_power']
  
  sc = SparkContext(appName="NNTraining")
  
  # The output of the mapping
  results = []
  
  try:
    lines = sc.textFile('hdfs:///data/uci/mini-data.txt', 1)
    
    header = lines.first()
    lines = lines.filter(lambda line: line != header)
    
    result = lines.map(lambda line: mapper(line, parameters, 5, 5))
    
    results.append(result.cache())
  
  except Exception as e:
    print(e.message)
    
  # RDD to hold the output of all of our mapping
  map_results = sc.emptyRDD()
  
  map_results = sc.union(results)

  # map_results = map_results.reduceByKey(reducer).cache()
  map_results.collect()
  map_results.saveAsTextFile('hdfs:///spark-out')
  
  sc.stop()
