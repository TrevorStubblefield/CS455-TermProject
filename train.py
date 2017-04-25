from pyspark import SparkContext
import sys
import os

def mapper(line, parameters, layers, nodes):
  pMap = {
    'date': 0,
    'global_active_power': 1,
    'global_reactive_power': 2,
    'voltage': 3,
    'global_intensity': 4,
    'sub_metering_1': 5,
    'sub_metering_2': 6,
    'sub_metering_3': 7
  }
  
  parts = line.split(';')
  inputs = []
  
  for param in parameters:
    inputs.append(parts[pMap[param]])
  
  for x in range(1, layers):
    for y in range(1, nodes):
      return (x, y), inputs
  

#def reducer():
  #asda
  
if __name__ == "__main__":
  # input: <file>
  
  # The parameters we will be testing on
  parameters = ['date', 'global_active_power']
  
  sc = SparkContext(appName="NNTraining")
  
  # The output of the mapping
  results = []
  
  try:
    lines = sc.textFile('hdfs://test/data.txt')
    
    header = lines.first()
    lines = lines.filter(lambda line: line != header)
    
    result = lines.map(lambda line: mapper(line, parameters, 5, 5))
    
    results.append(result.cache())
  
  except Exception as e:
    print(e.message)
    
  # RDD to hold the output of all of our mapping
  map_results = sc.emptyRDD()
  
  map_results = sc.union(results)
  
  map_results = map_results.reduceByKey(reducer).cache()
  map_results.collect()
  map_results.saveAsTextFile("out")
  
  sc.stop()
