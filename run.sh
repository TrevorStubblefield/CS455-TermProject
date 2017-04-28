export PYSPARK_PYTHON=python3
export PYTHONHASHSEED=0
export SPARK_YARN_USER_ENV=PYTHONHASHSEED=0

$HADOOP_HOME/bin/hdfs dfs -rm -r -f  /cs455/hw4minidata-spark-out/
#$HADOOP_HOME/bin/hdfs dfs -mkdir /cs455/hw4minidata-spark-out/

/usr/local/spark/bin/spark-submit --master yarn --driver-memory 3g --executor-cores 3 --num-executors 12 train.py

$HADOOP_HOME/bin/hdfs dfs -cat /cs455/hw4minidata-spark-out/part-00000

