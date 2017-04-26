export PYSPARK_PYTHON=python3
export PYTHONHASHSEED=0
export SPARK_YARN_USER_ENV=PYTHONHASHSEED=0

hadoop fs -rmr /spark-out
/usr/local/spark/bin/spark-submit --master yarn --driver-memory 3g train.py
hadoop fs -cat /spark-out/part-00000 > out.txt

#rm -r out
#/usr/local/hadoop/bin/hadoop fs -get /user/$USER/out
