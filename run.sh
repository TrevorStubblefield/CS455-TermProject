
hadoop fs -rmr /spark-out
/usr/local/spark/bin/spark-submit --master local[8] --driver-memory 3g --executor-memory 100G train.py
hadoop fs -cat /spark-out/part-00000 > out.txt

#rm -r out
#/usr/local/hadoop/bin/hadoop fs -get /user/$USER/out
