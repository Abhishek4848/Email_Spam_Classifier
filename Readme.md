## Real world spark streaming and predictive data modelling.
Email spam Classification

1.  Each record consists of 3 features - the subject, the email content and the label

2.  Each email is one of 2 classes, spam or ham

3.  30k examples in train and 3k in test



**Dataset Link**: [Email spam](https://drive.google.com/drive/folders/10Ys7jqesPfChrAahi4y6rw7FDCGPFFA0)

## How to run
run the python file which will send the data over tcp connection

```python3 stream.py -f <dataset name> -b <batch size>```

execute the spark fetch with the help of spark submit

```$SPARK_HOME/bin/spark-submit spark_fetch.py 2>log.txt```
