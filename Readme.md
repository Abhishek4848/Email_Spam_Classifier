## Real world spark streaming and predictive data modelling.
Email spam Classification


1.  Each record consists of 3 features - the subject, the email content and the label

2.  Each email is one of 2 classes, spam or ham

3.  30k examples in train and 3k in test



**Dataset Link**: [Email spam](https://drive.google.com/drive/folders/1mMPa21_FInHVNOaG5irmve42Su6dI77K)

## How to run
run the python file which will send the data over tcp connection

```python3 stream.py -f <dataset name> -b <batch size>```

execute the spark fetch with the help of spark submit

```$SPARK_HOME/bin/spark-submit spark_fetch.py 2>log.txt```

## Demo to run
need to experiment with the batch size ( >1000).

running the stream.py file
![image](https://user-images.githubusercontent.com/54106076/143176095-dd4346e9-7c72-4b7f-ba4f-d18f616cd197.png)

running the spark_fetch file
![image](https://user-images.githubusercontent.com/54106076/143176371-3fb36a43-7fa0-491b-9092-16e5323fb21a.png)
