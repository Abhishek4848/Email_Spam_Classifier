## Real world spark streaming and predictive data modelling.
Pokemon Image Classification

1. Each image is a matrix of size 128x128x3

2. Each image is one of 151 classes, 0-150


**Dataset Link**: [Pokemon generation one](https://drive.google.com/drive/folders/10Ys7jqesPfChrAahi4y6rw7FDCGPFFA0)

## How to run
run the python file which will send the data over tcp connection

```python3 -f <filename> -b <batch size>```

execute the spark fetch with the help of spark submit

```$SPARK_HOME/bin/spark-submit spark_fetch.py 2>log.txt```
