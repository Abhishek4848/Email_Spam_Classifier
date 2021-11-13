from pyspark import SparkContext
from pyspark.sql import SQLContext
from pyspark.sql.types import * 
from pyspark.streaming import StreamingContext
from pyspark.sql import SparkSession
from pyspark.sql.functions import *


sc = SparkContext("local[2]","test")

'''
here 5 is the time in seconds after which the next stream of data will be read
'''
ssc = StreamingContext(sc,5)
sql_context = SQLContext(sc)

lines = ssc.socketTextStream('localhost',6100)
lines.pprint()

ssc.start()
ssc.awaitTermination()