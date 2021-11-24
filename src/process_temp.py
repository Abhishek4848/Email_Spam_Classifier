# some libraries are redundant
from pyspark import SparkContext
from pyspark.sql import SQLContext
from pyspark.sql.types import * 
from sklearn.naive_bayes import BernoulliNB
from pyspark.streaming import StreamingContext
from pyspark.sql import SparkSession
from pyspark.sql.functions import *
from pyspark.ml.feature import Tokenizer,StopWordsRemover, CountVectorizer,IDF,StringIndexer, HashingTF, NGram
from pyspark.ml.feature import VectorAssembler
from pyspark.ml.linalg import Vector
from sklearn.metrics import r2_score,accuracy_score, precision_score, recall_score
from pyspark.ml import Pipeline
from pyspark.ml.pipeline import PipelineModel
import joblib
import numpy as np


# model for bernoulli nb
def pre_process_spam_bnb(l,sc):
    spark = SparkSession(sc)
    df = spark.createDataFrame(l,schema='length long,subject_of_message string,content_of_message string,ham_spam string')

    # preprocessing part (can add/remove stuff) , right now taking the column subject_of_message for spam detection
    tokenizer = Tokenizer(inputCol="content_of_message", outputCol="token_text")
    stopwords = StopWordsRemover().getStopWords() + ['-']
    stopremove = StopWordsRemover().setStopWords(stopwords).setInputCol('token_text').setOutputCol('stop_tokens')
    bigram = NGram().setN(2).setInputCol('stop_tokens').setOutputCol('bigrams')
    #stopremove = StopWordsRemover(inputCol='token_text',outputCol='stop_tokens')
    #count_vec = CountVectorizer(inputCol='stop_tokens',outputCol='c_vec')
    ht = HashingTF(inputCol="bigrams", outputCol="ht",numFeatures=8000)
    #count_vec = CountVectorizer(inputCol='stop_tokens',outputCol='c_vec')
    #idf = IDF(inputCol="c_vec", outputCol="tf_idf")
    ham_spam_to_num = StringIndexer(inputCol='ham_spam',outputCol='label')
    #clean_up = VectorAssembler(inputCols=['tf_idf','length'],outputCol='features')

    '''
    Ignore the below code snippet, or can experiment with it.
    '''
    # saving the preprocessing model and reusing it
    # try:
    #     '''
    #     loading the existing processing pipeline
    #     '''
    #     cleaner = PipelineModel.load('../build/saved_pre_clean')
    # except Exception as e:
    #     '''
    #     saving the first preprocess pipeline
    #     '''
    #     print(e)
    #     # creating a pipeling for the preprocess.
    #     data_prep_pipe = Pipeline(stages=[ham_spam_to_num,tokenizer,stopremove,count_vec,idf])
    #     cleaner = data_prep_pipe.fit(df)
    #     cleaner.save("../build/saved_pre_clean")
    #     #joblib.dump(data_prep_pipe, 'saved_pre_clean.pkl')
    #     # data_prep_pipe = Pipeline(stages=[ham_spam_to_num,tokenizer,stopremove,count_vec,idf,clean_up])
    #     # cleaner = data_prep_pipe.fit(df)

    # applying the pre procesed pipeling model on the batches of data recieved
    data_prep_pipe = Pipeline(stages=[ham_spam_to_num,tokenizer,stopremove,bigram,ht])
    cleaner = data_prep_pipe.fit(df)
    clean_data = cleaner.transform(df)
    clean_data = clean_data.select(['label','stop_tokens','ht','bigrams'])

    # splitting the batch data into 70:30 training and testing data
    (training,testing) = clean_data.randomSplit([0.7,0.3])
    X_train = np.array(training.select('ht').collect())
    y_train = np.array(training.select('label').collect())

    # reshaping the data
    nsamples, nx, ny = X_train.shape
    X_train = X_train.reshape((nsamples,nx*ny))

    # doing the above stuff for the test data
    X_test = np.array(testing.select('ht').collect())
    y_test = np.array(testing.select('label').collect())
    nsamples, nx, ny = X_test.shape
    X_test = X_test.reshape((nsamples,nx*ny))
    '''
    Implement incremental learning
    '''
    try:
        '''
        loading the partial model
        '''
        print("Started increment learning")
        clf_load = joblib.load('build/saved_model_clf1.pkl')
        clf_load.partial_fit(X_train,y_train.ravel())
        pred_batch = clf_load.predict(X_test)
        print(pred_batch)
        # calculating the metrics for performance
        score = r2_score(y_test, pred_batch)
        acc = accuracy_score(y_test, pred_batch)
        pr = precision_score(y_test, pred_batch)
        re = recall_score(y_test, pred_batch)
        print("The r2 score is : ",score)
        print("the accuacy is ",acc)
        print("the precision: ",pr)
        print("the recall is:",re)
        joblib.dump(clf_load, 'build/saved_model_clf1.pkl')
    except Exception as e:
        '''
        training the model for the first time
        '''
        print("Started first train of model")
        clf = BernoulliNB()
        clf.partial_fit(X_train,y_train.ravel(),classes=np.unique(y_train))
        pred_batch = clf.predict(X_test)
        print(pred_batch)
        # calculating the metrics for performance
        score = r2_score(y_test, pred_batch)
        acc = accuracy_score(y_test, pred_batch)
        pr = precision_score(y_test, pred_batch)
        re = recall_score(y_test, pred_batch)
        print("The r2 score is : ",score)
        print("the accuacy is ",acc)
        print("the precision: ",pr)
        print("the recall is:",re)
        joblib.dump(clf, 'build/saved_model_clf1.pkl')

    # showing the data after preprocessing
    clean_data.show()


# other models can be added below.