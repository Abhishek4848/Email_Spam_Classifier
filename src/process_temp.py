# some libraries are redundant
# from sklearn.decomposition import PCA
from pyspark import SparkContext
from pyspark.sql import SQLContext
from pyspark.sql.types import * 
from sklearn.naive_bayes import BernoulliNB,MultinomialNB
from sklearn.cluster import MiniBatchKMeans
from sklearn.cluster import MiniBatchKMeans
from sklearn.linear_model import SGDClassifier
from pyspark.streaming import StreamingContext
from pyspark.sql import SparkSession
from pyspark.sql.functions import *
from pyspark.ml.feature import Tokenizer,StopWordsRemover, CountVectorizer,IDF,StringIndexer, HashingTF, NGram, Word2Vec
from pyspark.ml.feature import VectorAssembler, MinMaxScaler
from pyspark.ml.linalg import Vector
from sklearn.metrics import r2_score,accuracy_score, precision_score, recall_score
from pyspark.ml import Pipeline
from pyspark.ml.pipeline import PipelineModel
import joblib
import numpy as np
import csv

#writing into log file
def log_write(score, acc, pr, re, fscore, path):

    #column names
    fields = ['Score', 'Accuracy', 'Precision', 'Recall', 'F1 Score']
    #data row
    row = [score, acc, pr, re, fscore]
    
    filename = path[6:9]+".csv"

    # with open(filename, 'w') as csvfile:
    #     #creating csv writer object
    #     csvwriter = csv.writer(csvfile)
    #     #writing the columns
    #     csvwriter.writerow(fields)

    with open(filename, 'a') as csvfile:
        
        #creating csv writer object
        csvwriter = csv.writer(csvfile)
        #writing the data rows
        csvwriter.writerow(row)

def preprocess(l,sc):
    spark = SparkSession(sc)
    df = spark.createDataFrame(l,schema='length long,subject_of_message string,content_of_message string,ham_spam string')

    # preprocessing part (can add/remove stuff) , right now taking the column subject_of_message for spam detection

    #preprocessing the content of message column
    tokenizer = Tokenizer(inputCol="content_of_message", outputCol="token_text")
    stopwords = StopWordsRemover().getStopWords() + ['-']
    stopremove = StopWordsRemover().setStopWords(stopwords).setInputCol('token_text').setOutputCol('stop_tokens')
    bigram = NGram().setN(2).setInputCol('stop_tokens').setOutputCol('bigrams')
    word2Vec = Word2Vec(vectorSize=5, minCount=0, inputCol="bigrams", outputCol="feature2")
    mmscaler = MinMaxScaler(inputCol='feature2',outputCol='scaled_feature2')


    # pre-procesing the subject of message column
    tokenizer1 = Tokenizer(inputCol="subject_of_message", outputCol="token_text1")
    stopwords1 = StopWordsRemover().getStopWords() + ['-']
    stopremove1 = StopWordsRemover().setStopWords(stopwords).setInputCol('token_text1').setOutputCol('stop_tokens1')
    bigram1 = NGram().setN(2).setInputCol('stop_tokens1').setOutputCol('bigrams1')
    word2Vec1 = Word2Vec(vectorSize=5, minCount=0, inputCol="bigrams1", outputCol="feature1")
    mmscaler1 = MinMaxScaler(inputCol='feature1',outputCol='scaled_feature1')

    #ht = HashingTF(inputCol="bigrams", outputCol="ht",numFeatures=8000)
    ham_spam_to_num = StringIndexer(inputCol='ham_spam',outputCol='label')
    print("starting pre-processing of data (wait 2-3 min)")

    # applying the pre procesed pipeling model on the batches of data recieved
    data_prep_pipe = Pipeline(stages=[ham_spam_to_num,tokenizer,stopremove,bigram,word2Vec,mmscaler,tokenizer1,stopremove1,bigram1,word2Vec1,mmscaler1])
    cleaner = data_prep_pipe.fit(df)
    clean_data = cleaner.transform(df)
    clean_data = clean_data.select(['label','stop_tokens','bigrams','feature1','feature2','scaled_feature2','scaled_feature1'])
    print("pre process of data completed")

    # splitting the batch data into 70:30 training and testing data
    (training,testing) = clean_data.randomSplit([0.8,0.2])
    clean_data.show()

    X_train = np.array(training.select(['scaled_feature1','scaled_feature2']).collect())
    y_train = np.array(training.select('label').collect())
    print("splitting data for test and train complete")
    # reshaping the data
    nsamples, nx, ny = X_train.shape
    X_train = X_train.reshape((nsamples,nx*ny))

    # doing the above stuff for the test data

    X_test = np.array(testing.select(['scaled_feature1','scaled_feature2']).collect())
    y_test = np.array(testing.select('label').collect())
    nsamples, nx, ny = X_test.shape
    X_test = X_test.reshape((nsamples,nx*ny))
    return (X_test,y_test,X_train,y_train)


# function to meawsure peformance metrics
def perform_metrics(y_test,pred_batch):
    print(pred_batch)
    # calculating the metrics for performance
    score = r2_score(y_test, pred_batch)
    acc = accuracy_score(y_test, pred_batch)
    pr = precision_score(y_test, pred_batch)
    re = recall_score(y_test, pred_batch)
    fscore = (2*re*pr)/(re+pr)
    print("The r2 score is : ",score)
    print("the accuacy is ",acc)
    print("the precision: ",pr)
    print("the recall is:",re)
    print("the F1 score is:",fscore)
    return(score, acc, pr, re, fscore)

# model for bernoulli nb
def pre_process_spam_bnb(l,sc):
    X_test,y_test,X_train,y_train = preprocess(l,sc)
    '''
    Implement incremental learning
    '''
    try:
        '''
        loading the partial model
        '''
        print("Started increment learning")
        clf_load = joblib.load('build/bNB.pkl')
        clf_load.partial_fit(X_train,y_train.ravel())
        pred_batch = clf_load.predict(X_test)
        score, acc, pr, re, fscore = perform_metrics(y_test,pred_batch)
        log_write(score, acc, pr, re, fscore,'build/bNB')
        joblib.dump(clf_load, 'build/bNB.pkl')
    except Exception as e:
        '''
        training the model for the first time
        '''
        print("Started first train of bernoulli model")
        clf = BernoulliNB()
        clf.partial_fit(X_train,y_train.ravel(),classes=np.unique(y_train))
        pred_batch = clf.predict(X_test)
        score, acc, pr, re, fscore = perform_metrics(y_test,pred_batch)
        log_write(score, acc, pr, re, fscore,'build/bNB')
        joblib.dump(clf, 'build/bNB.pkl')

    # showing the data after preprocessing
    # clean_data.show()


# multinomial nb model
def pre_process_spam_mnb(l,sc):
    X_test,y_test,X_train,y_train = preprocess(l,sc)
    '''
    Implement incremental learning
    '''
    try:
        '''
        loading the partial model
        '''
        print("Started increment learning")
        clf_load = joblib.load('build/mNB.pkl')
        clf_load.partial_fit(X_train,y_train.ravel())
        pred_batch = clf_load.predict(X_test)
        score, acc, pr, re, fscore = perform_metrics(y_test,pred_batch)
        log_write(score, acc, pr, re, fscore, 'build/mNB')
        joblib.dump(clf_load, 'build/mNB.pkl')
    except Exception as e:
        '''
        training the model for the first time
        '''
        print("Started first train of multinomial model")
        clf = MultinomialNB()
        clf.partial_fit(X_train,y_train.ravel(),classes=np.unique(y_train))
        pred_batch = clf.predict(X_test)
        score, acc, pr, re, fscore = perform_metrics(y_test,pred_batch)
        log_write(score, acc, pr, re, fscore, 'build/mNB')
        joblib.dump(clf, 'build/mNB.pkl')

    # showing the data after preprocessing
    # clean_data.show()
    
# Model for SGDClassifier
def pre_process_spam_SGD(l,sc):
    X_test,y_test,X_train,y_train = preprocess(l,sc)
    '''
    Implement incremental learning
    '''
    try:
        '''
        loading the partial model
        '''
        print("Started increment learning")
        clf_load = joblib.load('build/SGD.pkl')
        clf_load.partial_fit(X_train,y_train.ravel())
        pred_batch = clf_load.predict(X_test)
        score, acc, pr, re, fscore = perform_metrics(y_test,pred_batch)
        log_write(score, acc, pr, re, fscore, 'build/SGD')
        joblib.dump(clf_load, 'build/SGD.pkl')
    except Exception as e:
        '''
        training the model for the first time
        '''
        print("Started first train of SGD model")
        clf = SGDClassifier()
        clf.partial_fit(X_train,y_train.ravel(),classes=np.unique(y_train))
        pred_batch = clf.predict(X_test)
        score, acc, pr, re, fscore = perform_metrics(y_test,pred_batch)
        log_write(score, acc, pr, re, fscore, 'build/SGD')
        joblib.dump(clf, 'build/SGD.pkl')

    # showing the data after preprocessing
    # clean_data.show()

# Model for Clustering
def pre_process_spam_KMC(l,sc):
    X_test,y_test,X_train,y_train = preprocess(l,sc)
    # pca = PCA(3)	# Decomposition to 3D
    # X_train = pca.fit_transform(X_train)
    # X_test = pca.transform(X_test)
    '''
    Implement incremental learning
    '''
    try:
        '''
        loading the partial model
        '''
        print("Started increment learning")
        clf_load = joblib.load('build/KMC.pkl')
        clf_load.partial_fit(X=X_train,y=y_train.ravel())
        pred_batch = clf_load.predict(X_test)
        score, acc, pr, re, fscore = perform_metrics(y_test,pred_batch)
        log_write(score, acc, pr, re, fscore, 'build/KMC')
        joblib.dump(clf_load, 'build/KMC.pkl')
    except Exception as e:
        '''
        training the model for the first time
        '''
        print("Started first train of KMC model")
        clf = MiniBatchKMeans(n_clusters=2)
        clf.partial_fit(X=X_train,y=y_train.ravel())
        pred_batch = clf.predict(X_test)
        score, acc, pr, re, fscore = perform_metrics(y_test,pred_batch)
        log_write(score, acc, pr, re, fscore, 'build/KMC')
        joblib.dump(clf, 'build/KMC.pkl')

    # showing the data after preprocessing
    # clean_data.show()
