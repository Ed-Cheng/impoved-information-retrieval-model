# impoved-information-retrieval-model

To run all the files, you will need:
python matplotlib numpy nltk sklearn gensim xgboost

Add the data directly into this folder

There are 2 ways to go through the code.
1. Faster way by using the given bm25 files
2. Slower way that trains everything from scratch


1. Faster way, run the following files in order:
task1.py
task2.py
data_representation.py
task3.py
task4.py


2. Slower way, run the following files in order:
BM25_val.py
BM25_train.py
task1.py
task2.py
data_representation.py
task3.py
task4.py

########## The role of each py file ########## 
task1.py gives the metrics on BM25
task2.py gives the metrics on LR
task3.py gives the metrics on LM
task4.py gives the metrics on NN
utils.py: helper functions or codes from coursework 1
BM25_val.py: calculate bm25 on validation data
BM25_train.py: calculate bm25 on train data
data_representation.py: create the input training data for LM and NN
LogisticRegression.py: my LR class
metrics.py: evaluation metrics
