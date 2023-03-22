import utils
import numpy as np

from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer


def relevance2dictionary(ydata):
    '''store relevant qid/pid to dictionary'''
    relevant_dict = {}
    for row in ydata:
        if row[0] != 0.0:
            qid = str(int(row[1]))
            pid = str(int(row[2]))
            if qid not in relevant_dict:
                relevant_dict[qid] = [pid]
            else:
                relevant_dict[qid].append(pid)

    return relevant_dict

def create_dataset(BM25_data, relevant_dict, que, pas, que_tfidf, pas_tfidf):
    '''
    Feature decided to extract: BM25, cos similarity, tfidf ratio (q/p)
    Some features e.g. q/p length are discarded cuz they did not improve performance
    '''
    train = np.zeros((len(BM25_data), 3))
    label = np.zeros((len(BM25_data), 3))

    qid_len = 0
    group = []
    prev_qid = BM25_data[0][0].split(',')[0]
    for i, row in enumerate(BM25_data):
        ids = row[0].split(',')
        qid = ids[0]
        pid = ids[1]
        score = ids[2]

        query = que[qid]
        passg = pas[pid]

        # calculate the cosine similarity
        vec = CountVectorizer().fit_transform([query, passg])
        vecs = vec.toarray()
        cos_sim = cosine_similarity(vecs)[0,1]

        # calculate the tf-idf ratio
        tfidf_ratio = que_tfidf[qid] / pas_tfidf[pid]

        # store the features
        train[i] = (score, cos_sim, tfidf_ratio)

        # create label y dataset 
        if pid in relevant_dict[qid]:
            label[i] = (1, qid, pid)
        else:
            label[i] = (0, qid, pid)

        # if the next qid is different, it is the end of the same qid
        if prev_qid != qid:
            group.append(qid_len)
            qid_len = 1
        elif i == len(BM25_data)-1: # last group
            qid_len += 1
            group.append(qid_len)
        else:
            qid_len += 1

        prev_qid = qid

    return train, label, group 


def data2dictionary(data):
    que_dict = {}
    pas_dict = {}
    for row in data:
        qid = row[0]
        pid = row[1] 
        if qid not in que_dict:
            que_dict[qid] = row[2]
        if pid not in pas_dict:
            pas_dict[pid] = row[3]
    
    return que_dict, pas_dict


def tfidf(que, pas):
    vectorizer = TfidfVectorizer()
    X = vectorizer.fit_transform(list(que.values()) + list(pas.values()))

    X1 = np.sum(X[:len(que)], axis=1)
    X2 = np.sum(X[len(que):], axis=1)

    que_tfidf = {}
    for i, key in enumerate(list(que.keys())):
        que_tfidf[key] = float(X1[i])

    pas_tfidf = {}
    for i, key in enumerate(list(pas.keys())):
        pas_tfidf[key] = float(X2[i])

    return que_tfidf, pas_tfidf


if __name__ == '__main__':
    print('Start processing the data...')

    # load all the required data
    ytrain = np.loadtxt('AvgEmbedding_ytrain.csv', delimiter=',')
    yval = np.loadtxt('AvgEmbedding_yval.csv', delimiter=',')

    validation_data = utils.open_tsv("validation_data.tsv", '\t')
    train_data = utils.open_tsv("train_data.tsv", '\t')

    BM25_test = utils.open_tsv("bm25_w_relevance.csv", '\t')
    BM25_train = utils.open_tsv("bm25_w_relevance_train.csv", '\t')

    # convert data to dictionary 
    que_train, pas_train = data2dictionary(train_data)
    que_test, pas_test = data2dictionary(validation_data)

    # calculate tf-idf to extract other features later
    que_train_tfidf, pas_train_tfidf = tfidf(que_train, pas_train)
    que_test_tfidf, pas_test_tfidf = tfidf(que_test, pas_test)

    # creating the data representation
    print('Finish preprocessing, creating datasets...')
    rel_train = relevance2dictionary(ytrain)
    rel_test = relevance2dictionary(yval)
    _, _, split_group = create_dataset(BM25_train, rel_train, que_train, pas_train, que_train_tfidf, pas_train_tfidf)

    data_split = sum(split_group[:int(len(split_group) * 0.8)])

    train, train_label, train_group = create_dataset(BM25_train[:data_split], rel_train, que_train, pas_train, que_train_tfidf, pas_train_tfidf)
    val, val_label, val_group = create_dataset(BM25_train[data_split:], rel_train, que_train, pas_train, que_train_tfidf, pas_train_tfidf)
    test, test_label, test_group = create_dataset(BM25_test, rel_test, que_test, pas_test, que_test_tfidf, pas_test_tfidf)

    # save to csv file
    np.savetxt('DataRep_train.csv', np.concatenate((train, train_label)), delimiter=',')
    np.savetxt('DataRep_train_group.csv', train_group, delimiter=',')
    np.savetxt('DataRep_val.csv', np.concatenate((val, val_label)), delimiter=',')
    np.savetxt('DataRep_val_group.csv', val_group, delimiter=',')
    np.savetxt('DataRep_test.csv', np.concatenate((test, test_label)), delimiter=',')
    np.savetxt('DataRep_test_group.csv', test_group, delimiter=',')
    print('Finished!!!')