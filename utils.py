'''
Repeating functions that are used across all files, Those functions are mostly 
helper functions or functions from previous courseworks (BM25, inverted index...)
'''
import csv
import re
import pickle
import csv
import string
import numpy as np
import nltk
from nltk.tokenize import word_tokenize
from nltk.stem.wordnet import WordNetLemmatizer
from nltk.corpus import stopwords

lem = WordNetLemmatizer()
nltk.download('stopwords')


def open_tsv(file_name, delimiter):
    '''open tsv files into lines'''
    tsv_file = open(file_name, errors="ignore")
    read_tsv = csv.reader(tsv_file, delimiter=delimiter)

    file = []
    for row in read_tsv:
        file.append(row)

    tsv_file.close()
    return file


def pre_process(txt):
    '''Basic text preprocessing, including 
    to lowercases, remove punctuations/odd characters/slash_n/meaningless words, tokenize, lemmatization'''
    # convert to lowercases
    txt = txt.lower()

    # replace punctuations with spaces
    for punc in string.punctuation:
        txt = txt.replace(punc, ' ')

    # remove odd characters (keep alphabets only)
    txt = re.sub(r'[^a-z ]', '', txt)

    # tokenize the txt
    txt = word_tokenize(txt)

    # lemmatization
    txt = [lem.lemmatize(word) for word in txt]

    # stop word removel, too-short word removel
    stop_words = stopwords.words('english')
    txt = [w for w in txt if w not in stop_words and len(w) > 1]

    return txt


def get_inv_idx(raw_txt):
    '''calculate the inverted index and store with nested dictionaries:
    a dictionary stores all the terms
    each term contains a dictionary, stores all related doc and occurance
    '''
    print('Start calculating inverted index...')
    inverted_idx = {}

    exist_psg = {}
    count = 0

    one_fifth_progress = int(len(raw_txt) / 5)

    for i in range(len(raw_txt)):
        pid =  raw_txt[i][1]

        if pid not in exist_psg:

            exist_psg[pid] = 1
            count += 1

            passage = pre_process(raw_txt[i][3])

            for w in passage:
                if w in inverted_idx:
                    if pid in inverted_idx[w]:
                        inverted_idx[w][pid] += 1
                    else:
                        inverted_idx[w][pid] = 1
                else:
                    inverted_idx[w] = {}
                    inverted_idx[w][pid] = 1

        # print progress every 20%
        if i % one_fifth_progress == 0:
            print(f'Finished {i}/{len(raw_txt)}')
            
    print('Total terms: ', count)
    print('Finish calculating inverted index')
    return inverted_idx


def save_inv_idx(file, filename):
    '''save inverted index to pickle file'''
    pkl_file = open(filename, "wb")
    pickle.dump(file, pkl_file)
    pkl_file.close()
    print('inverted index saved!')


def open_pkl(file_name):
    '''open pickle files that stored nested dictionary'''
    pkl_file = open(file_name, "rb")
    file = pickle.load(pkl_file)
    pkl_file.close()

    return file
    

def cal_BM25(inv_idx, q_terms, k1, k2, b, avdl, passage, N, relevant_p):
    '''calculate BM25 score for one passage (pid)'''
    dl = passage[4]
    K = k1 * ((1-b) + b * (dl/avdl))
    R = len(relevant_p)

    pre_psg = pre_process(passage[3])

    BM25 = 0
    for term in np.unique(q_terms):
        if term in inv_idx:
            n = len(inv_idx[term])
            f = pre_psg.count(term)
            qf = q_terms.count(term)
            r = len(set(relevant_p) & set(inv_idx[term].keys()))

            BM25_1 = np.log(((r+0.5)/(R-r+0.5)) / ((n-r+0.5) / (N-n-R+r+0.5)))
            BM25_2 = ((k1+1) * f) / (K + f)
            BM25_3 = ((k2+1) * qf) / (k2 + qf)

            BM25 += BM25_1 * BM25_2 * BM25_3

    return BM25


def top100BM25(top1000, test_queries, N, inv_idx, relevant_dict):
    '''calculate top 100 BM25 score per query'''
    print('Start calculating top 100 BM25...')
    k1 = 1.2
    k2 = 100
    b = 0.75
    avdl = 0
    for i in range(len(top1000)):
        avdl += top1000[i][4]
    avdl /= len(top1000)

    # empty list to store all BM25
    BM25_all = []

    one_fifth_progress = int(len(test_queries) / 5)

    # for i in range(3): #########################################################
    for i in range(len(test_queries)):

        pre_query = pre_process(test_queries[i][1])

        # empty list to store top 100 BM25
        BM25_top100 = []
        for p in top1000:
            pid = int(p[1])
            qid = int(p[0])
            if qid == int(test_queries[i][0]):
                relevant_p = relevant_dict[str(qid)]

                BM25_p = cal_BM25(inv_idx, pre_query, k1, k2, b, avdl, p, N, relevant_p)

                BM25_top100.append([BM25_p, pid])
        
        BM25_top100.sort(reverse=True)
        BM25_all.append(BM25_top100[:100])

        # print progress every 20%
        if i % one_fifth_progress == 0:
        # if i % 2 == 0:
            print(f'Finished {i}/{len(test_queries)}')

    print('End calculating top 100 BM25')
    return BM25_all


def save2csv(file_name, file, queries):
    '''save to the csv file required'''
    with open(file_name, 'w', newline='') as f:
        writer = csv.writer(f)
        for q in range(len(file)):
            for top in range(len(file[q])):
                row = file[q][top]

                # save in the format: qid,pid,score
                config = [queries[q][0],row[1],row[0]]
                writer.writerow(config)
    print(f'----- {file_name} saved -----')
    f.close


def get_relevance(data):
    '''store related pid and qid pairs'''
    relevant_dict = {}
    for row in data:
        try:
            if float(row[-1]) != 0.0:
                qid = row[0]
                pid = row[1]
                if qid not in relevant_dict:
                    relevant_dict[qid] = [pid]
                else:
                    relevant_dict[qid].append(pid)
        except:
            pass
    return relevant_dict


def save_dictionary(file, filename):
    '''save inverted index to pickle file'''
    pkl_file = open(filename, "wb")
    pickle.dump(file, pkl_file)
    pkl_file.close()
    print(f'{filename} saved!')


def ask_user(question):
    '''ask the user to choose which process to proceed'''
    while True:
        print(question)
        print(f'Valid answers: train load')
        user = input()
        if user not in ['train', 'load']:
            print(f'NOT VALID, please enter: train OR load')
        else:
            print('Proceeding...')
            break
    
    return user
