import utils 


def sort_data(data):
    '''convert data to the required format for calculating bm25 with sub-sampling'''
    # we take 2000 queries only, and for each query take top 500 instead of top 1000
    top500 = []
    top500_dict = {}
    query2000_dict = {}

    for row in data:
        qid = row[0]
        rel = row[4]

        if (len(query2000_dict) < 2001) or (qid in query2000_dict):

            if qid not in top500_dict:
                top500_dict[qid] = 1
            else:
                top500_dict[qid] += 1

            # always store if relevant, otherwise store up to top500 
            if rel != '0.0' or top500_dict[qid] < 500:
                pid_len = len(row[3].split())
                temp_row = row[:4]
                temp_row.append(pid_len)
                top500.append(temp_row)
            
            if qid not in query2000_dict:
                query2000_dict[qid] = row[2]

    # remove the headings
    del query2000_dict['qid']
    del top500[0]

    # convert query to a list
    query_list = [[qid, query] for qid, query in query2000_dict.items()] 

    return top500, query_list

if __name__ == '__main__':
    print('Start calculating BM25 on validation set (with sub-sampling)')

    train = utils.open_tsv("train_data.tsv", "\t")

    # sort the data into the format for inverted index and BM25
    top500, query_list = sort_data(train)

    ## get the inverted index
    inverted_index = utils.get_inv_idx(top500)

    # store the inverted index
    utils.save_inv_idx(inverted_index, "inverted_index_train.pkl")

    # store the relevance into a dictionary 
    relevant_dict = utils.get_relevance(train)

    # open the stored inverted index
    inv_idx = utils.open_pkl("inverted_index_train.pkl")

    # number of documents in collection
    N = len(top500)

    # BM25 score
    BM25 = utils.top100BM25(top500, query_list, N, inv_idx, relevant_dict)
    utils.save2csv('bm25_w_relevance_train.csv', BM25, query_list)