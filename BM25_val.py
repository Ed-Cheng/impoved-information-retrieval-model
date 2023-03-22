import utils 


def sort_data(data):
    '''convert data to the required format for calculating bm25'''
    # create the top1000 var in bm25
    top1000 = []
    query_dict = {}
    for row in data:
        qid = row[0]
        pid_len = len(row[3].split())
        temp_row = row[:4]
        temp_row.append(pid_len)
        top1000.append(temp_row)

        if qid not in query_dict:
            query_dict[qid] = row[2]

    # remove the headings
    del query_dict['qid']
    del top1000[0]

    # convert query to a list
    query_list = [[qid, query] for qid, query in query_dict.items()] 

    return top1000, query_list


if __name__ == '__main__':
    print('Start calculating BM25 on validation set')

    val = utils.open_tsv("validation_data.tsv", "\t")

    # sort the data into the format for inverted index and BM25
    top1000, query_list = sort_data(val)

    # get the inverted index
    inverted_index = utils.get_inv_idx(top1000)

    # store the inverted index
    utils.save_inv_idx(inverted_index, "inverted_index.pkl")

    # store the relevance into a dictionary 
    relevant_dict = utils.get_relevance(val)

    # open the stored inverted index
    inv_idx = utils.open_pkl("inverted_index.pkl")

    # number of documents in collection
    N = len(top1000)

    # BM25 score
    BM25 = utils.top100BM25(top1000, query_list, N, inv_idx, relevant_dict)
    utils.save2csv('bm25_w_relevance.csv', BM25, query_list)

