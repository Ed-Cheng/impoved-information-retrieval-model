import utils 
from metrics import calculate_mAP, calculate_NDCG


def bm25_dictionary(bm25_data):
    # convert bm25 file to dictionary to run the evaluation
    BM25_dict = {}
    for row in bm25_data:
        ids = row[0].split(',')
        qid = ids[0]
        pid = ids[1]
        if qid not in BM25_dict:
            BM25_dict[qid] = [pid]
        else:
            BM25_dict[qid].append(pid)
    
    return BM25_dict


if __name__ == '__main__':
    print('Evaluating Retrieval Quality on BM25')

    val = utils.open_tsv("validation_data.tsv", "\t")

    # open the saved bm25 file
    BM25_saved = utils.open_tsv("bm25_w_relevance.csv", "\t")

    # convert bm25 to a dictionary
    BM25_dict = bm25_dictionary(BM25_saved)

    relevant_dict = utils.get_relevance(val)

    # get the metrics
    mAP = calculate_mAP(relevant_dict, BM25_dict)
    NDCG = calculate_NDCG(relevant_dict, BM25_dict, True)
