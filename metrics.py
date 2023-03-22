'''
The evaluation metrics for all the tasks
'''

import numpy as np

def discounted_gain(rank):
    '''return discounted gain while gain is always 1 in our case'''
    return 1 / (np.log2(rank+1))


def calculate_mAP(relevant_dict, evaluate_dict):
    '''calculate the mean average percision of the top 3/10/100 ranking'''
    cutoff_points = [3,10,100]
    results = []

    for cutoff in cutoff_points:
        all_avg_precision = []
        for qid in relevant_dict.keys():
            relevant_pid = relevant_dict[qid]
            precision_score = []

            # some relevant pid is not in the top 100 BM25 result
            for pid in relevant_pid:
                try:
                    relevant_idx = evaluate_dict[qid].index(pid) + 1
                except:
                    relevant_idx = np.Inf
                if relevant_idx > cutoff:
                    relevant_idx = np.Inf

                precision_score.append((len(precision_score)+1) / relevant_idx)

            # calculate the average precision
            avg_precision = sum(precision_score) / len(precision_score)
            all_avg_precision.append(avg_precision)

        mean_avg_precision = sum(all_avg_precision) / len(all_avg_precision)
        print(f'mAP @ {cutoff}: \t{mean_avg_precision}')
        results.append(mean_avg_precision)
    return results


def calculate_NDCG(relevant_dict, evaluate_dict, print_result=True):
    '''calculate the NDCG of the top 3/10/100 ranking'''
    cutoff_points = [3,10,100]
    results = []
    for cutoff in cutoff_points:
        all_NDCG = []
        for qid in relevant_dict.keys():
            relevant_pid = relevant_dict[qid]
            dis_gain = []

            # some relevant pid is not in the top 100 BM25 result
            for pid in relevant_pid:
                try:
                    relevant_idx = evaluate_dict[qid].index(pid) + 1
                except:
                    relevant_idx = np.Inf
                if relevant_idx > cutoff:
                    relevant_idx = np.Inf

                dis_gain.append(discounted_gain(relevant_idx))
                # print(relevant_idx, dis_gain)

            DCG = sum(dis_gain)
            optDCG = sum([discounted_gain(i) for i in range(1, len(dis_gain)+1)])
            all_NDCG.append(DCG/optDCG)


        avg_NDCG = sum(all_NDCG) / len(all_NDCG)
        if print_result:
            print(f'NDCG @ {cutoff}: \t{avg_NDCG}')
        results.append(avg_NDCG)

    return results