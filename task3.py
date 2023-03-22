import utils
from metrics import calculate_mAP, calculate_NDCG
import numpy as np

import xgboost as xgb


def top100LM(pred):
    '''calculate top 100 LM results'''
    LM_result = []
    one_query = []
    for i in range(len(pred) - 1):
        one_query.append(list(pred[i]))

        # if the next q is different
        if pred[i][1] != pred[i+1][1]:
            # sort by the score
            one_query.sort(key=lambda x:x[3], reverse=True)
            LM_result.append(one_query[:100])
            one_query = []
    return LM_result


def saveLM(LM_result):
    '''save top 100 LM results'''
    output_file = open('LM.txt', 'w')
    first_row = True
    for _, que in enumerate(LM_result):
        rank = 0
        for _, row in enumerate(que):
            if not first_row:
                output_file.write('\n')
            rank += 1
            # qid1 A1 pid1 rank1 score1 algoname2
            config = f'{int(row[1])} A1 {int(row[2])} {rank} {row[3]} LM'
            output_file.write(config)
            first_row = False

    output_file.close()


def LM_dictionary(LM_ranking):
    '''save the ranking results to dictionary for later analysis'''
    evaluate_dict = {}
    for row in LM_ranking:
        qid = row[0]
        pid = row[2]
        if qid not in evaluate_dict:
            evaluate_dict[qid] = [pid]
        else:
            evaluate_dict[qid].append(pid)
    
    return evaluate_dict


def tuning(param, tuned_param, dtrain):
    '''train on a pair of parameter'''
    for key in tuned_param.keys():
        param[key] = tuned_param[key]

    cv_results = xgb.cv(param,
                        dtrain,
                        num_boost_round=200,
                        nfold=3,
                        metrics={'ndcg@10'},
                        early_stopping_rounds=10)

    max_ndcg10 = cv_results['test-ndcg@10-mean'].max()

    return max_ndcg10


def grid_search_2params(tune_dict, param, dtrain):
    '''tuning 2 XGBoost parameters at once'''
    optimal = 0
    two_keys = list(tune_dict.keys())
    for param1 in tune_dict[two_keys[0]]:
        for param2 in tune_dict[two_keys[1]]:
            tuned_param = {two_keys[0]: param1,
                            two_keys[1]: param2}
            
            # print(f'max_depth: {depth}, min_child_weight: {child_weight}')
            max_ndcg10 = tuning(param, tuned_param, dtrain)
            if max_ndcg10 > optimal:
                optimal = max_ndcg10
                best_param = tuned_param
                print(f'max_ndcg10: {max_ndcg10} {two_keys[0]}: {param1} {two_keys[1]} {param2}')
    print('Finish tuning for this pair')
    return best_param


def update_param(param, updates):
    '''update tuned parameters'''
    for key in updates.keys():
        param[key] = updates[key]
    return param


def LM_dictionary(LM_ranking):
    evaluate_dict = {}
    for row in LM_ranking:
        qid = row[0]
        pid = row[2]
        if qid not in evaluate_dict:
            evaluate_dict[qid] = [pid]
        else:
            evaluate_dict[qid].append(pid)
    return evaluate_dict


if __name__ == '__main__':
    print('Start LambdaMART')

    # load the data representations
    train = np.loadtxt('DataRep_train.csv', delimiter=',')
    train_group = np.loadtxt('DataRep_train_group.csv', delimiter=',')
    val = np.loadtxt('DataRep_val.csv', delimiter=',')
    val_group = np.loadtxt('DataRep_val_group.csv', delimiter=',')
    test = np.loadtxt('DataRep_test.csv', delimiter=',')
    test_group = np.loadtxt('DataRep_test_group.csv', delimiter=',')

    train_label = train[int(len(train)/2):]
    val_label = val[int(len(val)/2):]
    test_label = test[int(len(test)/2):]
    train = train[:int(len(train)/2)]
    val = val[:int(len(val)/2)]
    test = test[:int(len(test)/2)]

    # this is for calculating metrics
    validation_data = utils.open_tsv("validation_data.tsv", '\t')

    # convert data to XGBoost data type 
    print('Converting data type')
    dtrain = xgb.DMatrix(train, label=train_label[:, 0])
    dtrain.set_group(np.array(train_group))

    dval = xgb.DMatrix(val, label=val_label[:, 0])
    dval.set_group(np.array(val_group))

    dtest = xgb.DMatrix(test, label=test_label[:, 0])
    dtest.set_group(np.array(test_group))

    ### initial parameters for XGBoost, start the tuning process ###
    param = {'objective': 'rank:pairwise',
        'eta': 0.5,
        'gamma': 1.0,
        'max_depth': 5,
        'min_child_weight': 1,
        'eval_metric': 'ndcg@10'}
    print('Tuning hyper-parameter, initial parameters:')
    print(param)

    evallist = [(dval, 'train')]

    tuning_1 = {'max_depth': range(1,6),
            'min_child_weight': range(1,6)}

    tuning_2 = {'eta': [i/20 for i in range(3,10)],
            'gamma': [i/20 for i in range(0,6)]}

    best_param_one = grid_search_2params(tuning_1, param, dtrain)
    param = update_param(param, best_param_one)

    best_param_two = grid_search_2params(tuning_2, param, dtrain)
    param = update_param(param, best_param_two)
    print('Finish tuning, the result is:')
    print(param)
    ################ Finish fine tuning ################

    num_round = 500
    bst = xgb.train(param, dtrain, num_round, evallist, early_stopping_rounds=20)

    pred = bst.predict(dtest)
    pred = pred.reshape(pred.shape[0], 1)
    pred = np.concatenate((test_label, pred), axis=1)

    LM_result = top100LM(pred)
    saveLM(LM_result)

    # read the saved LM results
    with open("LM.txt", "r") as tf:
        lines = tf.read().split('\n')
    LM_ranking = []
    for line in lines:
        LM_ranking.append(line.split())

    relevant_dict = utils.get_relevance(validation_data)
    LM_dict = LM_dictionary(LM_ranking)

    mAP = calculate_mAP(relevant_dict, LM_dict)
    NDCG = calculate_NDCG(relevant_dict, LM_dict, True)