import utils
from metrics import calculate_mAP, calculate_NDCG
import numpy as np
from sklearn.neural_network import MLPClassifier


def top100NN(pred):
    '''calculate top 100 LM results'''
    NN_result = []
    one_query = []
    for i in range(len(pred) - 1):
        one_query.append(list(pred[i]))

        # if the next q is different
        if pred[i][1] != pred[i+1][1]:
            # sort by the score
            one_query.sort(key=lambda x:x[3], reverse=True)
            NN_result.append(one_query[:100])
            one_query = []
    return NN_result


def saveNN(NN_result):
    '''save top 100 NN results'''
    output_file = open('NN.txt', 'w')
    first_row = True
    for _, que in enumerate(NN_result):
        rank = 0
        for _, row in enumerate(que):
            if not first_row:
                output_file.write('\n')
            rank += 1
            # qid1 A1 pid1 rank1 score1 algoname2
            config = f'{int(row[1])} A1 {int(row[2])} {rank} {row[3]} NN'
            output_file.write(config)
            first_row = False

    output_file.close()


def NN_dictionary(NN_ranking):
    '''save the ranking results to dictionary for later analysis'''
    evaluate_dict = {}
    for row in NN_ranking:
        qid = row[0]
        pid = row[2]
        if qid not in evaluate_dict:
            evaluate_dict[qid] = [pid]
        else:
            evaluate_dict[qid].append(pid)
    
    return evaluate_dict


def run_save_MLP(lr_rate, hidden_layer_sizes, X, y, test):
    '''run MLP model and save the results'''
    clf = MLPClassifier(alpha=lr_rate, hidden_layer_sizes=hidden_layer_sizes)
    clf.fit(X, y)

    ans = clf.predict_proba(test)
    score = ans[:, 1]
    score = score.reshape(score.shape[0], 1)
    pred = np.concatenate((test_label, score), axis=1)

    NN_result = top100NN(pred)
    saveNN(NN_result)

    # read the saved LM results
    with open("NN.txt", "r") as tf:
        lines = tf.read().split('\n')
    NN_ranking = []
    for line in lines:
        NN_ranking.append(line.split())

    return NN_ranking


def tune_MLP(X, y, test, validation_data):
    '''tune MLP parameters'''
    best_NDCG = 0
    best_pairs = []
    relevant_dict = utils.get_relevance(validation_data)
    for i in [0.5, 0.1, 0.01]:
        for j in [(5,5), (5,10), (5,), (10,)]:
            NDCG_sum = 0
            # repeat the training 3 times to get a more reliable result
            print(f'alpha = {i}, hidden_layer_sizes = {j}')
            for _ in range(3):
                NN_ranking = run_save_MLP(i, j, X, y, test)
                NNdict = NN_dictionary(NN_ranking)
                NDCG_sum += sum(calculate_NDCG(relevant_dict, NNdict, False))

            if NDCG_sum > best_NDCG:
                best_NDCG = NDCG_sum
                best_pairs = [i, j]
                print(f'\t New optimal found: alpha = {i}, hidden_layer_sizes = {j}')

    return best_pairs


if __name__ == '__main__':
    # load the data representations
    train = np.loadtxt('DataRep_train.csv', delimiter=',')
    val = np.loadtxt('DataRep_val.csv', delimiter=',')
    test = np.loadtxt('DataRep_test.csv', delimiter=',')

    train_label = train[int(len(train)/2):]
    val_label = val[int(len(val)/2):]
    test_label = test[int(len(test)/2):]
    train = train[:int(len(train)/2)]
    val = val[:int(len(val)/2)]
    test = test[:int(len(test)/2)]

    # this is for calculating metrics
    validation_data = utils.open_tsv("validation_data.tsv", '\t')

    train_mix = np.vstack((train, val))
    train_label_mix = np.vstack((train_label, val_label))

    # tune the parameters
    print('Start tuning parameters')
    best_pairs = tune_MLP(train_mix, train_label_mix[:, 0], test, validation_data)

    # train and evaluate 
    print(f'Start running the final model with alpha = {best_pairs[0]}, hidden_layer_sizes = {best_pairs[1]}')
    NN_ranking = run_save_MLP(best_pairs[0], best_pairs[1], train_mix, train_label_mix[:, 0], test)
    NNdict = NN_dictionary(NN_ranking)
    relevant_dict = utils.get_relevance(validation_data)

    mAP = calculate_mAP(relevant_dict, NNdict)
    NDCG = calculate_NDCG(relevant_dict, NNdict, True)