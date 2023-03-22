import utils
from LogisticRegression import LogisticRegression
from metrics import calculate_mAP, calculate_NDCG
import matplotlib.pyplot as plt
from gensim.models.word2vec import Word2Vec
import numpy as np

def tokenize_data(data):
    '''tokenize all the passages and queries'''
    passage_dict = {}
    query_dict = {}

    for i,row in enumerate(data):
        qid = row[0]
        pid = row[1]

        if qid not in query_dict:
            tokened = utils.pre_process(row[2])
            query_dict[qid] = tokened

        if pid not in passage_dict:
            tokened = utils.pre_process(row[3])
            passage_dict[pid] = tokened

        if i % int(len(data) * 0.2) == 0:
            print(f'Finished {i}/{len(data)}')

    print(f'Finished {i}/{len(data)}')

    # remove the headings
    del passage_dict['pid']
    del query_dict['qid']

    return passage_dict, query_dict


def get_relevance_score(data):
    '''store the relevant score 0/1 with qid'''
    relevant_dict = {}
    for row in data:
        try:
            if float(row[-1]) != 0.0:
                qid = row[0]
                pid = row[1]
                if qid not in relevant_dict:
                    relevant_dict[qid] = 0
        except:
            pass

    return relevant_dict


def sentence_embedding(qid, pid, tok_que, tok_psg, Word2Vec_model):
    '''find the embedding of the terms in a sentence'''
    # some words might not appear, skip them with try/except
    qid_vec = []
    for word in tok_que[qid]:
        try:
            qid_vec.append(Word2Vec_model.wv[word])
        except:
            pass

    pid_vec = []
    for word in tok_psg[pid]:
        try:
            pid_vec.append(Word2Vec_model.wv[word])
        except:
            pass
    
    return qid_vec, pid_vec


def create_dataset(data, Word2Vec_model, tok_psg, tok_que):
    '''convert the whole dataset to sentence embeddings'''
    # remove the heading
    data = data[1:]

    relevant_dict = get_relevance_score(data)

    x = []
    y = []

    for i,row in enumerate(data):
        qid = row[0]
        pid = row[1]

        # train set with relevant pairs, duplicate the pairs 200 times
        if len(data) > 2e6 and float(row[4]) > 0:
            qid_vec, pid_vec = sentence_embedding(qid, pid, tok_que, tok_psg, Word2Vec_model)

            if len(qid_vec) != 0 and len(pid_vec) != 0:
                for _ in range(200):
                    x.append([np.mean(qid_vec), np.mean(pid_vec)])
                    y.append([float(data[i][4]), int(qid), int(pid)])


        # train set with non-relevant pairs, only keep at most 200 pairs
        elif (len(data) > 2e6 and relevant_dict[qid] < 200):
            
            qid_vec, pid_vec = sentence_embedding(qid, pid, tok_que, tok_psg, Word2Vec_model)

            if len(qid_vec) != 0 and len(pid_vec) != 0:
                x.append([np.mean(qid_vec), np.mean(pid_vec)])
                y.append([float(data[i][4]), int(qid), int(pid)])
                relevant_dict[qid] += 1


        # validation set, keep all of them
        elif len(data) < 2e6:
            # some words might not appear
            qid_vec, pid_vec = sentence_embedding(qid, pid, tok_que, tok_psg, Word2Vec_model)

            if len(qid_vec) != 0 and len(pid_vec) != 0:
                x.append([np.mean(qid_vec), np.mean(pid_vec)])
                y.append([float(data[i][4]), int(qid), int(pid)])


    print('Finish creating dataset')
    return np.array(x), np.array(y)


def shuffle_xy(x, y):
    '''shuffle x,y data without messing up the orders'''
    rand_idx = np.random.permutation(len(x)) #return a random index
    x = x[rand_idx]
    y = y[rand_idx]

    return x, y


def analyze_LR(xtrain, ytrain, xval):
    lr_range = [0.01, 0.1, 1, 5]
    lr_loss = []
    opt_loss_pair = [100, 0]
    for lr in lr_range:
        LR = LogisticRegression()
        loss = LR.fit(xtrain, ytrain[:, 0], lr_rate=lr, iter=300, show_loss_per=np.Inf) # change this to print loss
        lr_loss.append(loss)

        if loss[-1] < opt_loss_pair[0]:
            opt_loss_pair = [loss[-1], lr]

    opt_LR = LogisticRegression()
    loss = opt_LR.fit(xtrain, ytrain[:, 0], lr_rate=opt_loss_pair[1], iter=200, show_loss_per=np.Inf)
    val_pred = opt_LR.predict(xval)

    # visualize the effect of learning rate
    for i in range(len(lr_loss)):
        plt.plot(lr_loss[i], label=f'lr = {lr_range[i]}')
    plt.legend()
    plt.xlabel("Iteration")
    plt.ylabel("loss")
    plt.savefig('LR.png')

    return val_pred


def save_LRresults(val_pred, yval):
    val_pred = val_pred.reshape(val_pred.shape[0], 1)
    val_pred = np.concatenate((yval, val_pred), axis=1)

    # sort by qid
    val_pred_sorted = val_pred[val_pred[:, 1].argsort()]

    # find the top 100 results
    LR_result = []
    one_query = []
    for i in range(len(val_pred_sorted) - 1):
        one_query.append(list(val_pred_sorted[i]))

        # if the next q is different
        if val_pred_sorted[i][1] != val_pred_sorted[i+1][1]:
            # sort by the score
            one_query.sort(key=lambda x:x[3], reverse=True)
            LR_result.append(one_query[:100])
            one_query = []

    # store the top 100 results
    output_file = open('LR.txt', 'w')
    first_row = True
    for _, que in enumerate(LR_result):
        rank = 0
        for _, row in enumerate(que):
            if not first_row:
                output_file.write('\n')
            rank += 1
            # qid1 A1 pid1 rank1 score1 algoname2
            config = f'{int(row[1])} A1 {int(row[2])} {rank} {row[3]} LR'
            output_file.write(config)
            first_row = False

    output_file.close()
    

def LR_dictionary(LR_ranking):
    evaluate_dict = {}
    for row in LR_ranking:
        qid = row[0]
        pid = row[2]
        if qid not in evaluate_dict:
            evaluate_dict[qid] = [pid]
        else:
            evaluate_dict[qid].append(pid)
    return evaluate_dict


if __name__ == '__main__':
    print('Start Logistic Regression')
    
    train = utils.open_tsv("train_data.tsv", '\t')
    val = utils.open_tsv("validation_data.tsv", '\t')

    print('Tokenizing data...')
    train_psg, train_que = tokenize_data(train)
    val_psg, val_que = tokenize_data(val)

    print('Training Word2Vec...')
    train_word2vec = list(train_psg.values()) + list(val_psg.values())
    model = Word2Vec(train_word2vec)

    xtrain, ytrain = create_dataset(train, model, train_psg, train_que)
    xtrain, ytrain = shuffle_xy(xtrain, ytrain)

    xval, yval = create_dataset(val, model, val_psg, val_que)

    # save the embeddings to csv files
    np.savetxt('AvgEmbedding_xtrain.csv', xtrain, delimiter=',')
    np.savetxt('AvgEmbedding_ytrain.csv', ytrain, delimiter=',')
    np.savetxt('AvgEmbedding_xval.csv', xval, delimiter=',')
    np.savetxt('AvgEmbedding_yval.csv', yval, delimiter=',')

    print('Evaluating LR results... ')
    # get the predictions on val set and save it
    val_pred = analyze_LR(xtrain, ytrain, xval)
    save_LRresults(val_pred, yval)

    # load the saved LR results
    with open("LR.txt", "r") as tf:
        lines = tf.read().split('\n')
    LR_ranking = []
    for line in lines:
        LR_ranking.append(line.split())

    # convert LR result to a dictionary
    LR_dict = LR_dictionary(LR_ranking)
    relevant_dict = utils.get_relevance(val)

    # get the metrics
    mAP = calculate_mAP(relevant_dict, LR_dict)
    NDCG = calculate_NDCG(relevant_dict, LR_dict, True)