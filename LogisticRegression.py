import numpy as np

class LogisticRegression:
    def _init__(self):
        self.weight = None
        
    def sigmoid(self, x, weight):
        dot_product = np.dot(x, weight)
        return 1 / (1 + np.exp(-dot_product))

    def cal_loss(self, sig, ytrain):
        return (-ytrain * np.log(sig) - (1 - ytrain) * np.log(1 - sig)).mean()

    def cal_gradient(self, xtrain, ytrain, sig):
        return np.dot(xtrain.T, (sig-ytrain)) / ytrain.shape[0]
        

    def fit(self, xtrain, ytrain, lr_rate, iter, show_loss_per):
        xtrain = np.concatenate((np.ones((xtrain.shape[0], 1)), xtrain), axis=1)
        self.weight = np.zeros(xtrain.shape[1])
        loss_history = []

        for i in range(iter):
            sig = self.sigmoid(xtrain, self.weight)
            loss = self.cal_loss(sig, ytrain)
            grad = self.cal_gradient(xtrain, ytrain, sig)
            self.weight -= lr_rate * grad
            loss_history.append(loss)

            if i % show_loss_per == 0:
                print(f'Iter {i}, loss = {loss}')

        return loss_history

    def predict(self, xval):
        Xnew = np.concatenate((np.ones((xval.shape[0], 1)), xval), axis=1)
        return np.dot(Xnew, self.weight)

    def evaluate(self, ytrue, ypred):
        # true positive
        tp = np.sum(np.where(ypred == ytrue, ypred, 0))
        # true negative
        tn = np.sum(np.where(ypred == ytrue, 1, 0)) - tp
        # false positive
        fp = np.sum(np.where(ypred != ytrue, ypred, 0))
        # false negative
        fn = np.sum(np.where(ypred != ytrue, 1, 0)) - fp

        accuracy = (tp+tn) / ytrue.shape[0]
        percision = tp / (tp+fp)
        recall = tp / (tp+fn)

        return [accuracy, percision, recall]