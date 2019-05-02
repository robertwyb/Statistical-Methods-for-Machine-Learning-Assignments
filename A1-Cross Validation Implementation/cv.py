import scipy.io as sio
import random
import numpy as np
import matplotlib.pyplot as plt

# part a  load in mat data
dataset = sio.loadmat('./dataset.mat')
data_train_X = dataset['data_train_X']
data_train_y = dataset['data_train_y'][0]
data_test_X = dataset['data_test_X']
data_test_y = dataset['data_test_y'][0]


# use random.shuffle to get random permutation of data
def shuffle_data(data):
    temp = data[:]
    random.shuffle(temp)
    return temp


def split_data(data, num_folds, fold):
    # get each split's size
    size = int(len(data) / num_folds)
    total = []
    index = 0
    for i in range(num_folds):
        total.append(data[index:(index+size)])
        index += size
    # get specified data_fold based on index fold
    data_fold = total[fold-1]
    total.pop(fold-1)
    data_rest = []
    for i in total:
        data_rest.extend(i)
    return data_fold, data_rest


def train_model(data, lambd):
    # create correct size matrices for X and Y
    y, x = np.empty([1, len(data)]), np.empty([len(data), 400])
    for i in range(len(data)):
        y[0][i] = data[i][0]
        x[i] = data[i][1]
    I = np.identity(400)
    beta = np.dot(np.dot(np.linalg.inv((np.dot(x.transpose(), x) + np.dot(lambd, I))), x.transpose()), y.transpose())
    return beta


def predict_model(data, model):
    res = []
    for d in data:
        res.append(np.dot(d[1].reshape([1, 400]), model.reshape([400, 1])))
    return res


# get summation of all differences then divided by number of value
def loss(data, model):
    prediction = predict_model(data, model)
    real, diff = [], []
    for i in data:
        real.append(i[0])
    for i in range(len(real)):
        diff.append((real[i] - prediction[i]) ** 2)
    return sum(diff)/len(diff)


def cross_validation(data, num_folds, lambd_seq):
    data = shuffle_data(data)
    cv_error = []
    for i in range(50):
        lambd = lambd_seq[i]
        cv_loss_lmd = 0
        for fold in range(1, num_folds+1):
            val_cv, train_cv = split_data(data, num_folds, fold)
            model = train_model(train_cv, lambd)
            cv_loss_lmd += loss(val_cv, model)
        cv_error.append(cv_loss_lmd / num_folds)
    return cv_error


# part b
total_train = []
for i in range(len(data_train_X)):
    total_train.append((data_train_y[i], data_train_X[i]))

total_test = []
for i in range(len(data_test_X)):
    total_test.append((data_test_y[i], data_test_X[i]))

lambd_seq = np.linspace(0.02, 1.5, 50)
cv_5_err = cross_validation(total_train, 5, lambd_seq)
cv_10_err = cross_validation(total_train, 10, lambd_seq)


# part c
def compute_loss(train_data, test_data, lambd_seq):
    train_loss, test_loss  = [], []
    for i in lambd_seq:
        model = train_model(train_data, i)
        train_loss.append(loss(train_data, model))
        test_loss.append(loss(test_data, model))
    return train_loss, test_loss


train_error, test_error = compute_loss(total_train, total_test, lambd_seq)

# part d


def clip_data(data):
    res = []
    for i in data:
        res.append(i[0][0])
    return res


def plot_graph():
    x = np.linspace(0.02, 1.5, 50)
    plt.plot(x, clip_data(train_error), label="train error")
    plt.plot(x, clip_data(test_error), label="test error")
    plt.plot(x, clip_data(cv_5_err), label="5 fold")
    plt.plot(x, clip_data(cv_10_err), label="10 fold")
    plt.xlabel('lambda')
    plt.ylabel('loss')
    plt.legend()
    plt.show()


plot_graph()
