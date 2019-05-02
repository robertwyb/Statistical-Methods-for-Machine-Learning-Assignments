from __future__ import absolute_import
from __future__ import print_function
from future.standard_library import install_aliases
import math
from scipy.special import softmax
from scipy.special import logsumexp
install_aliases()

import numpy as np
import os
import gzip
import struct
import array

import matplotlib.pyplot as plt
import matplotlib.image
from urllib.request import urlretrieve


def download(url, filename):
    if not os.path.exists('data'):
        os.makedirs('data')
    out_file = os.path.join('data', filename)
    if not os.path.isfile(out_file):
        urlretrieve(url, out_file)


def mnist():
    base_url = 'http://yann.lecun.com/exdb/mnist/'

    def parse_labels(filename):
        with gzip.open(filename, 'rb') as fh:
            magic, num_data = struct.unpack(">II", fh.read(8))
            return np.array(array.array("B", fh.read()), dtype=np.uint8)

    def parse_images(filename):
        with gzip.open(filename, 'rb') as fh:
            magic, num_data, rows, cols = struct.unpack(">IIII", fh.read(16))
            return np.array(array.array("B", fh.read()), dtype=np.uint8).reshape(num_data, rows, cols)

    for filename in ['train-images-idx3-ubyte.gz',
                     'train-labels-idx1-ubyte.gz',
                     't10k-images-idx3-ubyte.gz',
                     't10k-labels-idx1-ubyte.gz']:
        download(base_url + filename, filename)

    train_images = parse_images('data/train-images-idx3-ubyte.gz')
    train_labels = parse_labels('data/train-labels-idx1-ubyte.gz')
    test_images  = parse_images('data/t10k-images-idx3-ubyte.gz')
    test_labels  = parse_labels('data/t10k-labels-idx1-ubyte.gz')

    return train_images, train_labels, test_images, test_labels


def load_mnist():
    partial_flatten = lambda x : np.reshape(x, (x.shape[0], np.prod(x.shape[1:])))
    one_hot = lambda x, k: np.array(x[:,None] == np.arange(k)[None, :], dtype=int)
    train_images, train_labels, test_images, test_labels = mnist()
    train_images = partial_flatten(train_images) / 255.0
    test_images  = partial_flatten(test_images)  / 255.0
    train_labels = one_hot(train_labels, 10)
    test_labels = one_hot(test_labels, 10)
    N_data = train_images.shape[0]

    return N_data, train_images, train_labels, test_images, test_labels


def plot_images(images, ax, ims_per_row=5, padding=5, digit_dimensions=(28, 28),
                cmap=matplotlib.cm.binary, vmin=None, vmax=None):
    """Images should be a (N_images x pixels) matrix."""
    N_images = images.shape[0]
    N_rows = np.int32(np.ceil(float(N_images) / ims_per_row))
    pad_value = np.min(images.ravel())
    concat_images = np.full(((digit_dimensions[0] + padding) * N_rows + padding,
                             (digit_dimensions[1] + padding) * ims_per_row + padding), pad_value)
    for i in range(N_images):
        cur_image = np.reshape(images[i, :], digit_dimensions)
        row_ix = i // ims_per_row
        col_ix = i % ims_per_row
        row_start = padding + (padding + digit_dimensions[0]) * row_ix
        col_start = padding + (padding + digit_dimensions[1]) * col_ix
        concat_images[row_start: row_start + digit_dimensions[0],
                      col_start: col_start + digit_dimensions[1]] = cur_image
    cax = ax.matshow(concat_images, cmap=cmap, vmin=vmin, vmax=vmax)
    plt.xticks(np.array([]))
    plt.yticks(np.array([]))
    return cax


def save_images(images, filename, **kwargs):
    fig = plt.figure(1)
    fig.clf()
    ax = fig.add_subplot(111)
    plot_images(images, ax, **kwargs)
    fig.patch.set_visible(False)
    ax.patch.set_visible(False)
    plt.savefig(filename)


N_data, train_images, train_labels, test_images, test_labels = load_mnist()


# -------------------- q1c --------------------
def q1c(train_images, train_labels):

    theta = np.zeros((10, 784))
    train_labels = np.argmax(train_labels, axis = 1)
    count_label =np.zeros(10)
    for i in range(train_labels.shape[0]):
        theta[train_labels[i]] += train_images[i]
        count_label[train_labels[i]] += 1
    for i in range(10):
        theta[i] = (theta[i]+1) / (count_label[i] + 2)
    save_images(theta, 'theta1.png')
    return theta


theta = q1c(train_images, train_labels)
# -------------------------- q1e -------------------

clip_train_labels = np.argmax(train_labels[:10000], axis=1)
clip_train_images = train_images[:10000]
clip_bi_train_images = 1.0 *(train_images[:10000] > 0.5)
clip_test_labels = np.argmax(test_labels[:10000], axis=1)
clip_test_images = test_images[:10000]
clip_bi_test_images = 1.0 *(test_images[:10000] > 0.5)


def p_x_given_c_theta(theta, x, c):
    ones = np.ones((1, 784))
    return theta[c, :] ** x.reshape(1, 784) * (1-theta[c, :]) ** np.subtract(ones, x.reshape(1, 784))


def p_xc_given_theta_pi(theta, x, c):
    result = np.prod(p_x_given_c_theta(theta, x, c))
    result /= 10
    return result


def p_c_given_x(theta, data, c):
    num = p_xc_given_theta_pi(theta, data, c)
    denom = 0
    for i in range(10):
        denom += p_xc_given_theta_pi(theta, data, i)
    return num / denom


# compute average log likelihood
def average_log_llh(theta, data, labels):
    result = 0
    for i in range(10000):
        if p_c_given_x(theta, data[i], labels[i]) == 0:
            print("error")
        log = math.log(p_c_given_x(theta, data[i], labels[i]))
        result += log
    return result / 10000


def accuracy(theta, images, labels):
    prediction = []
    for i in range(10000):
        computed_p = []
        for c in range(10):
            p = p_c_given_x(theta, images[i], c)
            # print(p)
            computed_p.append(p)
        prediction.append(np.argmax(computed_p))
        # print(prediction)
    labels = np.array(labels)
    prediction = np.array(prediction)
    correct = prediction[prediction==labels]
    correct_count = correct.shape[0]
    return correct_count / len(labels)


def q1e(theta, train_images, train_labels, test_images, test_labels):
    print("Training set accuracy: {}".format(accuracy(theta, train_images, train_labels)))
    print("Test set accuracy: {}".format(accuracy(theta, test_images, test_labels)))
    print(average_log_llh(theta, train_images, train_labels))
    print(average_log_llh(theta, test_images, test_labels))


q1e(theta, clip_train_images, clip_train_labels, clip_test_images, clip_test_labels)


# ---------------------- q2c ---------------------------
def q2c(theta):
    image = np.zeros((10, 784))
    for iter in range(1, 11):
        index = np.random.choice(np.arange(10), 1, p=[0.1] * 10)
        print(index)
        sample = []
        for i in range(784):
            prob = [theta[index, i][0], 1 - theta[index, i][0]]
            print(prob)
            s = np.random.choice(np.array([1, 0]), 1, p=prob)
            sample.append(s)
        sample = np.array(sample).reshape((1, 784))
        image[iter-1,:] = sample
    save_images(image, 'random_sample.png')

# q2c(theta)

# ------------------------ q2e --------------------------
def q2e(x, theta):
    x_top = x[:,:392]
    theta_top = theta[:, :392]
    joint_top = np.exp(np.dot(x_top, np.log(theta_top.T)) + np.dot(1-x_top, np.log(1-theta_top.T)))[:20]
    full_new_theta = []
    for i in range(20):
        new_theta = []
        for j in range(392, 784):
            joint_bot = np.exp(x[i,j] * np.log(theta[:, j]) + (1-x[i,j]) * (1-np.log(1-theta[:,j])))
            t1 = np.dot(theta[:, j], joint_top[i] * joint_bot)
            t2 = np.dot(1 - theta[:, j], joint_top[i] * joint_bot)
            new_theta.append(t1/(t1+t2))
        full_new_theta.append(new_theta)
    result = np.zeros((20, 784))
    for i in range(20):
        result[i][:392] = x[i][:392]
        result[i][392:] = full_new_theta[i]
    save_images(result, "q2e.png")

# q2e(clip_bi_train_images, theta)

# ---------------------- q3c ---------------------------------



def compute_prob(x, w):
    return softmax(np.dot(w, x))


def gd(x, c, w):
    result = np.zeros((10, 784))
    result[c, ] = x
    x_tile = np.tile(x, (10, 1))
    result -= np.dot(np.diag(compute_prob(x, w)), x_tile)
    return result

def lr(x, y):
    w = np.zeros((10, 784))
    ratio = x.shape[0] // 10000
    for i in range(50):
        for j in range(ratio):
            t1, t2 = j * 10000, (j+1) * 10000
            x_b = x[t1:t2]
            y_b = y[t1:t2]
            dw = sum([gd(x_b[k], y_b[k], w) for k in range(10000)])
            w += 0.01 * dw
    save_images(w, "q3c.png")
    return w

w = lr(clip_bi_train_images, clip_train_labels)

def compute_llh_acc(x, y, w):
    count, avg = 0, 0
    for i in range(10000):
        prediction = np.argmax(compute_prob(x[i], w))
        real = np.argmax(y[i])
        if prediction == real:
            count += 1

        avg += np.log(np.dot(y[i].reshape(10, 1).T, compute_prob(x[i], w)))
    avg /= 10000
    return count / 10000, avg


def q3d(train_images, train_labels, test_images, test_labels, w):
    train_acc, train_avg = compute_llh_acc(train_images, train_labels, w)
    test_acc, test_avg = compute_llh_acc(test_images, test_labels, w)
    print("Training accuracy: {}".format(train_acc))
    print("Training average log likelihood: {}".format(train_avg))
    print("Test accuracy: {}".format(test_acc))
    print("Test average log likelihood: {}".format(test_avg))

# q3d(clip_train_images, train_labels[:10000], clip_test_labels, test_labels[:10000], w)
# ------------------------ q3d -----------------------------------
