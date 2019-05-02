import numpy as np
from scipy.stats import multivariate_normal
import matplotlib.pyplot as plt
from loadMNIST import *
#***********************************Q4c*****************************************************************

mu1 = np.array([0.1, 0.1])
mu2 = np.array([6.0, 0.1])
va = [10, 10]
cova = 7
VCVmatrix = np.array([[va[0], cova],
                      [cova, va[1]]])
mvn1 = np.random.multivariate_normal(mu1, VCVmatrix, size=200)
mvn2 = np.random.multivariate_normal(mu2, VCVmatrix, size=200)
x1, y1 = mvn1.T
x2, y2 = mvn2.T
xc = np.concatenate((x1, x2))
yc = np.concatenate((y1, y2))
data1 = [mvn1]
data2 = [mvn2]
full_data = np.reshape(data1 +data2, (400, 2))
# full_data = np.concatenate((mvn1, mvn2))
true_label = []
for i in range(200):
    true_label.append(1)
for i in range(200):
    true_label.append(2)
fig = plt.figure()
plt.scatter(xc, yc, 24, c=true_label)
# plt.show()
fig.savefig("true-cluster.png")


def check_eq(a,b):
    if a[0] == b[0] and a[1] == b[1]:
        return False
    else:
        return True


def check_mis(new_c, true_class):
    true_num = 0
    for t in true_class:
        for n in new_c:
            if not check_eq(t, n):
                true_num += 1
    return true_num

def normal_density(data, mean, cov):
    return multivariate_normal.pdf(data, mean, cov)

def log_likelihood(pi_1, pi_2 , data, mean1, cov1, mean2, cov2):
    nd1 = normal_density(data, mean1, cov1)
    nd2 = normal_density(data, mean2, cov2)
    log_lik = np.sum(np.log( np.dot(pi_1, nd1) + np.dot(pi_2, nd2)))

    return log_lik

def em_e_step(pi_1, pi_2 ,data, mean1, cov1, mean2, cov2):

    px1 = normal_density(data, mean1, cov1) * pi_1
    px2 = normal_density(data, mean2, cov2) * pi_2

    r1 = px1/(px1+px2)
    r2 = px2/(px1 +px2)
    return r1, r2

def em_m_step(data, r1, r2):
    new_mean1 = np.dot(r1, data) / np.sum(r1)
    new_mean2 = np.dot(r2, data) / np.sum(r2)

    new_sig1 = np.zeros((2,2))
    new_sig2 = np.zeros((2,2))
    for i in range(data.shape[0]):
        new_sig1 += np.dot( (np.dot(r1[i],(data[i] - new_mean1))).reshape(2,1), (data[i] - new_mean1).reshape(1,2))
        new_sig2 += np.dot( (np.dot(r2[i],(data[i] - new_mean2))).reshape(2,1), (data[i] - new_mean2).reshape(1,2))



    new_sig1 = new_sig1/np.sum(r1)
    new_sig2 = new_sig2/np.sum(r2)

    new_pi1 = np.sum(r1)/400
    new_pi2 = np.sum(r2)/400

    return new_mean1, new_mean2, new_sig1, new_sig2, new_pi1, new_pi2


def gmm_em(pi_1, pi_2 ,data, mean1, cov1, mean2, cov2):
    temp = log_likelihood(pi_1, pi_2, data, mean1, cov1, mean2, cov2)
    r1, r2 = em_e_step(pi_1, pi_2, data, mean1, cov1, mean2, cov2)
    new_mean1, new_mean2, new_sig1, new_sig2, new_pi1, new_pi2 = em_m_step(data, r1, r2)
    log_lik = log_likelihood(new_pi1, new_pi2, data, new_mean1, new_sig1, new_mean2, new_sig2)
    loglst = [log_lik]
    c = 1
    iter_lst = [1]

    while not temp == log_lik:
        temp = log_lik
        r1, r2 = em_e_step(new_pi1, new_pi2, data, new_mean1, new_sig1, new_mean2, new_sig2)
        new_mean1, new_mean2, new_sig1, new_sig2, new_pi1, new_pi2 = em_m_step(data, r1, r2)
        log_lik = log_likelihood(new_pi1, new_pi2, data, new_mean1, new_sig1, new_mean2, new_sig2)
        loglst.append(log_lik)
        c+= 1
        iter_lst.append( c)

    r1, r2 = em_e_step(new_pi1, new_pi2, data, new_mean1, new_sig1, new_mean2, new_sig2)
    c1 = []
    c2 = []
    for i in range(data.shape[0]):
        if r1[i] > r2[i]:
            c1.append(data[i])
        else:
            c2.append(data[i])

    c1 = np.array(c1)
    c2 = np.array(c2)

    num_true1 = check_mis(c1, mvn1)
    num_true2 = check_mis(c2, mvn2)
    print("number of true classified for c1 is", num_true1, "number of true classified for c2 is", num_true2)
    err_rate = 1 - ((num_true1 + num_true2) / data.shape[0])
    print("error rate for EM is", err_rate)
    print("numer of iteration for EM is", c)

    plt.scatter(np.transpose(c1)[0], np.transpose(c1)[1])
    plt.scatter(np.transpose(c2)[0], np.transpose(c2)[1])
    plt.show()

    plt.scatter(iter_lst, loglst)
    plt.show()




    return None

if __name__ == "__main__":
    N_data, train_images, train_labels, test_images, test_labels = load_mnist()
    train_images = 1.0 * (train_images[0:10000] > 0.5)
    train_labels = train_labels[0:10000]
    debug_train_images = 1.0 * (train_images[0:100]>0.5)
    debug_train_labels = train_labels[0:100]
    #print(train_images.shape, train_labels.shape, test_images.shape)
    #save_images(debug_train_images,'img0')



    #*******Q4b*****************************************************
    # data, data1, data2 = generate_data()
    # cost(data, data1, data2, [0.0 ,0.0], [1.0,1.0])
    #
    # *******Q4c*****************************************************
    normal_density(full_data, [0.0, 0.0], [[1, 0],[0, 1]])
    gmm_em(0.5, 0.5, full_data, [0.0, 0.0], [[1, 0],[0, 1]], [1.0, 1.0], [[1, 0],[0, 1]])

