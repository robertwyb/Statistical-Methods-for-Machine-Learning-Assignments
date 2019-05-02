import numpy as np
from matplotlib import pyplot as plt
import pandas as pd
import random
from scipy.stats import norm
from scipy.stats import multivariate_normal
from sklearn.cluster import KMeans
from sklearn.mixture import GaussianMixture as GMM

# part a

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
full_data = np.concatenate((mvn1, mvn2))
true_label = []
for i in range(200):
    true_label.append(1)
for i in range(200):
    true_label.append(2)
fig = plt.figure()
plt.scatter(xc, yc, 24, c=true_label)
# plt.show()
fig.savefig("true-cluster.png")





# part b
mu1b = np.array([0.0, 0.0])
mu2b = np.array([1.0, 1.0])


class K_Means:
    def __init__(self, k=2, tol=0.0001, max_iter=50):
        self.k = k
        self.tol = tol
        self.max_iter = max_iter

    def cost(self, dp):
        return [np.linalg.norm(dp-self.centroids[c]) for c in range(2)]

    def km_e_step(self, data):
        self.classifications = {}
        cost = 0
        for i in range(self.k):
            self.classifications[i] = []
        for d in data:
            cost_lst = self.cost(d)
            min_idx = cost_lst.index(min(cost_lst))
            self.classifications[min_idx].append((d[0], d[1]))
            cost += sum(cost_lst)
        return cost/20000

    def km_m_step(self):
        for c in range(len(self.classifications)):
            self.centroids[c] = np.average(self.classifications[c], axis=0)

    def fit(self,data):

        # initialize centroids randomly
        self.centroids = {}
        idx_lst = []
        for i in range(self.k):
            idx_lst.append(random.randint(0, 400))
        for i in range(self.k):
            self.centroids[i] = data[idx_lst[i]]
        print("Initial centroids: " + str(self.centroids))
        optimized = False
        i = 0
        log_lst, iter_lst = [], []
        while i < self.max_iter and not optimized:
            i += 1
            iter_lst.append(i)
            cost = self.km_e_step(data)
            log_lst.append(cost)
            prev_centroids = dict(self.centroids)
            self.km_m_step()

            for c in range(self.k):
                original_centroid = prev_centroids[c]
                current_centroid = self.centroids[c]
                print(original_centroid, current_centroid)
                diff = np.sum((current_centroid-original_centroid)/original_centroid*100.0)
                print("Iteration: {}. Shift: {}".format(i, diff))
                if abs(diff) < self.tol:
                    optimized = True
                    print("finished")
            fig = plt.figure()
            for cent in range(2):
                plt.scatter(self.centroids[cent][0], self.centroids[cent][1])
            label = [0] * len(self.classifications[0]) + [1] * len(self.classifications[1])
            full_cluster = np.concatenate((self.classifications[0], self.classifications[1]))
            x_pos = full_cluster[:, 0]
            y_pos = full_cluster[:, 1]
            plt.scatter(x_pos, y_pos, 24, c=label)
            fig.savefig("k_means_iteration{}.png".format(i))
        wrong1, wrong2 = 0, 0
        for i in data[:200]:
            if (i[0], i[1]) not in self.classifications[0]:
                wrong1 += 1
            if (i[0], i[1]) not in self.classifications[1]:
                wrong2 += 1
        fig = plt.figure()
        plt.plot(iter_lst, log_lst)
        fig.savefig("cost_iter.png")
        print("Error percentage: " + str(min(wrong1/200, wrong2 / 200)))



km = K_Means()
km.fit(full_data)
# print('-----------------------------------------')

class EM:
    def __init__(self, k=2, tol = 0.00001, max_iter=50):
        self.imat = np.array([1, 0, 0, 1]).reshape(2,2)
        self.mu1 = np.array([0, 0])
        self.mu2 = np.array([1,1])
        self.k = k
        self.tol = tol
        self.max_iter = max_iter
        self.params = {'mu1': self.mu1, 'mu2': self.mu2, 'cov1' : self.imat, 'cov2' : self.imat, 'ratio' : [0.5, 0.5]}
        self.pi = {0:0, 1:0}

    def normal_density(self, d, mu, cov):
        return multivariate_normal.pdf(d, mu, cov)

    def log_likelihood(self, d, mu, cov, ratio):
        r = ratio
        for j in range(len(d)):
            r *= self.normal_density(d[j], mu[j], cov[j, j])
        return r

    def em_e_step(self, data):
        self.classifications = {0:[], 1:[]}
        xc = data[:, 0]
        yc = data[:, 1]
        label = {}
        for i in range(data.shape[0]):
            p1 = self.log_likelihood([xc[i], yc[i]], self.params['mu1'], self.params['cov1'], self.params['ratio'][0])
            p2 = self.log_likelihood([xc[i], yc[i]], self.params['mu2'], self.params['cov2'], self.params['ratio'][1])
            self.pi[0] += p1
            self.pi[1] += p2
            if p1 > p2:
                self.classifications[0].append((xc[i], yc[i]))
            else:
                self.classifications[1].append((xc[i], yc[i]))

    def em_f_step(self):
        points_in_cluster_1, points_in_cluster_2 = np.array(self.classifications[0]), np.array(self.classifications[1])
        cluster_1_ratio = self.pi[0] / 200
        cluster_2_ratio = self.pi[1] / 200
        print(cluster_1_ratio, cluster_2_ratio)
        self.params['mu1'] = np.array([points_in_cluster_1[:, 0].mean(), points_in_cluster_1[:, 1].mean()])
        self.params['mu2'] = np.array([points_in_cluster_2[:, 0].mean(), points_in_cluster_2[:, 1].mean()])
        self.params['cov1'] = np.array([[points_in_cluster_1[:, 0].std(), 0], [0, points_in_cluster_1[:, 1].std()]])
        self.params['cov2'] = np.array([[points_in_cluster_2[:, 0].std(), 0], [0, points_in_cluster_2[:, 1].std()]])
        self.params['ratio'] = np.array([cluster_1_ratio, cluster_2_ratio])

    def compute_shift(self, cur_params, old_params):
        result = 0
        for p in ['mu1', 'mu2']:
            for i in range(2):
                result += (cur_params[p][i] - old_params[p][i]) ** 2
        return result ** 0.5

    def fit(self, data):
        optimized = False
        iteration = 0
        while iteration < self.max_iter and not optimized:
            iteration += 1
            prev_params = dict(self.params)
            self.em_e_step(data)
            self.em_f_step()
            shift = self.compute_shift(self.params, prev_params)
            print('Iteration: {}, Shift; {}'.format(iteration, shift))
            if shift < self.tol:
                optimized = True
            fig = plt.figure()
            label = [0] * len(self.classifications[0]) + [1] * len(self.classifications[1])
            full_cluster = np.concatenate((self.classifications[0], self.classifications[1]))
            x_pos = full_cluster[:, 0]
            y_pos = full_cluster[:, 1]
            plt.scatter(x_pos, y_pos, 24, c=label)
            fig.savefig("iteration{}.png".format(iteration))
        wrong1, wrong2 = 0, 0
        for i in data[:200]:
            if (i[0], i[1]) not in self.classifications[0]:
                wrong1 += 1
            if (i[0], i[1]) not in self.classifications[1]:
                wrong2 += 1
        print("Error percentage: " + str(min(wrong1/200, wrong2 / 200)))


em = EM()
em.fit(full_data)

# kmeans = KMeans(n_clusters=2, random_state=0).fit(full_data)
# label = kmeans.labels_
# fig = plt.figure()
# x_pos = full_data[:, 0]
# y_pos = full_data[:, 1]
# plt.scatter(x_pos, y_pos, 24, c=label)
# fig.savefig('kmeans.png')
# plt.show
#
# fig1 = plt.figure()
# gmm = GMM(n_components=2).fit(full_data)
# labels = gmm.predict(full_data)
# plt.scatter(x_pos, y_pos, c=labels, s=40, cmap='viridis');
# plt.show()
# fig1.savefig('em.png')
#


