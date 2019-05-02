class EM:
    def __init__(self):
        self.imat = np.array([1, 0, 0, 1]).reshape(2,2)
        self.mu1 = np.array([0, 0])
        self.mu2 = np.array([1,1])

    def normal_density(self, x):
        s = multivariate_normal.pdf(x, self.mu1, self.imat)
        q = multivariate_normal.pdf(x, self.mu2, self.imat)
        return s + q

    def log_likelihood(self, data):
        s = 0
        for i in range(data.shape[0]):
            s += self.normal_density(data[i])
        return s
    def em_e_step(self):
        pass

# em = EM()
# print(full_data[1])
# print(em.log_likelihood(full_data))

