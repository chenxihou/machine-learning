#coding:utf8
#reference:http://www.cnblogs.com/d-roger/articles/5719979.html

import numpy as np

class HMM:
    def __init__(self, pi, A, B, O):
        self.pi = pi
        self.A = A
        self.B = B
        self.O = O
        self.T = len(O)
        self.N = A.shape[0]
        self.M = B.shape[1]

    def forward(self):
        alpha = np.zeros((self.T, self.N))
        alpha[0,:] = pi * B[:,self.O[0]].T
        for t in xrange(1,self.T):
            for i in range(self.N):
                alpha[t][i] = np.dot(alpha[t-1,:], self.A[:,i]) * self.B[i][self.O[t]]
        pro = sum(alpha[-1])
        return pro, alpha

    def backword(self):
        beta = np.zeros((self.T, self.N))
        beta[-1:,:] = 1
        if self.T > 1:
            for t in reversed(range(self.T - 1)):
                for i in range(self.N):
                    for j in range(self.N):
                        beta[t][i] = sum(self.A[i,:] * self.B[:,self.O[t+1]].T * beta[t+1,:])
        pro = sum(beta[0,:] * self.pi.T * self.B[:,self.O[0]].T)
        return pro, beta

    def cal_gama_epsilon(self):
        gama = np.zeros((self.T, self.N), dtype=float)
        alpha, beta = self.forward()[1], self.backword()[1]
        for t in range(self.T):
            for i in range(self.N):
                gama[t][i] = alpha[t][i] * beta[t][i] / sum([alpha[t][j] * beta[t][j] for j in range(self.N)])
        epsilon = np.zeros((self.T - 1, self.N, self.N), dtype=float)
        for t in range(self.T - 1):
            for i in range(self.N):
                for j in range(self.N):
                    epsilon[t][i][j] = alpha[t][i] * self.A[i][j] * self.B[j][self.O[t+1]] * beta[t+1][j]/\
                        sum([alpha[t][m] * self.A[m][n] * self.B[n][self.O[t+1]] * beta[t+1][n] for m in range(self.N) for n in range(self.N)])
        return gama, epsilon


    def baum_welch_train(self):
        pass

    def viterbi(self, observed):
        delta = np.zeros(self.T, self.N)
        psai = np.zeros(self.T, self.N)
        # path = np.zeros((observed_num, self.status_num), dtype=int)
        delta[0,:] = self.pi * self.B[:,self.O[0]].T
        psai[0,:] = 0
        for t in range(1, self.T):
            for i in range(self.N):
                delta[t][i] = max(delta[t-1:] * self.A[:,i].T)

            for j in xrange(self.status_num):
                target = 0
                for k in xrange(self.status_num):
                    tmp = p[i-1][k] * self.trans[k][j] * self.omit[j][self.observation.index(observed[i])]
                    if tmp > target:
                        target = tmp
                        p[i][j] = tmp
                        path[i][j] = k
        walk = []
        walk.append(np.argmax(p[-1]))
        for i in reversed(range(1, observed_num)):
            walk.insert(0,path[i][walk[0]])
        return walk

pi = np.array([0.2, 0.4, 0.4])
A = np.array([[0.5, 0.2, 0.3],
              [0.3, 0.5, 0.2],
              [0.2, 0.3, 0.5]])
B = np.array([[0.5, 0.5],
              [0.4, 0.6],
              [0.7, 0.3]])
O = [0,1,0]
# obs_all = ['红', '白']

hmm = HMM(pi, A, B, O)
obs_seq = ['红', '白', '红']
alpha, r = hmm.forward()
print alpha, r
v = hmm.backword()
print v,1111

gama, epsilon = hmm.cal_gama_epsilon()
print gama
print epsilon

