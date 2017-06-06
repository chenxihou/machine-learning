#coding:utf8
#reference:http://www.cnblogs.com/d-roger/articles/5719979.html

import numpy as np

class HMM:
    def __init__(self, pi, A, B, obs_all):
        self.pi = pi
        self.trans = A
        self.omit = B
        self.obs_all = obs_all
        self.status_num = A.shape[0]

    def forward(self, obs_seq):
        T = len(obs_seq)
        alpha = np.zeros((T, self.status_num))
        alpha[0,:] = pi * B[:,self.obs_all.index(obs_seq[0])].T
        # for i in xrange(self.status_num):
        #     p[0][i] = self.pi[i] * self.omit[i][self.observation.index(observed[0])]
        for i in xrange(1,T):
            for j in xrange(self.status_num):
                total = 0
                for k in xrange(self.status_num):
                    total += alpha[i-1][k] * self.trans[k][j] * self.omit[j][self.obs_all.index(obs_seq[i])]
                alpha[i][j] = total
        print alpha
        return sum(alpha[-1]),alpha

    def backword(self, observed):
        observed_num = len(observed)
        beta = np.zeros((observed_num, self.status_num), dtype= float)
        beta[-1:,:] = 1
        if observed_num > 1:
            for i in reversed(range(observed_num - 1)):
                for j in range(self.status_num):
                    total = 0
                    for k in range(self.status_num):
                        total += self.trans[j][k] * self.omit[k][self.obs_all.index(observed[i+1])] * beta[i+1][k]
                    beta[i][j] = total
        result = 0
        print beta
        for i in range(self.status_num):
            result += self.pi[i] * self.omit[i][self.obs_all.index(observed[0])] * beta[0][i]
        return result

    def cal_gama(self, alpha, beta, obs_seq):
        T = len(obs_seq)
        gama = np.zeros((T, self.status_num), dtype=float)
        for i in range(T):
            for j in range(self.status_num):
                gama[i][j] = alpha[i][j] * beta[i][j] / sum([alpha[i][j] * beta[i][j] for i in range(self.status_num)])
        epsilon = np.zeros((T-1,self.status_num, self.status_num), dtype=float)
        for i in range(T-1):
            for j in range(self.status_num):
                for k in range(self.status_num):
                    epsilon[i][j][k] = alpha[i][j] * self.trans[j][k] * self.omit[k][self.obs_all.index(obs_seq[i])] * beta[i+1][k]/\
                        sum([alpha[i][m] * self.trans[m][n] * self.omit[n][self.obs_all.index(obs_seq[i])] * beta[i+1][n] for m in range(self.status_num) for n in range(self.status_num)])
        return gama


    def baum_welch_train(self):
        pass

    def viterbi(self, observed):
        observed_num = len(observed)
        p = np.zeros((observed_num, self.status_num))
        path = np.zeros((observed_num, self.status_num), dtype=int)
        for i in xrange(self.status_num):
            p[0][i] = self.pi[i] * self.omit[i][self.observation.index(observed[0])]
        for i in xrange(1, observed_num):
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
obs_all = ['红', '白']

hmm = HMM(pi, A, B, obs_all)
obs_seq = ['白', '白', '红']
# observed = ['白', '白', '红']
# observed = ['红']
pro = hmm.forward(obs_seq)
pro_b = hmm.backword(obs_seq)
# path = hmm.viterbi(observed)

print pro
print pro_b

