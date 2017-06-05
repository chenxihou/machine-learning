#coding:utf8

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
        p = np.zeros((T, self.status_num))
        p[0,:] = pi * B[:,self.obs_all.index(obs_seq[0])].T
        # for i in xrange(self.status_num):
        #     p[0][i] = self.pi[i] * self.omit[i][self.observation.index(observed[0])]
        for i in xrange(1,T):
            for j in xrange(self.status_num):
                total = 0
                for k in xrange(self.status_num):
                    total += p[i-1][k] * self.trans[k][j] * self.omit[j][self.obs_all.index(obs_seq[i])]
                p[i][j] = total
        print p
        return sum(p[-1])

    def backword(self, observed):
        observed_num = len(observed)
        p = np.zeros((observed_num, self.status_num), dtype= float)
        p[-1:,:] = 1
        if observed_num > 1:
            for i in reversed(range(observed_num - 1)):
                for j in range(self.status_num):
                    total = 0
                    for k in range(self.status_num):
                        total += self.trans[j][k] * self.omit[k][self.obs_all.index(observed[i+1])] * p[i+1][k]
                    p[i][j] = total
        result = 0
        print p
        for i in range(self.status_num):
            result += self.pi[i] * self.omit[i][self.obs_all.index(observed[0])] * p[0][i]
        return result



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

