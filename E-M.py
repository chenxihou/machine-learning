# coding:utf-8

from __future__ import division
import numpy as np
import matplotlib.pyplot as plt
import copy


def init_data(sigma, Eu1, Eu2, k, N):
    global Eu, Expections, X
    Eu = np.random.random(2)
    Expections = np.zeros((N, k))
    X = np.zeros(N)
    for i in range(N):
        if np.random.random() > 0.5:
            X[i] = np.random.normal() * sigma + Eu1
        else:
            X[i] = np.random.normal() * sigma + Eu2
    plt.hist(X, 50, normed=1,facecolor='g',alpha=0.75)


def E_step(k, sigma, N):
    print 100 * '@'
    global Eu, Expections, X
    for i in range(N):
        total = 0
        for j in range(k):
            total += np.exp((-1 / (2 * (sigma ** 2))) * ((X[i] - Eu[j]) ** 2))
        for j in range(k):
            tmp = np.exp((-1 / (2 * (sigma ** 2))) * ((X[i] - Eu[j]) ** 2))
            Expections[i][j] = tmp / total


def M_step(k, N):
    for j in range(k):
        total = 0
        tmp = 0
        for i in range(N):
            tmp += Expections[i][j]
            total += Expections[i][j] * X[i]
        Eu[j] = total / tmp


def run(sigma, Eu1, Eu2, k, N, iter_num, Epsilon):
    init_data(sigma, Eu1, Eu2, k, N)
    for i in range(iter_num):
        old_Eu = copy.deepcopy(Eu)
        E_step(k, sigma, N)
        M_step(k, N)
        print i, old_Eu, Eu
        if sum(abs(Eu - old_Eu)) < Epsilon:
            break


if __name__ == '__main__':
    run(6, 20, 40, 2, 1000, 100, 0.0001)
    plt.grid(linestyle = ':')
    plt.show()