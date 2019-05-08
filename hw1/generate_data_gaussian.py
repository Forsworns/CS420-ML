import numpy as np
# import json
import matplotlib.pyplot as plt
import os
import math
import json
from configs import *


def Gaussian(x, mu, Sigma):
    # 给定分布后求取似然
    mu = mu.reshape(mu.shape[0], 1)  # 从一维数组转化成n*1的列向量
    x = x.reshape(mu.shape[0], 1)
    d = x.shape[0]/2
    if d >= 1:
        return (1/(2*math.pi)**d)*(1/(np.linalg.det(Sigma)+EPSILON)**0.5)*math.exp(-0.5*np.dot(np.dot((x-mu).T, np.linalg.inv(Sigma+np.eye(Sigma.shape[0])*EPSILON)), x-mu))
    else:
        return (1/(2*math.pi)**d)*(1/Sigma**0.5)*math.exp(-0.5*(x-mu)**2/Sigma)


def generate_arbitry(cluster=3, num=POINTS_NUM*3, dimension=2, bRandom=False):
    filename = "data/arbitry_k{}_n{}_d{}.json".format(cluster, num, dimension)
    if not bRandom and os.path.exists(filename):
        data, klass, mus, Sigmas = load_data(filename)
        print("mus")
        print(mus)
        print("Sigmas")
        print(Sigmas)
        return list(map(lambda item: np.array(item), [data, klass, mus, Sigmas]))
    mus = np.zeros([cluster, dimension])
    Sigmas = np.zeros([cluster, dimension, dimension])

    epsilon = np.random.rand(dimension, dimension)*BOUNDARY
    mus[0, :] = np.random.random_integers(BOUNDARY, size=[dimension])
    Sigmas[0, ...] = np.diag(np.random.rand(dimension))

    data = np.random.multivariate_normal(mus[0, :], Sigmas[0, ...], POINTS_NUM)
    klass = np.zeros([POINTS_NUM, 1])
    for c in range(1, cluster):
        x1 = np.random.random_integers(BOUNDARY)
        x2 = np.random.random_integers(BOUNDARY)
        epsilon = np.random.rand(dimension, dimension)*BOUNDARY
        mus[c, :] = np.random.random_integers(BOUNDARY, size=[dimension])
        Sigmas[c, ...] = np.diag(np.random.rand(dimension))
        data = np.append(data,
                         np.random.multivariate_normal(mus[c, :], Sigmas[c, ...], POINTS_NUM), axis=0)
        klass = np.append(klass, c*np.ones([POINTS_NUM, 1]))
    if not bRandom:
        save_data(filename, list(map(lambda item: item.tolist(),
                                     [np.around(data, 4), klass, mus, Sigmas])))
    print("mus")
    print(mus)
    print("Sigmas")
    print(Sigmas)
    return np.around(data, 4), klass, mus, Sigmas


def generate_3c_2d():
    # 生成三组数据，均为2维
    mu1 = np.array([0, 0])
    sigma1 = np.array([[1, 0], [0, 10]])

    mu2 = np.array([10, 10])
    sigma2 = np.array([[10, 0], [0, 1]])

    mu3 = np.array([10, 0])
    sigma3 = np.array([[3, 0], [0, 4]])

    klass = np.zeros([POINTS_NUM, 1])
    klass = np.append(klass, 1*np.ones([POINTS_NUM, 1]))
    klass = np.append(klass, 2*np.ones([POINTS_NUM, 1]))

    if os.path.exists('data/3clusters.txt'):
        data = load_data('data/3clusters.txt')
        return data, klass, np.array([mu1, mu2, mu3]), np.array([sigma1, sigma2, sigma3])

    data = np.random.multivariate_normal(mu1, sigma1, POINTS_NUM)

    data = np.append(data,
                     np.random.multivariate_normal(mu2, sigma2, POINTS_NUM), axis=0)

    data = np.append(data,
                     np.random.multivariate_normal(mu3, sigma3, POINTS_NUM), axis=0)
    save_data('data/3clusters.txt', data)
    # 保留四位小数
    return np.around(data, 4), klass, np.array([mu1, mu2, mu3]), np.array([sigma1, sigma2, sigma3])


def save_data(filename, data):
    if "txt" in filename:
        with open(filename, "w") as f:
            for i in range(data.shape[0]):
                f.write("{},{}\n".format(data[i, 0], data[i, 1]))
    elif "json" in filename:
        with open(filename, "w") as f:
            f.write(json.dumps(data))
    elif "npy" in filename:
        data = np.array(data)
        np.save(filename, data)


def load_data(filename):
    data = []
    if "txt" in filename:
        with open(filename, "r") as f:
            for line in f.readlines():
                data.append([float(item) for item in line.split(',')])
        return np.array(data)
    elif "json" in filename:
        with open(filename, "r") as f:
            data = json.loads(f.read())
        return data
    elif "npy" in filename:
        data = np.load(filename)
        return data


def show_scatter():
    data, klass, mus, Sigmas = generate_arbitry(num=300)
    plt.scatter(data[:, 0], data[:, 1])
    plt.scatter(mus[:, 0], mus[:, 1], c=COLORS[0],marker='x')
    plt.axis()
    plt.legend()  # 防止label不显示
    plt.title("clusters")
    plt.xlabel("x1")
    plt.ylabel("x2")
    plt.show()


if __name__ == "__main__":
    if not os.path.exists('data/3clusters.txt'):
        res = generate_3c_2d()
        save_data('data/3clusters.txt', res[0])
    data = load_data('data/3clusters.txt')
    print(data.shape)
    data2 = generate_arbitry(num=300)
    show_scatter()
    print(data2[0].shape)
    print(Gaussian(np.array([0]), np.array([0]), np.array([1])))
