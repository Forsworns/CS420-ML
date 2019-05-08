from generate_data_gaussian import generate_3c_2d, Gaussian, generate_arbitry
import math
import numpy as np
import matplotlib.pyplot as plt
from configs import *


class RP_kmean():
	def __init__(self, data, kernel_num=3, eta=0.003, mus=np.zeros([3, 2]), Sigmas=np.zeros([3, 2, 2]), max_ite=MAX_ITE, delta=DELTA, bInit=False, verbose=True):
		# data is an numpy array of size (n x d)
		# mus is (k x d), Sigma is (k x d x d)
		self.N = data.shape[0]  # data amount
		self.d = data.shape[1]  # data dimension
		self.x = data
		self.eta = eta
		self.r = np.zeros([self.N, kernel_num])
		self.epsilon = np.zeros([self.N, kernel_num])
		self.Nk = np.zeros([kernel_num])
		self.pi = np.ones([kernel_num])/kernel_num  # 初始化一个随机的先验：随机属于哪个类
		self.max_ite = max_ite
		self.delta = delta
		self.verbose = verbose
		self.deserted = np.zeros([kernel_num],dtype=bool)
		self.init_gaussian(kernel_num, mus, Sigmas, bInit)

	def init_gaussian(self, kernel_num, mus, Sigmas, bInit):
		self.K = kernel_num
		self.real_mus = mus
		_, _, self.mus, self.Sigmas = generate_arbitry(
			kernel_num, self.N, self.d, bRandom=True)
		if bInit:
			if mus.shape[0] == kernel_num and mus.shape[1] == self.d:
				self.mus = mus
			else:
				if mus.shape[0] < kernel_num and mus.shape[1] == self.d:
					self.mus[:mus.shape[0], :] = mus
			if Sigmas.shape[0] == kernel_num and Sigmas.shape[1] == self.d and Sigmas.shape[2] == self.d:
				self.Sigmas = Sigmas
			else:
				if Sigmas.shape[0] < kernel_num and Sigmas.shape[1] == self.d and Sigmas.shape[2] == self.d:
					self.Sigmas[:Sigmas.shape[0], ...] = Sigmas
					for i in range(Sigmas.shape[0], kernel_num):
						self.Sigmas[i, ...] = np.diag(np.random.rand(self.d))
		if self.verbose:
			y = np.zeros([self.N])
			self.show_scatter(y, "orig")

	def main_loop(self):
		for ite in range(1, self.max_ite):
			if self.verbose:
				print("iteration {}".format(ite))
			self.ite = ite
			mus = self.mus
			Sigmas = self.Sigmas
			self.update_r_epsilon()
			self.update_mu()
			self.update_Sigma()
			if ite > 1 and (self.mus-mus).max() < self.delta or (self.Sigmas-Sigmas).max() < self.delta:
				print("converge in {} iterations".format(ite))
				y = np.zeros([self.N])  # 不能写成np.zeros([self.N,1])的形式
				for n in range(0, self.N):
					y[n]=self.r[n, :].argsort()[-1]
				return y
			if self.verbose:
				y = np.zeros([self.N])
				for n in range(0, self.N):
					y[n]=self.r[n, :].argsort()[-1]
				self.show_scatter(y, "iteration {}".format(ite))
		print("can't converge")

	def update_r_epsilon(self):
		self.r=np.zeros(self.r.shape)
		self.epsilon=np.zeros(self.epsilon.shape)
		for n in range(0, self.N):
			# 只需要求一下属于哪个的概率大
			klass_prob=np.array([self.pi[k]*Gaussian(self.x[n, :].T, self.mus[k, :].T,
													   np.squeeze(self.Sigmas[k, :]))for k in range(0, self.K)])
			klass_prob[np.isnan(klass_prob)] = 0
			klass_prob[np.isinf(klass_prob)] = 0
			max_k=klass_prob.argsort()[-1]
			second_k=klass_prob.argsort()[-2]
			self.r[n, max_k]=1
			self.epsilon[n, second_k]=-self.eta

	def update_mu(self):
		self.mus=np.zeros(self.mus.shape)
		# 先求均值
		for k in range(0, self.K):
			self.Nk[k]=self.r[:, k].sum()
			if not self.deserted[k]:
				self.pi[k]=self.Nk[k] / self.N
			if self.pi[k] < THRESHOLD:
				self.deserted[k] = True # 概率过低的一个点被丢弃掉
				self.pi[k] = 0
			for n in range(0, self.N):
				self.mus[k, :]=self.mus[k, :] + (self.r[n, k]*self.x[n, :])
			self.mus[k, :]=self.mus[k, :] / (self.Nk[k]+EPSILON)
		# 驱逐竞争对手
		for k in range(0, self.K):
			for j in range(0, self.K):
				for n in range(0, self.N):
					self.mus[j, :]=self.mus[j, :] + self.epsilon[n, k] * \
						(self.mus[k, :]-self.mus[j, :]) / self.ite
		if self.verbose:
			print("mus are")
			print(self.mus)
			print("")
			print(self.pi)

	def update_Sigma(self):
		self.Sigmas=np.zeros(self.Sigmas.shape)
		for k in range(0, self.K):
			if self.deserted[k]:
				continue
			temp_sum=np.zeros(self.Sigmas[k, :].shape)
			for n in range(0, self.N):
				temp_sum=temp_sum + \
					self.r[n, k]*np.outer(self.x[n, :]-self.mus[k, :],
										  self.x[n, :]-self.mus[k, :])
			self.Sigmas[k, :]=temp_sum / (self.Nk[k]+EPSILON)
		if self.verbose:
			print("Sigmas are")
			print(self.Sigmas)
			print("")

	def show_scatter(self, y, title):
		for k, color in enumerate(COLORS[0:self.real_mus.shape[0]]):
			data=self.real_mus[k]
			plt.scatter(data[0], data[1], c=color, marker='o')
		for k, color in enumerate(COLORS[0:self.K]):
			data=self.x[y == k]
			plt.scatter(data[:, 0], data[:, 1], s=0.8,
						c=color, label="{}th-cluster".format(k))
			if k < self.mus.shape[0]:
				mu=self.mus[k]
				plt.scatter(mu[0], mu[1], c=color, marker='x')
		plt.axis()
		plt.legend()  # 防止label不显示
		plt.title(title)
		plt.xlabel("x1")
		plt.ylabel("x2")
		plt.show()


if __name__ == "__main__":
	data, klass, mus, covs=generate_3c_2d()
	data, klass, mus, covs=generate_arbitry(
		cluster=3, num=POINTS_NUM*3, bRandom=True)
	print(data.shape)
	RP_test=RP_kmean(data, kernel_num=4, eta=0.003,
					   mus=mus, Sigmas=covs, max_ite=MAX_ITE, bInit=True, verbose=True)
	y=RP_test.main_loop()
	RP_test.show_scatter(y, "RL-kmean")
