from sklearn.mixture import (GaussianMixture, BayesianGaussianMixture)
from sklearn.metrics.pairwise import pairwise_distances_argmin
from configs import *
from generate_data_gaussian import *
import math
import matplotlib.pyplot as plt


class MS_Experiment():
	def __init__(self, K=3, N=300, d=2, max_ite=MAX_ITE, delta=0.001, verbose=True, bInit=False):
		self.x, self.real_y, self.real_mus, self.real_Sigmas = generate_arbitry(
			K, N, d)
		self.N = self.x.shape[0]
		self.d = self.x.shape[1]
		self.max_ite = max_ite
		self.delta = delta
		self.verbose = verbose
		self.bInit = bInit

	def init_k_cluster(self,K):
		self.K = K
		self.Nk = np.zeros([K])
		self.pi = np.ones([K])/K
		self.posterior = np.zeros([self.N, K])
		_, _, self.mus, self.Sigmas = generate_arbitry(
			K, self.N, self.d, bRandom=True)
		if self.bInit:
			if self.real_mus.shape[0] == K and self.real_mus.shape[1] == self.d:
					self.mus = self.real_mus
			else:
				if self.real_mus.shape[0] < K and self.real_mus.shape[1] == self.d:
					self.mus[:self.real_mus.shape[0], :] = self.real_mus
			if self.real_Sigmas.shape[0] == K and self.real_Sigmas.shape[1] == self.d and self.real_Sigmas.shape[2] == self.d:
				self.Sigmas = self.real_Sigmas
			else:
				if self.real_Sigmas.shape[0] < K and self.real_Sigmas.shape[1] == self.d and self.real_Sigmas.shape[2] == self.d:
					self.Sigmas[:self.real_Sigmas.shape[0], ...] = self.real_Sigmas
					for i in range(self.real_Sigmas.shape[0], K):
						self.Sigmas[i, ...] = np.diag(np.random.rand(self.d))

	def free_para(self):
		# Sigmas d**2*K, mu d*k free parameters
		return self.K*(self.d*self.d+self.d)

	def AIC(self):
		log_likelihood = self.log_likelihood()
		return 0.5*self.free_para()-log_likelihood

	def BIC(self):
		log_likelihood = self.log_likelihood()
		return 0.5*self.free_para()*math.log(self.N, math.e)-log_likelihood

	def correct_order(self, y, clf):
		order = pairwise_distances_argmin(
			self.real_mus, clf.means_, axis=1, metric="euclidean")
		idx = np.empty([self.K, self.N], dtype=bool)
		ordered_y = np.ones([self.N])*self.K
		for i in range(0, min(order.shape[0], self.K)):
			idx[i] = y == order[i]
		for i in range(0, min(order.shape[0], self.K)):
			ordered_y[idx[i]] = i
		return ordered_y

	def estimate(self):
		# E of EM
		for n in range(0, self.N):
			likelihood_sum = 0
			likelihood_sum = sum(
				[self.pi[k]*Gaussian(self.x[n, :], self.mus[k, :], self.Sigmas[k, ...]) for k in range(0, self.K)])
			for k in range(0, self.K):
				# p(Z|X) = p(X,Z)/p(X) = P(Z)*P(X|Z)/sum_{Z}(P(Z)*P(X|Z))
				self.posterior[n, k] = self.pi[k]*Gaussian(
					self.x[n, :], self.mus[k, :], self.Sigmas[k, ...])/likelihood_sum

	def maximize(self):
		# M of EM
		for k in range(0, self.K):
			self.Nk[k] = sum([self.posterior[n, k] for n in range(0, self.N)])
			self.mus[k, :] = 1/self.Nk[k] * np.array([self.posterior[n, k]*self.x[n, :]
													  for n in range(0, self.N)]).sum(axis=0)

			self.Sigmas[k, ...] = 1/self.Nk[k]*np.array([self.posterior[n, k]*np.outer(self.x[n, :]-self.mus[k, :],
																					   self.x[n, :]-self.mus[k, :]) for n in range(0, self.N)]).sum(axis=0)
			self.pi[k] = self.Nk[k]/self.N

	def EM(self, K):
		self.init_k_cluster(K)
		for ite in range(1, self.max_ite):
			if self.verbose:
				print("iteration {}".format(ite))
			mus = self.mus
			Sigmas = self.Sigmas
			self.estimate()
			self.maximize()
			if (self.mus-mus).max() < self.delta or (self.Sigmas-Sigmas).max() < self.delta:
				print("converge in {} iterations".format(ite))
				y = self.real_y
				for n in range(0,self.N):
					y[n] = np.array([self.pi[k]*Gaussian(
						self.x[n, :], self.mus[k, :], self.Sigmas[k, ...]) for k in range(0,self.K)]).argsort()[-1]
				return y
		print("can't converge")

	def log_likelihood(self):
		log_p = 0
		for n in range(0, self.N):
			p = 0
			for k in range(0, self.K):
				p += self.pi[k]*Gaussian(self.x[n, :],
										 self.mus[k, :], self.Sigmas[k, ...])
			log_p += math.log(p, math.e)
		return log_p

	def ms_EM(self):
		# 使用这里实现的EM和AIC,BIC
		aic = []
		bic = []
		for k in range(MIN_K, MAX_K):
			y = self.EM(k)
			# self.show_scatter(y, "EM with {} clusters".format(k))
			aic.append(self.AIC())
			bic.append(self.BIC())
			print(aic)
			print(bic)
			save_data(CRI_FILES[2], aic)
			save_data(CRI_FILES[3], bic)

	def ms_compare(self):
		# 直接使用sklearn中的GMM
		aic = []
		bic = []
		for k in range(MIN_K, MAX_K):
			self.K = k
			clf = GaussianMixture(
				n_components=k, covariance_type="full", max_iter=200, random_state=0)
			y = self.correct_order(clf.fit_predict(self.x, self.real_y), clf)
			self.mus = clf.means_ # fit过后才有attributes = =
			print(y)
			# self.show_scatter(y, "GMM with {} clusters".format(k))
			accuracy = np.mean(self.real_y.ravel() == y.ravel())
			print(accuracy)
			aic.append(clf.aic(self.x))
			bic.append(clf.bic(self.x))
			print(aic)
			print(bic)
		save_data(CRI_FILES[0], aic)
		save_data(CRI_FILES[1], bic)

	def ms_VB(self):
		# 直接使用 sklearn 中的 VBGMM
		self.K = MAX_K
		clf = BayesianGaussianMixture(
			n_components=MAX_K, covariance_type="full", max_iter=200, random_state=0)
		y = self.correct_order(clf.fit_predict(self.x), clf)
		self.mus = clf.means_
		print(y)
		self.show_scatter(y, "VBGMM")
		accuracy = np.mean(self.real_y.ravel() == y.ravel())
		print(accuracy)

	def show_scatter(self, y, title):
		for k, color in enumerate(COLORS[0:self.real_mus.shape[0]]):
			data = self.real_mus[k]
			plt.scatter(data[0], data[1], c=color, marker='o',s=2*POINTSIZE)
		for k, color in enumerate(COLORS[0:self.K]):
			data = self.x[y == k]
			plt.scatter(data[:, 0], data[:, 1], s=POINTSIZE,
						c=color, label="{}th-cluster".format(k))
			if k < self.mus.shape[0]:
				mu = self.mus[k]
				plt.scatter(mu[0], mu[1], c=color, marker='x',s=2*POINTSIZE)
		plt.axis(fontsize=FONTSIZE)
		plt.legend(fontsize=FONTSIZE)  # 防止label不显示
		plt.title(title,fontsize=2*FONTSIZE)
		plt.xlabel("x1",fontsize=2*FONTSIZE)
		plt.ylabel("x2",fontsize=2*FONTSIZE)
		plt.show()

	def show_line(self):
		data = list(map(lambda file: load_data(file), CRI_FILES))
		plt.figure(figsize=(6, 6))
		plt.subplots_adjust(bottom=.10, top=0.95, hspace=.25, wspace=.15,
							left=.05, right=.99)
		for i in range(0, 4):
			plt.subplot(2, 2, i+1)
			plt.plot([item for item in range(MIN_K, MAX_K)], data[i])
			plt.title(CRI_FILES[i].split(".")[0],fontsize=FONTSIZE)
			plt.xlabel("cluster k",fontsize=FONTSIZE)
			plt.ylabel("AIC/BIC",fontsize=FONTSIZE)
		plt.show()


if __name__ == "__main__":
	MS_E = MS_Experiment(bInit=True) # 是否以真实中心作为初始化点
	MS_E.ms_VB()
	MS_E.ms_compare()
	MS_E.ms_EM()
	MS_E.show_line()
