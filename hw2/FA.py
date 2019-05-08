from sklearn.decomposition import FactorAnalysis
from sklearn import datasets as ds
import matplotlib.pyplot as plt
import mpl_toolkits.mplot3d.axes3d as p3
import numpy as np
import math
from configs import *
from generate_data_gaussian import Gaussian,save_data,load_data


class FA_Test():
	def __init__(self, N=100, n=10, m=3, M=2, sigma=0.1, mu=None, A=None,verbose=True):
		if mu is None:
			self.mu = np.zeros([n, 1])
		else:
			self.mu = mu
		if A is None:
			self.A = np.random.rand(n, m)
		else:
			self.A = A
		self.N = N
		self.n = n
		self.real_m = m
		self.M = M
		self.sigma = sigma**0.5
		self.verbose = verbose
		self.generate_data()

	def generate_data(self):
		self.x = np.zeros([self.N, self.n])
		for i in range(0, self.N):
			self.y = np.random.randn(self.real_m, 1)
			self.e = np.random.randn(self.n, 1)*self.sigma
			self.x[i, :] = (self.A@self.y+self.mu+self.e).squeeze()

	def main_loop(self):
		self.aic_score = np.zeros(2*self.M+1)
		self.bic_score = np.zeros(2*self.M+1)
		for i in range(self.real_m-self.M, self.real_m+self.M+1):
			self.m = i
			fa_model = FactorAnalysis(n_components=self.m)
			fa_model.fit(self.x)
			self.log_likelihood = fa_model.score(self.x)*self.N
			self.aic_score[i-self.real_m+self.M] = self.AIC()
			self.bic_score[i-self.real_m+self.M] = self.BIC()
		if self.verbose:
			self.show_line()

	def free_para(self):
		# A mn, sigma 1 free parameters
		return self.m*self.n + 1

	def AIC(self):
		return self.log_likelihood - self.free_para()

	def BIC(self):
		return self.log_likelihood - 0.5*self.free_para()*math.log(self.N, math.e)

	def show_line(self):
		plt.figure(figsize=(6, 6))
		plt.subplots_adjust(bottom=.10, top=0.95, hspace=.25, wspace=.15,
							left=.1, right=.99)
		plt.subplot(2, 1, 1)
		plt.plot([i+1 for i in range(0, self.M)], self.aic_score)
		plt.xlabel("m components", fontsize=FONTSIZE)
		plt.ylabel("AIC", fontsize=FONTSIZE)
		plt.title("AIC/BIC-sklearn", fontsize=FONTSIZE)
		plt.subplot(2, 1, 2)
		plt.plot([i+1 for i in range(0, self.M)], self.aic_score)
		plt.xlabel("m components", fontsize=FONTSIZE)
		plt.ylabel("BIC", fontsize=FONTSIZE)
		plt.show()

	def Estep(self):
		self._W = self._A.T@np.linalg.inv(
			self._A@self._A.T+self._sigma_square*np.eye(self.n))
		self._y = self._W@self.x.T  # m*N

	def Mstep(self):
		A1 = np.zeros([self.n, self.m])
		A2 = np.zeros([self.m, self.m])
		S = np.zeros([self.n, self.n])
		for i in range(self.N):
			A1 += (self.x[i, :].reshape(-1, 1))@(self._y[:, i].reshape(1, -1))
			A2 += (np.eye(self.m)-self._W@self._A+self._W@
				   (self.x[i, :].reshape(-1, 1))@(self.x[i, :].reshape(1, -1))@self._W.T)
		self._A = A1@A2
		for i in range(self.N):
			S += ((self.x[i, :].reshape(-1, 1))@(self.x[i, :].reshape(1, -1)) -
				  self._A@(self._y[:, i].reshape(-1, 1))@(self.x[i, :].reshape(1, -1)))
		self._sigma_square = (1/(self.N*self.m)*np.trace(S))

	def log_like(self):
		log_p = 0
		for n in range(0, self.N):
			p = Gaussian(self.x[n, :], self._A@(self._y[:, n].reshape(-1, 1))+self.mu,
						 self._sigma_square*np.eye(self.n))
			log_p += math.log(p, math.e)
		return log_p

	def EM(self):
		for ite in range(MAX_ITE):
			sigma = self._sigma_square
			A = self._A.copy()
			self.Estep()
			self.Mstep()
			if abs(self._sigma_square-sigma) < DELTA or abs(self._A-A).max() < DELTA:
				print("converge in {} iterations".format(ite))
				return True
			print("can't converge")
		return False

	def EM_loop(self):
		self.aic_score = np.zeros(2*self.M+1)
		self.bic_score = np.zeros(2*self.M+1)
		for i in range(self.real_m-self.M, self.real_m+self.M+1):
			self.m = i
			self._A = np.random.rand(self.n, self.m)
			self._sigma_square = np.random.rand()
			converged = self.EM()
			if converged:
				self.log_likelihood = self.log_like()
			else:
				self.log_likelihood = -INFINITE
			self.aic_score[i-self.real_m+self.M] = self.AIC()
			self.bic_score[i-self.real_m+self.M] = self.BIC()
		self.show_line()


if __name__ == "__main__":
	params = []
	for n in [10,100]:
		for m in [3,7,10]:
			for sigma in [0.1,1,10]:
				params.append({'n':n,'m':m,'sigma':sigma})
	aic_accuracy = []
	bic_accuracy = []
	for param in params:
		test = FA_Test(verbose=False,**param)
		aic_choosen = []
		bic_choosen = []
		for _ in range(MAX_ITE):
			test.main_loop() # directly use sklearn
			# test.EM_loop()  # private implementation
			print(param,test.aic_score)
			print(param,test.bic_score)
			aic_choosen.append(np.argmax(test.aic_score))
			bic_choosen.append(np.argmax(test.bic_score))
		aic_accuracy.append({**param, 'acc':np.sum(np.array(aic_choosen)==2)/MAX_ITE}) # self.M, in fact
		bic_accuracy.append({**param, 'acc':np.sum(np.array(bic_choosen)==2)/MAX_ITE})
	
	save_data(CRI_FILES[0],aic_accuracy)
	save_data(CRI_FILES[1],bic_accuracy)

