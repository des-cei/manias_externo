from sklearn.metrics import accuracy_score
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import StratifiedKFold

from multiprocessing import Process, Manager

import numpy as np
import matplotlib.pyplot as plt


# ===========================
#  PROCESAMIENTO EN PARALELO
# ===========================
class Parallel:
	# Los procesos son modelados como y(x)
	# Los resultados --y-- se guardan como atributo de la clase
	def __init__(self, func2eval, n_processes=12):
		self.func2eval  = func2eval
		self.n_processes = n_processes
		self.manager    = Manager()
		self.y          = self.manager.dict()

	def split_lista(self, lst, n):
		k, m = divmod(len(lst), n)
		return [lst[i * k + min(i, m):(i + 1) * k + min(i + 1, m)] for i in range(n)]


	def worker(self,chunk,chunk_idx):
			resultados_chunk = []
			for x_i in chunk:
				resultados_chunk.append( self.func2eval(x_i) )
			self.y[chunk_idx] = resultados_chunk

	def evaluateY(self, x):
		x = self.split_lista(x, self.n_processes)
		self.y = self.manager.dict()
		
		all_process = []
		for chunk_idx,chunk in enumerate(x):
			t = Process(target=self.worker,args=[chunk,chunk_idx])
			t.start()
			all_process.append(t)

		for t in all_process:
			t.join()
		
		y = dict(self.y)
		all_y = []
		for key in sorted(y.keys()):
			all_y += y[key]
		return all_y

# ===========================
#       ANALISIS
# ===========================
def contarEnergia(instance, n_dim):
	e = [0 for _ in range(n_dim)]
	for time,idx in instance:
		e[idx] += 1
	return e

# ===========================
#       PLOTING
# ===========================
def rasterplot(instance):
	time,channel = zip(*instance)
	plt.scatter(time,channel, s=1, color='black')

# ===========================
#      EVALUATION
# ===========================

def fastEval(d_train, d_test, t_train, t_test, model='linear'):
	scaler = StandardScaler().fit(d_train)
	d_train = scaler.transform(d_train)
	d_test  = scaler.transform(d_test)

	if model=='linear'			: model  = LogisticRegression(solver='saga',max_iter=200)
	elif model=='random_forest' : model = RandomForestClassifier()
	model.fit(d_train,t_train)
	pred = model.predict(d_train)
	acc_train = accuracy_score( t_train, pred )
	
	pred = model.predict(d_test)
	acc_test = accuracy_score( t_test, pred )


	return acc_test,acc_train



def fastEvalKfold(data, targets, k=5, model='linear'):
	
	all_acc_train = []
	all_acc_test  = []
	kf = StratifiedKFold(n_splits=k,shuffle=True)
	for idx_train, idx_test in kf.split(data,targets):
		# ESCALAR DATOS
		scaler = StandardScaler()
		scaler.fit( data[idx_train] )
		d_train = scaler.transform( data[idx_train] )
		d_test  = scaler.transform( data[idx_test] )

		# FIT MODEL
		if model=='linear'			: model = LogisticRegression(solver='saga',max_iter=150)
		elif model=='random_forest' : model = RandomForestClassifier()
		model.fit( d_train, targets[idx_train] )
	
		# PREDICT
		pred = model.predict(d_train)
		all_acc_train.append( accuracy_score( targets[idx_train], pred ) )
		pred = model.predict(d_test)
		all_acc_test.append( accuracy_score( targets[idx_test], pred ) )

	acc		  = np.mean(all_acc_test).round(3)
	acc_train = np.mean(all_acc_train).round(3)
	std       = np.std(all_acc_test).round(3)
	std_train = np.std(all_acc_train).round(3)

	return acc, std, acc_train, std_train