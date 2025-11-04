from sklearn.metrics import accuracy_score
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier

from multiprocessing import Process, Manager
import numpy as np


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


def contarEnergia(instance, n_dim):
	e = [0 for _ in range(n_dim)]
	for time,idx in instance:
		e[idx] += 1
	return e



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