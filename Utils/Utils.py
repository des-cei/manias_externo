from sklearn.metrics import accuracy_score
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier

import warnings
from sklearn.exceptions import ConvergenceWarning
warnings.filterwarnings("ignore", category=ConvergenceWarning)


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