import numpy as np
from sklearn.utils import shuffle
from sklearn.model_selection import train_test_split

import pickle
from tqdm import tqdm

import os
import librosa

class FSDD:
	def __init__(self, time_bin=80, max_spikes=20, n_dim=26, hop_lenght=256):
		self.n_dim 	    = n_dim
		self.time_bin   = time_bin
		self.max_spikes = max_spikes
		self.hop_lenght = hop_lenght

		self.file_route = './Data/fsdd_procesado.pickle'
		self.folds      = []
	
	def encodeAudio(self,y,sr):
		bins = librosa.feature.mfcc(y=y, sr=sr,n_mfcc=self.n_dim, hop_length=self.hop_lenght, n_fft=2*self.hop_lenght)
		return bins
	
	def normalizeData(self,data_train, data_test):				# data debe tener forma (samples, fetaures, time)
		mins = data_train.min(axis=(0,2), keepdims=True)
		maxs = data_train.max(axis=(0,2), keepdims=True)
		data_train = (data_train - mins) / (maxs - mins + 1e-8)
		data_test  = (data_test  - mins) / (maxs - mins + 1e-8)

		data_test = np.clip(data_test, 0, 1)
		return data_train, data_test

	def binFrecsToSpikeSeries( self, instance, time_bin, max_spikes ):
		"""
		Sample  : bins con frecuencias con shape (canales, timesteps). Los timesteps representan bins de tiempo que 
				indican frecuencias siendo 1 la frecuencia maxima (es decir, 1 spike por dt completando todo el bin)
		time_bin: Cuanto tiempo sera simulado cada bin
		dt		: El delta-time de la simulación 
		"""
		assert instance.max()<=1,'Los instances deben venir escalados'

		instance_num_spikes = (instance*max_spikes).astype(np.int32)	# aqui sample tiene shape (canales, num_spikes en cada bin)
		instance_spikes     = []										# sample expresado en spikes

		dt = 0.1	
		for channel_idx, channel in enumerate(instance_num_spikes):	# cada channel es una serie de tiempo
			for bin_idx, bin_num_spikes in enumerate(channel):		# cada bin es un valor que representa la cantidad de spikes en ese bin de tiempo
				inicio_bin = bin_idx*time_bin
				fin_bin    = (bin_idx+1)*time_bin-dt
				spikes 	   = np.linspace(inicio_bin,fin_bin,bin_num_spikes,dtype=np.float32) # momentos para los spikes
				for spike_time in spikes: 
					instance_spikes.append( (float(spike_time), channel_idx) )

		return sorted(instance_spikes, key=lambda item:item[0])
	
	def preProcessAndSave(self, random_state, k=5):
		# CARGAR DATOS
		folder = './Data/recordings'
		files = os.listdir(folder)
		data    = []
		target  = []
		for file in tqdm(files):
			y, sr = librosa.load(f'{folder}/{file}', sr=8_000)
			y = librosa.util.fix_length(y, size=3000)
			bins = self.encodeAudio(y,sr) 
			data.append( bins )
			target.append( int(file.split('_')[0]) )
		
		data   = np.array(data)
		target = np.array(target)

		# SEPARAR DATOS PARA REALIZAR ESCALADO Y DATOS UTILES QUE PARTICIPARAN DEL PROCESO
		d_scale, d_util, t_scale, t_util = train_test_split(data,target,test_size=0.9, random_state=random_state, shuffle=True, stratify=target)
		d_scale, d_util = self.normalizeData(d_scale, d_util)

		# CONVERTIR LOS DATOS UTILES EN SPIKES
		d_util = [ self.binFrecsToSpikeSeries(instance, self.time_bin, self.max_spikes) for instance in d_util ]
		
		# CON LOS SPIKES UTILES HACER TRAIN TEST SPLIT
		d_train, d_test, t_train, t_test = train_test_split(d_util, t_util, train_size=0.8, random_state=random_state, stratify=t_util)
		

		# GENERAR K-FOLD
		idx_kfold = np.arange(len(d_train))
		idx_kfold = shuffle(idx_kfold, random_state=random_state)
		b = np.linspace(0,len(idx_kfold),k+1, dtype=np.int32)

		folds_idx = []
		for i in range(k):
			folds_idx.append( idx_kfold[b[i]:b[i+1]] )


		folds = []
		for i in range(k):
			folds.append({
				'd':[d_train[idx] for idx in folds_idx[i]],
				't':[t_train[idx] for idx in folds_idx[i]]
			})

		# SAVE
		with open('Data/fsdd_procesado.pickle','wb') as file:
			pickle.dump( {'folds':folds,
				 		  'train':{'d':d_train, 't':t_train},
						  'test' :{'d':d_test,  't':t_test}}, file )

	def load(self):
		with open('Data/fsdd_procesado.pickle','rb') as file:
			data_saved = pickle.load(file)
		self.folds 	= data_saved['folds']
		self.train  = data_saved['train']
		self.test   = data_saved['test']
		return self
	
	def getFolds(self):
		for fold in self.folds:
			yield fold
	