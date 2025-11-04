import tonic
import numpy as np
from abc import ABC, abstractmethod
from sklearn.model_selection import train_test_split

import os
import librosa

class Dataset(ABC):
	def __init__(self, random_state):
		self.load_data(random_state)
		
		if (not hasattr(self, 'd_train')) or (not hasattr(self, 't_train')) or (not hasattr(self, 'd_test')) or (not hasattr(self, 't_train')):
			raise Exception('d_train,t_train,d_test o t_test no inicializado')
		
		if not hasattr(self,'n_dim'):
			raise Exception('El dataset debe tener n_dim')

	@abstractmethod
	def load_data(self, random_state):
		pass


class TIDIGITS(Dataset):
	def __init__(self,random_state=None):
		self.n_dim = 64
		super().__init__(random_state)

	def process_instance(self, instance):
		instance = [ (time/1000,idx) for time,idx,_ in instance ]
		return instance

	def load_data(self, random_state):
		data_train = tonic.datasets.NTIDIGITS18(save_to="./Data", train=True, single_digits=True)
		data_test  = tonic.datasets.NTIDIGITS18(save_to="./Data", train=False, single_digits=True)
		
		d_train,t_train = zip(*data_train)
		d_test ,t_test  = zip(*data_test)

		# data = [s for s in data_train ] + [ s for s in data_test ]
		# data, target = zip(*data)
		# data = [ self.process_instance(instance) for instance in data ]
		# data_train, data_test, target_train, target_test = train_test_split(data, target, train_size=0.5, stratify=target, random_state=random_state)
		
		self.d_train = [self.process_instance(instance) for instance in d_train]
		self.d_test  = [self.process_instance(instance) for instance in d_test]
		self.t_train = t_train
		self.t_test  = t_test

class FSDD(Dataset):
	def __init__(self,time_bin=80, max_spikes=20, n_dim=13, hop_lenght=256, random_state=None):
		self.time_bin   = time_bin
		self.max_spikes = max_spikes
		self.n_dim      = n_dim
		self.hop_lenght = hop_lenght
		super().__init__(random_state)

	def encodeAudio(self,y,sr):
		bins = librosa.feature.mfcc(y=y, sr=sr,n_mfcc=self.n_dim, hop_length=self.hop_lenght, n_fft=self.hop_lenght)
		return bins
	
	def normalizeData(self,data_train, data_test):				# data debe tener forma (samples, fetaures, time)
		mins = data_train.min(axis=(0,2), keepdims=True)
		maxs = data_train.max(axis=(0,2), keepdims=True)
		data_train = (data_train - mins) / (maxs - mins + 1e-8)
		data_test  = (data_test  - mins) / (maxs - mins + 1e-8)

		data_test = np.clip(data_test, 0, 1)
		return data_train, data_test

	def binFrecsToSpikeSeries( self, sample, time_bin, max_spikes ):
		"""
		Sample  : bins con frecuencias con shape (canales, timesteps). Los timesteps representan bins de tiempo que 
				indican frecuencias siendo 1 la frecuencia maxima (es decir, 1 spike por dt completando todo el bin)
		time_bin: Cuanto tiempo sera simulado cada bin
		dt		: El delta-time de la simulación 
		"""
		assert sample.max()<=1,'Los samples deben venir escalados'

		sample_num_spikes = (sample*max_spikes).astype(np.int32)	# aqui sample tiene shape (canales, num_spikes en cada bin)
		sample_spikes     = []										# sample expresado en spikes

		dt = 0.1	
		for channel_idx, channel in enumerate(sample_num_spikes):	# cada channel es una serie de tiempo
			for bin_idx, bin_num_spikes in enumerate(channel):		# cada bin es un valor que representa la cantidad de spikes en ese bin de tiempo
				inicio_bin = bin_idx*time_bin
				fin_bin    = (bin_idx+1)*time_bin-dt
				spikes 	   = np.linspace(inicio_bin,fin_bin,bin_num_spikes,dtype=np.float32) # momentos para los spikes
				for spike_time in spikes: 
					sample_spikes.append( (spike_time, channel_idx) )

		return sorted(sample_spikes, key=lambda item:item[0])

	def load_data(self, random_state):
		# LOAD AND PREPROCESS
		folder = './Data/recordings'
		files = os.listdir(folder)
		data    = []
		target  = []
		for file in files:
			y, sr = librosa.load(f'{folder}/{file}', sr=8_000)
			y = librosa.util.fix_length(y, size=3000)
			bins = self.encodeAudio(y,sr) 
			data.append( bins )
			target.append( int(file.split('_')[0]) )

		data 		  = np.array( data )
		target 		  = np.array(target)

		data_train,data_test, target_train, target_test = train_test_split(data, target, train_size=0.8,
																		shuffle=True, stratify=target, random_state=random_state)
		data_train, data_test = self.normalizeData(data_train, data_test)

		self.d_train = [ self.binFrecsToSpikeSeries( instance, time_bin=self.time_bin, max_spikes=self.max_spikes )  for instance in data_train]
		self.d_test  = [ self.binFrecsToSpikeSeries( instance, time_bin=self.time_bin, max_spikes=self.max_spikes )  for instance in data_test]
		self.t_train = target_train
		self.t_test  = target_test