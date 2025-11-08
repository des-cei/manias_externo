import ctypes
from enum import IntEnum
import numpy as np
import matplotlib.pyplot as plt

# ======================
#  ESTRUCTURAS DE DATOS
# ======================

class Config(ctypes.Structure):
    _fields_ = [
        ('n_neu'            ,ctypes.c_int),
        ('n_inputs'         ,ctypes.c_int),
        ('threshold'        ,ctypes.c_float),
        ('v_rest'           ,ctypes.c_float),
        ('tau'              ,ctypes.c_float),
        ('refractory_time'  ,ctypes.c_float),
        ('syn_delay'        ,ctypes.c_float),
        ('coda'		        ,ctypes.c_float),
        ('max_sim_time'     ,ctypes.c_float)]

class InputsRaw(ctypes.Structure):
	_fields_ = [('len',ctypes.c_int),
			   ('pre',ctypes.POINTER(ctypes.c_int)),
			   ('times',ctypes.POINTER(ctypes.c_float))]
	
class SynapsesRaw(ctypes.Structure):
	_fields_ = [('len' , ctypes.c_int),
			    ('pre' , ctypes.POINTER(ctypes.c_int)),
				('post', ctypes.POINTER(ctypes.c_int)),
				('w'   , ctypes.POINTER(ctypes.c_float))]
	
class VoltageMarker(ctypes.Structure):
    _fields_ = [('neu',ctypes.c_int),
				('time',ctypes.c_float),
				('voltage',ctypes.c_float)]	
	
class SpikeMarker(ctypes.Structure):
	_fields_ = [('neu',ctypes.c_int),
			    ('time',ctypes.c_float)]
	
class Node(ctypes.Structure):
	pass
Node._fields_ = [('data',ctypes.c_void_p),
				 ('next',ctypes.POINTER(Node))]

NodePtr = ctypes.POINTER(Node)

# ======================
#     EXACT-BALLENA
# ======================

class eBallena:
	# ======================
	#   CONFIGURACION C_LIB
	# ======================
	c_lib = ctypes.CDLL('./eBallena/ceBallena.so')

	c_lib.simulate.argtypes = [Config, InputsRaw, SynapsesRaw,SynapsesRaw, 
							ctypes.POINTER(NodePtr), ctypes.POINTER(NodePtr)]
	c_lib.simulate.restype  = None

	c_lib.freeList.argtypes = [NodePtr]
	c_lib.freeList.restype  = None

	# ======================
	#    CREATE OBJECTS
	# ======================

	def createConfig(n_inputs, n_neu, tau, max_sim_time, 
					refractory_time=5, threshold=30,v_rest=-70, syn_delay=0.01, coda=10):
			return Config(n_neu,n_inputs,threshold,v_rest, tau, refractory_time, syn_delay, coda, max_sim_time)

	def createInputsRaw(instance):	# INSTANCE=(TIME,IDX)
		if len(instance):
			instance = sorted( instance, key=lambda item:item[0], reverse=True ) # LOS TIEMPOS DEBEN VENIR AL REVES
			times,pre = zip(*instance)

			len_inputs = len(pre)
			return InputsRaw(
				len   = len_inputs,
				pre   = (ctypes.c_int*len_inputs)(*pre),
				times = (ctypes.c_float*len_inputs)(*times))
		else:
			return InputsRaw(len=0)


	def createSynapsesRaw(synapses):	# SYNAPSES = (PRE,POST,W)
		if len(synapses):
			pre,post,w = zip(*synapses)
			pre = np.array(pre, dtype=np.int32)
			post = np.array(post, dtype=np.int32)
			len_syn = len(synapses)
			return SynapsesRaw(
				len  = len_syn,
				pre  = (ctypes.c_int*len_syn)(*pre),
				post = (ctypes.c_int*len_syn)(*post),
				w    = (ctypes.c_float*len_syn)(*w)
			)
		else:
			return SynapsesRaw(len=0)

	# ======================
	#    PARSE RESULTS
	# ======================

	def parseVoltages(voltages):
		voltages_parsed = []
		while voltages:
			marker = ctypes.cast( voltages.contents.data, ctypes.POINTER(VoltageMarker) ).contents
			voltages_parsed.append( (marker.neu, marker.time, marker.voltage) )
			voltages = voltages.contents.next
		return np.array(voltages_parsed)

	def parseSpikes(spikes):	# (TIME,IDX)
		spikes_parsed = []
		while spikes:
			spike = ctypes.cast( spikes.contents.data, ctypes.POINTER(SpikeMarker) ).contents
			spikes_parsed.append( (spike.time,spike.neu) )
			spikes = spikes.contents.next
		return spikes_parsed


	def plotVoltage(voltage_markers, neu, config, n_points=10000, color='b'):	# voltage = (neu,time,volt)
		def lif_eq(t,t_init,v0):
			return config.v_rest + (v0-config.v_rest)*np.e**( -(t-t_init)/config.tau )

		voltage_markers = voltage_markers[voltage_markers[:,0]==neu][::-1]
		
		curva = []
		times = np.linspace(0,int(voltage_markers[-1,1])-0.01,n_points)
		for i in range(len(voltage_markers)-1):
			neu,      marker_time,     marker_volt 		= voltage_markers[i]
			next_neu, next_marker_time,next_marker_volt = voltage_markers[i+1]

			time_in_curve = times[ (times>=marker_time)&(times<next_marker_time) ]
			curva += [ lif_eq(t,marker_time, marker_volt) for t in time_in_curve ]

		plt.plot(times,curva,color=color, label=f'eBallena: neu {int(neu)}')

	def getState(spikes, config):
		state = np.array([0 for _ in range(config.n_neu)])
		for spike in spikes:
			state[spike[0]] += 1
		return state

	# =============
	#   SIMULATE
	# =============

	def simulate(config, instance, syn_in, syn_re, mode='SPIKES'):
		"""
		instance	 : [(time,idx)...]
		syn_{in-re}  : [(pre,post,w)...]
		mode	     : SPIKES | VOLTAGES | STATE
		"""
		if len(instance):	assert np.array(instance)[:,1].max() < config.n_inputs,'Hay input_idx mayores a los declarados en config'
		if len(syn_in):		assert np.array(syn_in)[:,1].max() < config.n_neu, 'Hay neu_idx mayores a los declarados en config'
		if len(syn_re):		assert np.array(syn_re)[:,:2].max() < config.n_neu, 'Hay neu_idx mayores a los declarados en config'

		inputs_raw = eBallena.createInputsRaw(instance)
		syn_in = eBallena.createSynapsesRaw( syn_in )
		syn_re = eBallena.createSynapsesRaw( syn_re )

		voltages = NodePtr()
		spikes   = NodePtr()

		eBallena.c_lib.simulate(config, inputs_raw, syn_in, syn_re, ctypes.pointer(voltages), ctypes.pointer(spikes))

		if mode=='STATE':
			ret_spikes   = eBallena.parseSpikes(spikes)
			ret			 = eBallena.getState(ret_spikes, config)
		elif mode=='SPIKES':
			ret = eBallena.parseSpikes(spikes)
		elif mode=='VOLTAGES':
			ret = eBallena.parseVoltages(voltages)

		eBallena.c_lib.freeList(voltages)
		eBallena.c_lib.freeList(spikes)

		return ret


