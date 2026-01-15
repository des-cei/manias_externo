from brian2 import *
from tqdm import tqdm

# ===========================
#  SIMULACIONES CON BRIAN2
# ===========================

def simulacionBrianStates( config, data, syn_in, syn_re):
	"""Simula un dataset completo y devuele los estados"""	
	start_scope()
	I = SpikeGeneratorGroup(config.n_inputs,[],[]*ms)

	tau 	= config.tau*ms
	v_rest  = config.v_rest*mV
	v_thres = config.threshold*mV
	refractory = config.refractory_time
	lif_eq  = """dv/dt = -(v-v_rest)/tau : volt (unless refractory)"""
	G = NeuronGroup(config.n_neu, lif_eq, method='exact', refractory=refractory*ms, threshold="v>=v_thres", reset="v=v_rest")
	G.v = v_rest

	pre,post,w_in = zip(*syn_in)
	pre  		  = np.array(pre, dtype=np.int32)
	post 		  = np.array(post, dtype=np.int32)
	SIN 	 	  = Synapses(I,G,model="w:volt",on_pre="v_post+=w")
	SIN.connect(i=pre, j=post)
	SIN.w 	 = np.array(w_in)*mV

	pre,post,w_re = zip(*syn_re)
	pre  		  = np.array(pre, dtype=np.int32)
	post 		  = np.array(post, dtype=np.int32)
	SRE 	 	  = Synapses(G,G,model="w:volt",on_pre="v_post+=w")
	SRE.connect(i=pre, j=post)
	SRE.w 	 = np.array(w_re)*mV

	spikemon = SpikeMonitor(G)

	defaultclock.dt = 0.5*ms
	store()

	states = []
	for instance in tqdm(data):
		restore()

		input_time,input_idx = zip(*instance)
		I.set_spikes(input_idx, input_time*ms)

		run(config.max_sim_time*ms)
		
		state = [0 for _ in range(config.n_neu)]
		for idx in spikemon.i:
			state[idx] += 1
		states.append(state)

	states = np.array(states)
	return states

def simulacionBrian2Voltages( config, instance, syn_in, syn_re):
	"""Simula 1 solo instance y devuele los voltajes"""
	
	start_scope()
	input_time,input_idx = zip(*instance)
	I = SpikeGeneratorGroup(config.n_inputs,input_idx,np.array(input_time)*ms)
	
	tau 	= config.tau*ms
	v_rest  = config.v_rest*mV
	v_thres = config.threshold*mV
	refractory = config.refractory_time
	lif_eq  = """dv/dt = -(v-v_rest)/tau : volt (unless refractory)"""
	G = NeuronGroup(config.n_neu, lif_eq, method='exact', refractory=refractory*ms, threshold="v>=v_thres", reset="v=v_rest")
	G.v = v_rest

	pre,post,w_in = zip(*syn_in)
	pre           = np.array(pre,dtype=np.int32)
	post          = np.array(post,dtype=np.int32)
	SIN 	 	  = Synapses(I,G,model="w:volt",on_pre="v_post+=w")
	SIN.connect(i=pre, j=post)
	SIN.w 	 = np.array(w_in)*mV

	pre,post,w_re = zip(*syn_re)
	pre           = np.array(pre,dtype=np.int32)
	post          = np.array(post,dtype=np.int32)
	SRE 	 	  = Synapses(G,G,model="w:volt",on_pre="v_post+=w")
	SRE.connect(i=pre, j=post)
	SRE.w 	 = np.array(w_re)*mV

	# spikemon = SpikeMonitor(G)
	statemon = StateMonitor(G,'v',record=True)
	run(config.max_sim_time*ms)

	return statemon.t/ms, statemon.v/mV


# ====================
#  GENERAR RESERVOIRS
# ====================

def generateReservoir(num_neuron=1000,prob_exitatory=0.8, L=3, c_ee=0.2, c_ei=0.1, c_ie=0.05, c_ii=0.3):
	assert round(num_neuron**(1/3))**3 == num_neuron, 'num_ neuron debe ser un cubo'

	# GENERATE NEURON POSITIONS AND TYPE (EX, INH)
	reservoir_side 	= round(num_neuron**(1/3) )

	reservoir = []
	for x in range(reservoir_side):
		for y in range(reservoir_side):
			for z in range(reservoir_side):
				reservoir.append( (x,y,z) )

	reservoir = np.array(reservoir)
	neuron_type = np.random.random(size=len(reservoir))<prob_exitatory

	# GENERATE SYNAPSES
	syn_re = []
	for position_pre,pre_exitatory, pre_idx in zip(reservoir,neuron_type,range(num_neuron)):
		for position_post, post_exitatory, post_idx in zip(reservoir,neuron_type,range(num_neuron)):

			if pre_idx != post_idx:

				euclidian_dist = ((position_pre-position_post)**2).sum()**0.5
				
				if pre_exitatory and post_exitatory:			C=c_ee
				elif pre_exitatory and not post_exitatory:		C=c_ei
				elif not pre_exitatory and post_exitatory:		C=c_ie
				elif not pre_exitatory and not post_exitatory:	C=c_ii

				prob_connection = C*np.exp( -(euclidian_dist/L)**2 )
				if np.random.random() < prob_connection:
					syn_re.append( (pre_idx, post_idx) )
	return syn_re, neuron_type

def generateTopologyInput(n_inputs, n_neu_reservoir=1000, prob_connection=0.2):
	syn_in = []
	for pre in range(n_inputs):
		for post in range(n_neu_reservoir):
			if np.random.random()<prob_connection:
				syn_in.append( (pre,post) )

	syn_in = np.array(syn_in)
	syn_type = np.random.random( size=len(syn_in) )<0.5
	return syn_in, syn_type


def assignWeightsReservoir(syn_re, neuron_type, w_re):
	return np.array( [(pre,post,w_re) if neuron_type[pre] else (pre,post,-w_re) for pre,post in syn_re] )

def assignWeightsInputs(syn_in, syn_type, w_in):
	w_in = w_in*syn_type - w_in*(1-syn_type)
	return np.array([ (pre,post,w) for (pre,post),w in zip(syn_in,w_in) ])