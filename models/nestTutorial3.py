# SPECIFIC CONNECTIONS
from sklearn.svm import LinearSVC
from scipy.special import erf
import nest

import pylab
#
# Create input for neurons
noise_ex = nest.Create("poisson_generator")
noise_in = nest.Create("poisson_generator")
nest.SetStatus(noise_ex, {"rate":80000.0})
nest.SetStatus(noise_in, {"rate":15000.0})
syn_dict_ex = {"weight":1.2}
syn_dict_in = {"weight":-2.0}


# Create neurons
neuron = nest.Create("iaf_psc_alpha")
nest.SetStatus(neuron , {"I_e": 0.})


multimeter = nest.Create("multimeter")
nest.SetStatus( multimeter, {"withtime":True, "record_from":["V_m"]})

spikedetector = nest.Create("spike_detector", params={ "withgid":True, "withtime":True})

nest.Connect(multimeter, neuron)
nest.Connect( neuron, spikedetector)
nest.Connect( noise_ex, neuron, syn_spec=syn_dict_ex)
nest.Connect( noise_in, neuron, syn_spec=syn_dict_in)

nest.Simulate( 1000.0)

pylab.figure(1)
dmm = nest.GetStatus( multimeter)[0]

Vms = dmm["events"]["V_m"]
ts = dmm["events"]["times"]
pylab.plot( ts, Vms)


dSD = nest.GetStatus( spikedetector, keys="events")[0]

pylab.figure(2)
evs = dSD["senders"]
ts = dSD["times"]

pylab.plot(ts,evs,".")

pylab.show()


