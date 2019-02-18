from sklearn.svm import LinearSVC
from scipy.special import erf
import nest

import pylab


#
#
#
#
#

neuron = nest.Create("iaf_psc_alpha")

#print (nest.GetStatus(neuron)[0]['t_ref'])

nest.SetStatus(neuron , {"I_e": 376.})

multimeter = nest.Create("multimeter")
nest.SetStatus( multimeter, {"withtime":True, "record_from":["V_m"]})

spikedetector = nest.Create("spike_detector", params={ "withgid":True, "withtime":True})

nest.Connect(multimeter, neuron)
nest.Connect( neuron, spikedetector)

nest.Simulate( 1000.0)

dms = nest.GetStatus( multimeter)[0]
Vms = dms["events"]["V_m"]
ts = dms["events"]["times"]

pylab.figure(1)
pylab.plot( ts, Vms)

dSD = nest.GetStatus( spikedetector, keys="events")[0]
evs = dSD["senders"]
ts = dSD["times"]
pylab.figure(2)
pylab.plot(ts,evs,".")

pylab.show()


