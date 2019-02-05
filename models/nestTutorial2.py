from sklearn.svm import LinearSVC
from scipy.special import erf
import nest

import pylab
#
#
#

# Create neurons
neuron1 = nest.Create("iaf_psc_alpha")
nest.SetStatus(neuron1 , {"I_e": 376.})

neuron2 = nest.Create("iaf_psc_alpha")
nest.SetStatus(neuron2 , {"I_e": 378.})


multimeter = nest.Create("multimeter")
nest.SetStatus( multimeter, {"withtime":True, "record_from":["V_m"]})

spikedetector = nest.Create("spike_detector", params={ "withgid":True, "withtime":True})

nest.Connect(multimeter, neuron1)
nest.Connect(multimeter, neuron2)
nest.Connect( neuron1, spikedetector)
nest.Connect( neuron2, spikedetector)

nest.Simulate( 1000.0)

pylab.figure(1)
dmm = nest.GetStatus( multimeter)[0]

#print (dmm["events"]["V_m"].shape)
#print (dmm["events"]["V_m"][::2].shape)
#print (dmm["events"]["V_m"][1::2].shape)

Vms1 = dmm["events"]["V_m"][::2]
ts1 = dmm["events"]["times"][::2]
pylab.plot( ts1, Vms1)

Vms2 = dmm["events"]["V_m"][1::2]
ts2 = dmm["events"]["times"][1::2]
pylab.plot( ts2, Vms2)



dSD = nest.GetStatus( spikedetector, keys="events")[0]

pylab.figure(2)
evs = dSD["senders"]
ts = dSD["times"]

pylab.plot(ts,evs,".")

pylab.show()


