# TWO CONNECTED NEURONS
from sklearn.svm import LinearSVC
from scipy.special import erf
import nest

import pylab

# Create two neurons 

# Create neurons
neuron1 = nest.Create("iaf_psc_alpha")
nest.SetStatus(neuron1 , {"I_e": 376.})

neuron2 = nest.Create("iaf_psc_alpha")



# Create data gatherer
multimeter = nest.Create("multimeter")
nest.SetStatus( multimeter, {"withtime":True, "record_from":["V_m"]})



# Connect objects
nest.Connect(neuron1, neuron2, syn_spec = {"weight":0.5})
nest.Connect(multimeter, neuron1)
nest.Connect(multimeter, neuron2)


nest.Simulate( 1000.0)

# Plot results
pylab.figure(1)
dmm = nest.GetStatus( multimeter)[0]

Vms1 = dmm["events"]["V_m"][::2]
ts1 = dmm["events"]["times"][::2]
pylab.plot( ts1, Vms1)

pylab.figure(2)
Vms2 = dmm["events"]["V_m"][1::2]
ts2 = dmm["events"]["times"][1::2]
pylab.plot( ts2, Vms2)

pylab.show()


