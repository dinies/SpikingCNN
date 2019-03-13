import SpikingConvNet

start_from_scratch = False
phase = "Unused"
scn= SpikingConvNet.SpikingConvNet( phase,start_from_scratch)

magnitude_vec = scn.getWeightsStatistics()
 
