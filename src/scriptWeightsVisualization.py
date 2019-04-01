from SpikingConvNet import *
import matplotlib.pyplot as plt
import numpy as np

start_from_scratch = False
phase = "None"
scn= SpikingConvNet( phase,start_from_scratch)
index = 0
for layer in scn.layers:
    layer.loadWeights( scn.pathWeights, index )
    index+=1

'''
1st Layer w dim 5x5x4
'''
'''
w_1 = scn.layers[0].weights

[rows, cols,ch_ins,ch_outs] = w_1.shape

slice_1_w_1 = w_1[:,:,:,0]
slice_1_w_1_resh = slice_1_w_1.reshape( [5,5])

slice_2_w_1 = w_1[:,:,:,1]
slice_2_w_1_resh = slice_2_w_1.reshape( [5,5])

slice_3_w_1 = w_1[:,:,:,2]
slice_3_w_1_resh = slice_3_w_1.reshape( [5,5])

slice_4_w_1 = w_1[:,:,:,3]
slice_4_w_1_resh = slice_4_w_1.reshape( [5,5])

fig1, axes = plt.subplots(2, 2, figsize=(16, 10), tight_layout=True)
axes[0][0].imshow( slice_1_w_1_resh)
axes[0][1].imshow( slice_2_w_1_resh)
axes[1][0].imshow( slice_3_w_1_resh)
axes[1][1].imshow( slice_4_w_1_resh)
'''





'''
2nd Layer w dim 5x5x4
'''
w_2= scn.layers[2].weights


slice_1_w_2 = w_2[:,:,3,3]
slice_1_w_2_resh = slice_1_w_2.reshape( [17,17])

slice_2_w_2 = w_2[:,:,3,7]
slice_2_w_2_resh = slice_2_w_2.reshape( [17,17])

slice_3_w_2 = w_2[:,:,3,15]
slice_3_w_2_resh = slice_3_w_2.reshape( [17,17])

slice_4_w_2 = w_2[:,:,3,18]
slice_4_w_2_resh = slice_3_w_2.reshape( [17,17])


fig2, axes = plt.subplots(2, 2, figsize=(16, 10), tight_layout=True)
axes[0][0].imshow( slice_1_w_2_resh)
axes[0][1].imshow( slice_2_w_2_resh)
axes[1][0].imshow( slice_3_w_2_resh)
axes[1][1].imshow( slice_4_w_2_resh)
 

plt.show()




































