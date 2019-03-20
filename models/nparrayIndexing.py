import numpy as np
import itertools
import math
import time 
'''
Trying to optimize the iteration through multidimensional arrays in numpy
incredible, unoptimized pure python nested loop iteration is faster than
incensed numpy iterator
'''
start_optimized = time.time()
weights = np.ones( [7,7,154,30]) * .5
array_counter = np.zeros( [10,1])

for w in np.nditer( weights ):
    if w > 1. or w < 0. :
        print( 'weight out of bounds [0, 1] with val: '+str(w))
    else:
        index = math.floor(w*10)
        array_counter[index] += 1

end_optimized = time.time()
print(array_counter)
print('optimized: '+str(end_optimized - start_optimized)+'\n' )
    
start_unoptimized = time.time()

array_counter = np.zeros( [10,1])

[rows, cols,ch_ins,ch_outs] = weights.shape

for r in range( rows):
    for c in range(cols):
        for ch_in in range(ch_ins):
            for ch_out in range( ch_outs):
                w= weights[ r,c,ch_in,ch_out]
                if w > 1. or w < 0. :
                    print( 'weight out of bounds [0, 1] with val: '+str(w))
                else:
                    index = math.floor(w*10)
                    array_counter[index] += 1

end_unoptimized = time.time()
print(array_counter)
print('unoptimized: '+str(end_unoptimized - start_unoptimized) )


start_unoptimized = time.time()

array_counter = np.zeros( [10,1])

[rows, cols,ch_ins,ch_outs] = weights.shape

for r,c,ch_in,ch_out in itertools.product( range(rows),range(cols),range(ch_ins),range(ch_outs)):
    w= weights[ r,c,ch_in,ch_out]
    if w > 1. or w < 0. :
        print( 'weight out of bounds [0, 1] with val: '+str(w))
    else:
        index = math.floor(w*10)
        array_counter[index] += 1

end_unoptimized = time.time()
print(array_counter)
print('mized: '+str(end_unoptimized - start_unoptimized) )



