import SpikingConvNet
import numpy as np
'''
start_from_scratch = False
phase = "Unused"
scn= SpikingConvNet.SpikingConvNet( phase,start_from_scratch)

magnitude_vec = scn.getTotalWeightsStats()
print( magnitude_vec)
'''
a_plus = .5
a_minus = -.1
a_decay = -.0

def modifyW( weight, a, n_times = 1):
    w = weight
    for _ in range(n_times):
        w += a * w * ( 1 - w)
    return w


w_init = 0.5
w_final = modifyW( w_init, a_plus,  1)
print( w_final)


'''

initial_w_to_str_100_times = 0.3
w_to_str_100_times = initial_w_to_str_100_times
for i in range(100):
    w_to_str_100_times += a_plus * w_to_str_100_times * ( 1- w_to_str_100_times)
    w_to_str_100_times += a_decay* w_to_str_100_times * ( 1- w_to_str_100_times)
print( "strenghtening  100 times "+str( initial_w_to_str_100_times) +" ==> "+ str( w_to_str_100_times))

initial_w_to_weaken_100_times = 0.7
w_to_weak_100_times = initial_w_to_weaken_100_times
for i in range(100):
    w_to_weak_100_times += a_minus* w_to_weak_100_times* ( 1- w_to_weak_100_times)
    w_to_weak_100_times += a_decay* w_to_weak_100_times* ( 1- w_to_weak_100_times)

print( "weakening 100 times "+str( initial_w_to_weaken_100_times) +" ==> "+ str( w_to_weak_100_times))


p = np.zeros( [3,3])
p = np.add( p , 0.5)
r1 = np.add( p, np.multiply(a_decay, np.multiply( p , np.subtract(1 , p))))
k = np.zeros( [3,3])
k += 0.5
r2= k + ( a_decay* k*( 1-k))

print( r1)
print( r2)
'''





