import SpikingConvNet
import matplotlib.pyplot as plt
import numpy as np

'''
start_from_scratch = False
phase = "Unused"
scn= SpikingConvNet.SpikingConvNet( phase,start_from_scratch)

magnitude_vec = scn.getTotalWeightsStats()
print( magnitude_vec)
'''
a_plus = .008
a_minus = -.0
a_decay = -.008

def modifyW( weight, a, n_times = 1):
    w = weight
    for _ in range(n_times):
        w += a * w * ( 1 - w)
    return w


w_init = 0.5
w_all_curr = w_init
w_up_curr = w_init
epochs = range(1000)
w_all_s = [ ]
w_up_s = [ ]
for _ in epochs:
    w_all_s.append(w_all_curr)
    w_up_s.append(w_up_curr)
    w_all_curr = modifyW( w_all_curr, a_plus, 0)
    w_all_curr = modifyW( w_all_curr, a_decay,1)
    w_up_curr = modifyW( w_up_curr, a_plus, 1)

fig, ax1 = plt.subplots(1, 1, figsize=(17, 11.5), tight_layout=True)
ax1.plot(epochs, w_all_s, 'r')
ax1.plot(epochs, w_up_s, 'b')
plt.show()

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





