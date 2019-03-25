import context
from src.Layer import *


class ConvolutionalLayer(Layer):

    def __init__(self, padding_type, stride, filter_dim, threshold_potential,\
            expected_input_dim, expected_output_dim, encoding_t,\
            a_plus = 0.02, a_minus = -0.01, a_decay = -0.001, stdp_flag = True):

        super().__init__( padding_type, stride, expected_output_dim)
        self.filter_dim = filter_dim
        self.threshold_potential = threshold_potential
        self.weights = None
        self.oldPotentials = tfe.Variable( np.zeros( expected_output_dim ))
        self.K_inh = np.ones(( expected_output_dim[1], expected_output_dim[2])).astype(np.uint8)
        self.encoding_t = encoding_t
        self.spikes_presyn = np.zeros(  expected_input_dim +[self.encoding_t])
        self.spikes_postsyn = np.zeros(  expected_output_dim +[self.encoding_t])
        self.curr_iteration = 0
        self.expected_input_dim = expected_input_dim
        self.a_plus = a_plus
        self.a_minus = a_minus
        self.a_decay = a_decay
        self.stdp_flag = stdp_flag
        self.counter_strenghtened =0
        self.counter_weakened =0
        self.map_deconvolution_indexes = createMapDeconvIndexes(\
                filter_dim, expected_input_dim, expected_output_dim)

    
    def resetStoredData( self):
        self.curr_iteration = 0
        self.spikes_presyn = np.zeros(  self.expected_input_dim +[self.encoding_t])
        self.spikes_postsyn = np.zeros(  self.expected_output_dim +[self.encoding_t])
        
    def resetOldPotentials( self):
        self.oldPotentials = tfe.Variable( np.zeros( self.expected_output_dim ))

    def resetInhibition( self):
        self.K_inh = np.ones(( self.expected_output_dim[1],\
                self.expected_output_dim[2])).astype(np.uint8)

    def resetLayer(self):
        self.resetOldPotentials()
        self.resetInhibition()
        self.resetStoredData()

    def createWeights(self):
        self.weights = np.ones(self.filter_dim)*0.5

    def loadWeights(self,path,layer_index):
        self.weights = np.load( path + 'weight_'+ str(layer_index)+ '.npy')

    def saveWeights(self,path,layer_index):
        np.save( path + 'weight_'+ str(layer_index)+ '.npy', self.weights)

    def getWeightsStatistics( self):

        array_counter = np.zeros( [10,1])
        [rows, cols,ch_ins,ch_outs] = self.weights.shape

        for r,c,ch_in,ch_out in itertools.product( range(rows),range(cols),range(ch_ins),range(ch_outs)):

            w= self.weights[ r,c,ch_in,ch_out]
            if w > 1. or w < 0. :
                print( 'weight out of bounds [0, 1] with val: '+str(w))
            else:
                index = math.floor(w*10)
                array_counter[index] += 1
        return array_counter
                        
    def getSynapseChangeInfo(self):
        return self.counter_strenghtened, self.counter_weakened

    def makeOperation( self, input_to_layer ):
        input_filter = tfe.Variable( self.weights )
        out_conv = tf.nn.conv2d(input_to_layer,input_filter,self.stride,self.padding_type)

        currSpikesNp = np.zeros( self.expected_output_dim)
        newPotentialsNp = tf.math.add( out_conv , self.oldPotentials).numpy()

       
        [_, rows, cols, channels] = newPotentialsNp.shape

        for row, column, channel in itertools.product(range(rows),range(cols),range(channels)):
            if newPotentialsNp[0,row,column,channel] >= self.threshold_potential:
                currSpikesNp[0, row, column, channel ] = 1.0
                newPotentialsNp[0, row, column, channel ] = 0.0
        S, K_inh = self.lateral_inh_CPU( currSpikesNp, newPotentialsNp, self.K_inh)
        self.K_inh = K_inh

        self.spikes_presyn[:,:,:,:,self.curr_iteration] = input_to_layer
        self.spikes_postsyn[:,:,:,:,self.curr_iteration] = S

        if self.stdp_flag:
            self.STDP_learning()
                    
        currSpikes = tfe.Variable( S )
        newPotentials = tfe.Variable( newPotentialsNp )

        self.oldPotentials.assign( newPotentials)

        self.curr_iteration +=1

        return currSpikes

   
    def STDP_learning( self):
        [ _ , rows_out, cols_out, channels_out, _] = self.spikes_postsyn.shape
        channels_in = self.spikes_presyn.shape[3]
        self.counter_strenghtened = 0
        self.counter_weakened = 0

        for r_out, c_out, ch_out in itertools.product( range(rows_out),range(cols_out),range(channels_out)):
            if self.spikes_postsyn[0,r_out,c_out,ch_out,self.curr_iteration] == 1:
                for [in_row , in_col , r_w, c_w] in self.map_deconvolution_indexes[str(r_out)+','+str(c_out)] :   
                    for ch_in, t_input in itertools.product( range(channels_in),range( self.curr_iteration+1)):
                        presyn_neuron = self.spikes_presyn[0,in_row,in_col,ch_in,t_input]
                        if presyn_neuron == 1:
                            self.counter_strenghtened +=1
                            w= self.weights[r_w,c_w,ch_in,ch_out]
                            self.weights[r_w,c_w,ch_in,ch_out] = modifyWeight(w,self.a_plus)

        self.weights += self.a_decay * self.weights * ( 1 - self.weights)       


               


    # BEGIN Function borrowed from the paper autors
    @jit
    def lateral_inh_CPU(self, S, V, K_inh):
        S_inh = np.ones(S.shape, dtype=S.dtype)
        K = np.ones(K_inh.shape, dtype=K_inh.dtype)
        for i in range(V.shape[1]):
            for j in range(V.shape[2]):
                for k in range(V.shape[3]):
                    flag = False
                    if S[0,i, j, k] != 1:
                        continue
                    if K_inh[i, j] == 0:
                        S_inh[0,i, j, k] = 0
                        continue
                    for kz in range(V.shape[3]):
                        if S[0,i, j, kz] == 1 and V[0,i, j, k] < V[0,i, j, kz]:
                            S_inh[0,i, j, k] = 0
                            flag = True
                    if flag:
                        continue
                    else:
                        K[i, j] = 0
        S *= S_inh
        K_inh *= K
        return S, K_inh
    # END Function borrowed from the paper autors






# Static functions
def createMapDeconvIndexes( filt_dim, input_dim, output_dim ):
    map_indexes = {}
    [ _ , rows, columns,  _] = output_dim

    for r,c in itertools.product(range(rows),range(columns)):
        indexes = computeDeconvolutionIndexesSamePaddingOddFilterDim(r,c,filt_dim,input_dim,output_dim)
        key= str(r)+','+str(c)
        map_indexes[key] = indexes
        
    return map_indexes


# Given a coordinate of a square of the output layer of a convolution
# returns a list of quadruples of the form :
# [ input row, input column, weight row, weight column ] 
# to optimize the code bring this execution at the creation of the class and
# store all the indexes in a map with [r,c] as keys and the correspondences as values
def computeDeconvolutionIndexesSamePaddingOddFilterDim(\
    row, column, filter_dim, expected_input_dim, expected_output_dim ):

    indexes_list = []
    offset_r = math.ceil( (filter_dim[0]-1)/2)
    offset_c = math.ceil( (filter_dim[1]-1)/2)
    i = 0
    for r in range( row - offset_r, row + offset_r+1):
        j = 0
        for c in range( column - offset_c, column + offset_c+1):
            if 0<=r<expected_input_dim[1] and 0<=c<expected_input_dim[2]: 
                indexes_list.append( [ r, c, i, j] )
            j += 1
        i +=1

    return indexes_list 
    
def modifyWeight( weight, a, n_times = 1):
    w = weight
    for _ in range(n_times):
        w += a * w * ( 1 - w)
    return w

