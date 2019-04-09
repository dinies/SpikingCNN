import context
from src.Layer import *

import matplotlib.pyplot as plt

class ConvolutionalLayer(Layer):

    def __init__(self, padding_type, stride, filter_dim, threshold_potential,\
            expected_input_dim, expected_output_dim, encoding_t, stdp_threshold,\
            a_plus = 0.02, a_minus = -0.01, a_decay = -0.001, stdp_flag = True):

        super().__init__( padding_type, stride, expected_output_dim)
        self.filter_dim = filter_dim
        self.threshold_potential = threshold_potential
        self.weights = None
        self.oldPotentials = tfe.Variable( np.zeros( expected_output_dim ))
        self.K_inh = np.ones(( expected_output_dim[1], expected_output_dim[2])).astype(np.uint8)
        self.encoding_t = encoding_t
        self.stdp_threshold = stdp_threshold
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
        self.spiked_counter=0
        self.inhibited_counter=0
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
        mu = 0.8
        std_dev = 0.05
        weights = np.random.normal(mu,std_dev,self.filter_dim)
        [rows, cols,ch_ins,ch_outs] = weights.shape
        for r,c,ch_in,ch_out in itertools.product( range(rows),range(cols),range(ch_ins),range(ch_outs)):
            w= weights[ r,c,ch_in,ch_out]
            if w > 1.:
                weights[ r,c,ch_in,ch_out]= 1
                print('outlier positive')
            elif w < 0.:
                weights[ r,c,ch_in,ch_out]= 0
                print('outlier negative')

        self.weights = weights
     
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
            elif w == 1:
                array_counter[-1] += 1
            else:
                index = math.floor(w*10)
                array_counter[index] += 1
        return array_counter
                        
    def getIterationInfo(self):
        return self.counter_strenghtened, self.counter_weakened, self.spiked_counter, self.inhibited_counter

    def makeOperation( self, input_to_layer, flag_plots = False):
        input_filter = tfe.Variable( self.weights )
        out_conv = tf.nn.conv2d(input_to_layer,input_filter,self.stride,self.padding_type)

        currSpikesNp = np.zeros( self.expected_output_dim)
        newPotentialsNp = tf.math.add( out_conv , self.oldPotentials).numpy()

       
        [_, rows, cols, channels] = newPotentialsNp.shape
        self.spiked_counter = 0

        for row, column, channel in itertools.product(range(rows),range(cols),range(channels)):
            if newPotentialsNp[0,row,column,channel] >= self.threshold_potential and self.K_inh[ row, column]==1 :

                currSpikesNp[0, row, column, channel ] = 1.0
                self.spiked_counter +=1
                # newPotentialsNp[0, row, column, channel ] = 0.0

        if flag_plots:
            K_inh_before = self.K_inh.copy()
            old_weights = self.weights.copy()
            old_spikes = currSpikesNp.copy()

        self.inhibited_counter= 0
        S, K_inh = self.lateral_inh_CPU( currSpikesNp, newPotentialsNp, self.K_inh)
        self.K_inh = K_inh

        self.spikes_presyn[:,:,:,:,self.curr_iteration] = input_to_layer
        self.spikes_postsyn[:,:,:,:,self.curr_iteration] = S

        if self.stdp_flag:
            self.STDP_learning()
                    
        currSpikes = tfe.Variable( S )
        newPotentials = tfe.Variable( newPotentialsNp )

        self.oldPotentials.assign( newPotentials)

        '''
        verifying check
        ggg = 0
        currSpikesVeryfication = currSpikes.numpy()
        for row, column, channel in itertools.product(range(rows),range(cols),range(channels)):
            if currSpikesVeryfication[0,row,column,channel] == 1.0 :
                ggg += 1
        print('current iteration: '+str(self.curr_iteration)+ ' remaining spikes: '+ str(ggg)+'\n')
        '''
         
        self.curr_iteration +=1


        if flag_plots:


            input_slice = input_to_layer.numpy()
            inputs = []
            for k in range( self.expected_input_dim[3]):
                in_k = input_slice[:,:,:,k].reshape([self.expected_input_dim[1],self.expected_input_dim[2]])
                inputs.append( in_k)

            out_conv_np = out_conv.numpy()
            outs = []
            for k in range( self.expected_output_dim[3]):
                out_k = out_conv_np[:,:,:,k].reshape([self.expected_output_dim[1],self.expected_output_dim[2]]) 
                outs.append( out_k)

            spikes_before = []
            spikes_after= []
            for k in range( self.expected_output_dim[3]):
                spike_b_k = old_spikes[:,:,:,k].reshape([self.expected_output_dim[1],self.expected_output_dim[2]]) 
                spike_a_k = S[:,:,:,k].reshape([self.expected_output_dim[1],self.expected_output_dim[2]]) 
                spikes_before.append( spike_b_k)
                spikes_after.append( spike_a_k)


            neuronal_maps_before_STDP = []
            neuronal_maps_after_STDP = []
            for i,j in itertools.product( range( self.filter_dim[2]), range( self.filter_dim[3])):
                map_b_k = old_weights[:,:,i,j].reshape([self.filter_dim[0],self.filter_dim[1]]) 
                map_a_k = self.weights[:,:,i,j].reshape([self.filter_dim[0],self.filter_dim[1]]) 
                neuronal_maps_before_STDP.append( map_b_k)
                neuronal_maps_after_STDP.append( map_a_k)



        
            '''
            1stLayer
            '''
            if  self.expected_output_dim == [1,160,250,4] :

               
                plt.figure( figsize=( 25,8))
                ax1 = plt.subplot( 121)
                ax1.imshow( inputs[0])
                ax1.set_title('Input to layer')
                ax2 = plt.subplot( 243)
                ax2.imshow( outs[0])
                ax3 = plt.subplot( 244)
                ax3.imshow( outs[1])
                ax4 = plt.subplot( 247)
                ax4.imshow( outs[2])
                ax5 = plt.subplot( 248)
                ax5.imshow( outs[3])


                fig1, axes1 = plt.subplots(2, 5, figsize=(25, 10), tight_layout=True)
                axes1[0][0].imshow( spikes_before[0])
                axes1[0][0].set_title('Spikes before inhibition')
                axes1[0][0].axis('off')
                axes1[0][1].imshow( spikes_before[1])
                axes1[0][1].axis('off')
                axes1[0][2].imshow( spikes_before[2])
                axes1[0][2].axis('off')
                axes1[0][3].imshow( spikes_before[3])
                axes1[0][3].axis('off')
                axes1[0][4].imshow( K_inh_before)
                axes1[0][4].set_title('K before inhibition')
                axes1[0][4].axis('off')

                axes1[1][0].imshow( spikes_after[0])
                axes1[1][0].set_title('Spikes after inhibition')
                axes1[1][0].axis('off')
                axes1[1][1].imshow( spikes_after[1])
                axes1[1][1].axis('off')
                axes1[1][2].imshow( spikes_after[2])
                axes1[1][2].axis('off')
                axes1[1][3].imshow( spikes_after[3])
                axes1[1][3].axis('off')
                axes1[1][4].imshow( K_inh)
                axes1[1][4].set_title('K after inhibition')
                axes1[1][4].axis('off')

                fig2, axes2 = plt.subplots(2, 4, figsize=(25, 10), tight_layout=True)
                axes2[0][0].imshow( neuronal_maps_before_STDP[0])
                axes2[0][0].set_title('Weights before STDP')
                axes2[0][0].axis('off')
                axes2[0][1].imshow( neuronal_maps_before_STDP[1])
                axes2[0][1].axis('off')
                axes2[0][2].imshow( neuronal_maps_before_STDP[2])
                axes2[0][2].axis('off')
                axes2[0][3].imshow( neuronal_maps_before_STDP[3])
                axes2[0][3].axis('off')

                axes2[1][0].imshow( neuronal_maps_after_STDP[0])
                axes2[1][0].set_title('Weights after STDP')
                axes2[1][0].axis('off')
                axes2[1][1].imshow( neuronal_maps_after_STDP[1])
                axes2[1][1].axis('off')
                axes2[1][2].imshow( neuronal_maps_after_STDP[2])
                axes2[1][2].axis('off')
                axes2[1][3].imshow( neuronal_maps_after_STDP[3])
                axes2[1][3].axis('off')


            '''
            2nd layer
            '''
            if self.expected_output_dim == [1,27,42,20]:
                plt.figure( figsize=( 25,8))
                ax1 = plt.subplot( 121)
                ax1.imshow( inputs[2])
                ax1.set_title('Input to layer')
                ax2 = plt.subplot( 243)
                ax2.imshow( outs[2])
                ax3 = plt.subplot( 244)
                ax3.imshow( outs[7])
                ax4 = plt.subplot( 247)
                ax4.imshow( outs[12])
                ax5 = plt.subplot( 248)
                ax5.imshow( outs[18])


                fig1, axes1 = plt.subplots(2, 5, figsize=(25, 10), tight_layout=True)
                axes1[0][0].imshow( spikes_before[2])
                axes1[0][0].set_title('Spikes before inhibition')
                axes1[0][0].axis('off')
                axes1[0][1].imshow( spikes_before[7])
                axes1[0][1].axis('off')
                axes1[0][2].imshow( spikes_before[12])
                axes1[0][2].axis('off')
                axes1[0][3].imshow( spikes_before[18])
                axes1[0][3].axis('off')
                axes1[0][4].imshow( K_inh_before)
                axes1[0][4].set_title('K before inhibition')
                axes1[0][4].axis('off')

                axes1[1][0].imshow( spikes_after[2])
                axes1[1][0].set_title('Spikes after inhibition')
                axes1[1][1].imshow( spikes_after[7])
                axes1[1][2].imshow( spikes_after[12])
                axes1[1][3].imshow( spikes_after[18])
                axes1[1][4].imshow( K_inh)
                axes1[1][4].set_title('K after inhibition')

                fig2, axes2 = plt.subplots(2, 6, figsize=(25, 10), tight_layout=True)
                axes2[0][0].imshow( neuronal_maps_before_STDP[2])
                axes2[0][0].set_title('Weights before STDP')
                axes2[0][0].axis('off')
                axes2[0][1].imshow( neuronal_maps_before_STDP[5])
                axes2[0][1].axis('off')
                axes2[0][2].imshow( neuronal_maps_before_STDP[9])
                axes2[0][2].axis('off')
                axes2[0][3].imshow( neuronal_maps_before_STDP[11])
                axes2[0][3].axis('off')
                axes2[0][4].imshow( neuronal_maps_before_STDP[15])
                axes2[0][4].axis('off')
                axes2[0][5].imshow( neuronal_maps_before_STDP[18])
                axes2[0][5].axis('off')

                axes2[1][0].imshow( neuronal_maps_after_STDP[2])
                axes2[1][0].set_title('Weights after STDP')
                axes2[1][0].axis('off')
                axes2[1][1].imshow( neuronal_maps_after_STDP[5])
                axes2[1][1].axis('off')
                axes2[1][2].imshow( neuronal_maps_after_STDP[9])
                axes2[1][2].axis('off')
                axes2[1][3].imshow( neuronal_maps_after_STDP[11])
                axes2[1][3].axis('off')
                axes2[1][4].imshow( neuronal_maps_after_STDP[15])
                axes2[1][4].axis('off')
                axes2[1][5].imshow( neuronal_maps_after_STDP[18])
                axes2[1][5].axis('off')
       
        return currSpikes

   
    def STDP_learning( self):
        [ _ , rows_out, cols_out, channels_out, _] = self.spikes_postsyn.shape
        channels_in = self.spikes_presyn.shape[3]
        self.counter_strenghtened = 0
        self.counter_weakened = 0

        for r_out, c_out, ch_out in itertools.product( range(rows_out),range(cols_out),range(channels_out)):
            current_matches = []
            if self.spikes_postsyn[0,r_out,c_out,ch_out,self.curr_iteration] == 1:
                for [in_row , in_col , r_w, c_w] in self.map_deconvolution_indexes[str(r_out)+','+str(c_out)] :   
                    for ch_in, t_input in itertools.product( range(channels_in),range( self.curr_iteration+1)):
                        presyn_neuron = self.spikes_presyn[0,in_row,in_col,ch_in,t_input]
                        if presyn_neuron == 1:
                            current_matches.append( [r_w,c_w,ch_in])
                if len( current_matches) >= self.stdp_threshold*self.curr_iteration:
                    self.counter_strenghtened += len( current_matches)
                    for [ r_w,c_w,ch_in]  in current_matches:
                        w= self.weights[r_w,c_w,ch_in,ch_out]
                        self.weights[r_w,c_w,ch_in,ch_out] = modifyWeight(w,self.a_plus)

        self.weights= np.add(self.weights, np.multiply(self.a_decay, np.multiply( self.weights,  np.subtract( 1 , self.weights))))    


               


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
                    if K_inh[i, j] == 0 or K[i,j] == 0:
                        S_inh[0,i, j, k] = 0
                        self.inhibited_counter +=1
                        continue
                    for kz in range(V.shape[3]):
                        if S[0,i, j, kz] == 1 and V[0,i, j, k] < V[0,i, j, kz]:
                            S_inh[0,i, j, k] = 0
                            self.inhibited_counter +=1
                            flag = True
                    if flag:
                        continue
                    else:
                        K[i, j] = 0
        S = np.multiply( S, S_inh)
        K_inh = np.multiply( K_inh, K)
        return S, K_inh






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

