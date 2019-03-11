from DoG_filt_cuda import *
from os.path import dirname, realpath
import pdb
import matplotlib.pyplot as plt


class DoGwrapper(object):
    def __init__(self, img_dict, total_time = 15):
        self.num_layers = 6
        self.total_time = total_time
        
        self.DoG_params = {'img_size': (250, 160),
                'DoG_size': 7, 'std1': 1., 'std2': 2.} 
        self.img_dict = img_dict

    def getSpikeTrains( self):
        spikeTrain = self.applyDoGFilter( self.img_dict['path'])
        return spikeTrain
            

    def applyDoGFilter(self, imgPath):
        img_size = self.DoG_params['img_size']

        filt = DoG(
                self.DoG_params['DoG_size'],
                self.DoG_params['std1'],
                self.DoG_params['std2'])

        st = DoG_filter(
            imgPath, filt,
            img_size, self.total_time,
            self.num_layers)

        return st



















    def obtainTemperatureMap(self,spike_train):
        tempMap = np.zeros(spike_train.shape[0:2])
        depth = spike_train.shape[3]
        i = 0
        for floor in spike_train:
            j = 0
            for trainEncaps in floor:
                max_index= -1
                k = 0
                for spike in trainEncaps[0]:
                    if spike == 1 :
                        max_index= k
                    k += 1
                if max_index >= 0 :
                    temperature = 1 - (depth - max_index)/depth
                    tempMap[i][j] = temperature
                j += 1
            i +=1
        return tempMap


    def testWithOneImg(self):
        path = dirname(dirname(realpath(__file__)))
        path_img = path + '/datasets/TrainingSet/Face/image_0297.jpg'
        
        st = self.applyDoGFilter( path_img)

        st = np.expand_dims(st, axis=2)

        tempMap = self.obtainTemperatureMap(st)
        #   window =tempMap[130:190][60:100] 
        #   print( window.shape())
        imagePlot = Image.open(path_img)

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6.5), tight_layout=True)
        im1 = ax1.imshow( imagePlot )
        im2 = ax2.imshow( tempMap )
        plt.show()



if __name__ == '__main__':
    dog = DoGwrapper([])
    dog.testWithOneImg()
