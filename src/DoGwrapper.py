from DoG_filt_cuda import *
from os.path import dirname, realpath
import pdb
import matplotlib.pyplot as plt


class DoGwrapper(object):
    def __init__(self, img_dict, total_time = 15):
        self.num_layers = 1 # this stays for the num of layers
        # that will be empty in the end of the spike train
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














    '''
    Deprecated
    '''

    def printSpikeSeries(self, st):
        path_img = self.img_dict['path']       
        imagePlot = Image.open(path_img)

        fig, axes = plt.subplots(4, 4, figsize=(16, 10), tight_layout=True)
        axes[0][0].imshow( imagePlot)
        axes[0][1].imshow( st[:,:,0])
        axes[0][2].imshow( st[:,:,1])
        axes[0][3].imshow( st[:,:,2])
        axes[1][0].imshow( st[:,:,3])
        axes[1][1].imshow( st[:,:,4])
        axes[1][2].imshow( st[:,:,5])
        axes[1][3].imshow( st[:,:,6])
        axes[2][0].imshow( st[:,:,7])
        axes[2][1].imshow( st[:,:,8])
        axes[2][2].imshow( st[:,:,9])
        axes[2][3].imshow( st[:,:,10])
        axes[3][0].imshow( st[:,:,11])
        axes[3][1].imshow( st[:,:,12])
        axes[3][2].imshow( st[:,:,13])
        axes[3][3].imshow( st[:,:,14])

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

    def testPlotReport(self):
        path = dirname(dirname(realpath(__file__)))
        path_face = path + '/datasets/LearningSet/Face/image_0231.jpg'
        path_motor = path + '/datasets/LearningSet/Motor/motor_0011.jpg'

        st_face = self.applyDoGFilter( path_face)
        st_motor= self.applyDoGFilter( path_motor)

        st_face_expanded = np.expand_dims(st_face, axis=2)
        st_motor_expanded = np.expand_dims(st_motor, axis=2)
        face_image = Image.open(path_face)
        motor_image = Image.open(path_motor)
        face_tempMap = self.obtainTemperatureMap(st_face_expanded)
        motor_tempMap = self.obtainTemperatureMap(st_motor_expanded)

        fig1 , axes1 = plt.subplots(1, 2, figsize=(16, 5), tight_layout=True)
        axes1[0].imshow(face_image )
        axes1[0].axis('off')
        axes1[1].imshow(face_tempMap )
        axes1[1].axis('off')

        fig2 , axes2 = plt.subplots(2, 2, figsize=(16, 10), tight_layout=True)
        axes2[0][0].imshow( st_face[:,:,1])
        axes2[0][0].axis('off')
        axes2[0][1].imshow( st_face[:,:,2])
        axes2[0][1].axis('off')
        axes2[1][0].imshow( st_face[:,:,3])
        axes2[1][0].axis('off')
        axes2[1][1].imshow( st_face[:,:,4])
        axes2[1][1].axis('off')

        fig3 , axes3 = plt.subplots(1, 2, figsize=(16, 5), tight_layout=True)
        axes3[0].imshow(motor_image )
        axes3[0].axis('off')
        axes3[1].imshow(motor_tempMap )
        axes3[1].axis('off')

        fig4 , axes4 = plt.subplots(2, 2, figsize=(16, 10), tight_layout=True)
        axes4[0][0].imshow( st_motor[:,:,1])
        axes4[0][0].axis('off')
        axes4[0][1].imshow( st_motor[:,:,2])
        axes4[0][1].axis('off')
        axes4[1][0].imshow( st_motor[:,:,3])
        axes4[1][0].axis('off')
        axes4[1][1].imshow( st_motor[:,:,4])
        axes4[1][1].axis('off')

        fig1.savefig(path + '/images/face_temp')
        fig2.savefig(path + '/images/face_dog_out')
        fig3.savefig(path + '/images/motor_temp')
        fig4.savefig(path + '/images/motor_dog_out')




    def testWithOneImgDepthMap(self):
        path = dirname(dirname(realpath(__file__)))
        path_img = path + '/datasets/TrainingSet/Face/image_0297.jpg'
        path_img = path + '/datasets/LearningSet/Motor/motor_0127.jpg'
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


    def testWithOneImg(self):
        path = dirname(dirname(realpath(__file__)))
        
        path_img = path + '/datasets/LearningSet/Motor/motor_0127.jpg'
        imagePlot = Image.open(path_img)
        st = self.applyDoGFilter( path_img)

        fig, axes = plt.subplots(4, 4, figsize=(16, 10), tight_layout=True)
        axes[0][0].imshow( imagePlot)
        axes[0][1].imshow( st[:,:,0])
        axes[0][2].imshow( st[:,:,1])
        axes[0][3].imshow( st[:,:,2])
        axes[1][0].imshow( st[:,:,3])
        axes[1][1].imshow( st[:,:,4])
        axes[1][2].imshow( st[:,:,5])
        axes[1][3].imshow( st[:,:,6])
        axes[2][0].imshow( st[:,:,7])
        axes[2][1].imshow( st[:,:,8])
        axes[2][2].imshow( st[:,:,9])
        axes[2][3].imshow( st[:,:,10])
        axes[3][0].imshow( st[:,:,11])
        axes[3][1].imshow( st[:,:,12])
        axes[3][2].imshow( st[:,:,13])
        axes[3][3].imshow( st[:,:,14])

        plt.show()


if __name__ == '__main__':
    dog = DoGwrapper([], 10)
    dog.testPlotReport()
