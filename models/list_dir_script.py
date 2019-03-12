from os.path import dirname, realpath
import os

path = dirname(dirname(realpath(__file__)))
path_img_folder = path + '/datasets/TestingSet/Motor/'
l = os.listdir( path_img_folder)
print(l)
print(len(l))
        
