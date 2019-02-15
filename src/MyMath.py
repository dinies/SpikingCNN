import numpy as np


class MyMath(object):

    def boxplus( rad_angle_alfa, rad_angle_beta):
        c_a, s_a= np.cos(rad_angle_alfa), np.sin(rad_angle_alfa) 

        c_b, s_b= np.cos(rad_angle_beta), np.sin(rad_angle_beta) 

        R_alfa = np.array([[ c_a, -s_a], [s_a, c_a ]])
        R_beta = np.array([[ c_b, -s_b], [s_b, c_b ]])
        R = np.matmul( R_alfa , R_beta)
        return np.arctan2( R[1][0], R[0][0])

