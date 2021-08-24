'''
@author: Li Jiaying
'''

import tensorflow.compat.v1 as tf
import numpy as np
tf.disable_v2_behavior()


class NetworkProperty:
    def __init__(self, nr_categories):
        self.type = 0
        self.target = 0
        self.nr_labels = nr_categories
        self.str = "{PropertyUnset}"


    def set_robust(self, target, is_targeted=True):
        self.type = 1
        assert target>=0 and target<self.nr_labels
        self.target = target if is_targeted is True else -1-target
        #mask = tf.one_hot(self.target, self.nr_labels, dtype=tf.float64)
        mask = np.zeros(self.nr_labels, dtype=float)
        mask[self.target] = 1
        xmask = np.ones(self.nr_labels, dtype=bool)
        xmask[self.target] = False
        self.mask = mask.reshape((-1, 1))
        self.xmask = xmask
        if is_targeted:
            self.str = "[RobustProperty]: target @<" + str(target) + ">"
        else:
            self.str = "[RobustProperty]: target @!<" + str(target) + ">"


    def __str__(self):
        return self.str
    
    
    def get_mask(self):
        return self.mask
    
    
    def get_xmask(self):
        return self.xmask