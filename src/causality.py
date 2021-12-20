import numpy as np
from executor import *
from typing import List
import tensorflow.compat.v1 as tf 
tf.disable_v2_behavior()

class Causality:
    def __init__(self, executor: Executor, specLB: List[int], specUB: List[int]) -> None:
        self.ir_list = executor.ir_list
        self.exe = executor
        self.specLB, self.specUB = specLB, specUB
        self.no_sample = 5000
        self.x0, self.output_x0 = self.gen_x0()
    
    def gen_x0(self):
        specLB, specUB = self.specLB, self.specUB
        valid_x0s = []
        no_sample = self.no_sample
        with tf.Session() as sess:
            self.X = tf.placeholder(tf.float64, (len(specLB), no_sample))
            Y = self.exe.get_parametric_model(1, self.X)
            rand_input = np.random.uniform(specLB, specUB, (no_sample, len(specLB)))
            rand_input = np.reshape(rand_input, (len(specLB), -1))
            y = sess.run(Y, feed_dict = {self.X:rand_input})
            valid_x0s.append((rand_input, y))
        return rand_input, y

    def get_ie(self, do_neuron: int, do_layer: int, target: int, lowB: float, upperB: float):
        x = self.X
        for i in range(1, do_layer):
            x = self.ir_list[i].transformer(None, x)
        do_output = x
        ie, num_step = [], 16
        for i in range(do_layer, len(self.ir_list)):
            x = self.ir_list[i].transformer(None, x)
        Y = x
        #print(lowB, upperB)
        with tf.Session() as sess:
            do_Y = sess.run(do_output, feed_dict = {self.X: self.x0})
            #print("Do Y:", do_Y)
            for h_val in np.linspace(lowB, upperB, num_step):
                do_Y[do_neuron][:] = h_val
                output_y = sess.run(Y, feed_dict = {do_output: do_Y})
                dy = abs(output_y[target] - self.output_x0[target])
                dy_sum = sum(dy)
                avg = dy_sum / self.no_sample
                ie.append(avg)
        mie = np.mean(np.array(ie))
        
        return mie
