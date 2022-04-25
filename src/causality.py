import numpy as np
from executor import *
from typing import List
import tensorflow.compat.v1 as tf 
tf.disable_v2_behavior()

class Causality:
    def __init__(self, executor: Executor, target: int) -> None:
        self.ir_list = executor.ir_list
        self.exe = executor
        self.no_sample = 3000
        #self.x0, self.output_x0 = self.gen_x0()
        self.y0s = [None] * len(executor.ir_list)
        self.target = target

    def get_ie(self, do_layer: int, specLB: List[float], specUB: List[float]) -> List[float]:
        rand_input = np.random.uniform(specLB, specUB, (self.no_sample, len(specLB)))
        rand_input = np.reshape(rand_input, (len(specLB), -1))
        ie, num_step = [], 16
        res = []
        target = self.target
        if self.y0s[do_layer] is None:
            x = tf.placeholder(tf.float64, (len(specLB), self.no_sample))
            y = self.exe.get_parametric_model(do_layer, x)
            self.y0s[do_layer] = (x, y)
        else:
            x , y = self.y0s[do_layer]
        with tf.Session() as sess:
            output_x0 = sess.run(y, feed_dict = {x: rand_input})
            for do_neuron in range(len(specLB)):
                temp = rand_input.copy()
                for h_val in np.linspace(specLB[do_neuron], specUB[do_neuron], num_step):
                    temp[do_neuron][:] = h_val
                    output_y = sess.run(y, feed_dict = {x: temp})
                    dy = abs(output_y[target] - output_x0[target])
                    dy_sum = sum(dy)
                    avg = dy_sum / self.no_sample
                    ie.append(avg)
                mie = np.mean(np.array(ie))
                res.append(mie)
        
        return res
