'''
@author: Li Jiaying
'''

import numpy as np
from concrete_nodes import *
from functools import reduce



# ir_list can be obtained via optimizer
class Executor:
    def __init__(self, ir_list, testing = False):
        """
        Arguments
        ---------
        ir_list: list
            list of Node-Objects (e.g. from DeepzonoNodes), first one must create an abstract element
        """
        self.ir_list = ir_list
        self.concrete_model_cache = {}
        self.concrete_model_list_cache = {}
        self.parametric_model_cache = {}
        self.testing = testing
        #self.sess = tf.get_default_session()

    
    # this function is a pure function to get concrete models
    # def get_concrete_models(self, start):
    #     # note: the output vector contains the first input tensor
    #     # and thus, the output vector has |end-start+1| dimension
    #     if not start in self.concrete_model_cache:
    #         model = []
    #         model.append(self.ir_list[start].get_fresh_input())
    #         for i in range(start, len(self.ir_list)):
    #             model.append(self.ir_list[i].transformer(None, model[-1]))
    #         self.concrete_model_cache[start] = model
    #     return self.concrete_model_cache[start]
    
    
    def get_concrete_model_list(self, start, start_tensor=None):
        # note: the output vector contains the first input tensor
        # and thus, the output vector has |end-start+1| dimension
        parametric = (start_tensor is None)
        if not parametric and start in self.concrete_model_list_cache:
            return self.concrete_model_list_cache[start]
        if parametric:
            start_tensor = self.ir_list[start].get_fresh_input()
        model = []
        model.append(start_tensor)
        for i in range(start, len(self.ir_list)):
            model.append(self.ir_list[i].transformer(None, model[-1]))
        if not parametric:
            self.concrete_model_list_cache[start] = model
            self.concrete_model_cache[start] = model[-1]
        return model
    
    
    def get_fresh_input(self, start):
        return self.ir_list[start].get_fresh_input()
        
        
    def get_concrete_model(self, start):
        # note: the output vector contains the first input tensor
        # and thus, the output vector has |end-start+1| dimension
        if start in self.concrete_model_cache:
            return self.concrete_model_cache[start]
        x = self.ir_list[start].get_fresh_input()
        for i in range(start, len(self.ir_list)):
            x = self.ir_list[i].transformer(None, x)
        self.concrete_model_cache[start] = x
        return x
    
    
    def get_parametric_model(self, start, start_tensor):
        # note: the output vector contains the first input tensor
        # and thus, the output vector has |end-start+1| dimension
        model = []
        x = start_tensor
        for i in range(start, len(self.ir_list)):
            x = self.ir_list[i].transformer(None, x)
        return x


    # bug! This function may lead to 'RecursionError: maximum recursion depth exceeded'
    # thus we discard this function. We may refine it in the future if needed.
    # def get_parametric_models(self, start):
    #     global depth
    #     # start_tensor is not None, we should use the parametric model
    #     if not start in self.parametric_model_cache:
    #         model = []
    #         layer_op = lambda in_tensor: in_tensor
    #         model.append(layer_op)
    #         for i in range(start, len(self.ir_list)):
    #             print('===================== start:', start, ' current:', i, '===========================')
    #             layer_op = lambda in_tensor: self.ir_list[i].transformer(None, model[-1](in_tensor))
    #             # layer_op = lambda x: parametric_model_function(self.ir_list[i].transformer, model[-1](x), start, i)
    #             # print('=== depth:', depth, '===')
    #             model.append(layer_op)
    #         self.parametric_model_cache[start] = model
    #     return self.parametric_model_cache[start]



    def forward(self, spec, start=1, full_layers=False):
        spec = np.reshape(spec, (-1, 1))
        # we evaluate all the output vectors for each IR:
        if full_layers:
            models = self.get_concrete_model_list(start)
            ys = models.eval(feed_dict={models[0]:spec})
            y = ys[-1]
            return np.argmax(y), ys
        else:  # we only evaluate the output vectors of the final layer
            x = self.get_fresh_input(start)
            model = self.get_concrete_model(start)
            with tf.Session() as sess:
               y = sess.run(model, feed_dict={x:spec})
            return np.argmax(y), y
        # dominant_class = np.argmax(y)
        # y = np.reshape(y, (-1, )).tolist()
        # print('input:', spec.tolist(), ' \noutput:', y, ' \nlabel:', dominant_class)
        # return dominant_class, y 
