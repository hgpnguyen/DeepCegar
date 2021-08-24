'''
@author: LI Jiaying
'''


class Layers:
    def __init__(self, specLB, specUB):
        self.layertypes = []
        self.weights = []
        self.biases = []
        self.filters = []
        self.numfilters = []
        self.filter_size = [] 
        self.input_shape = []
        self.strides = []
        self.padding = []
        self.out_shapes = []
        self.pool_size = []
        self.numlayer = 0
        self.ffn_counter = 0
        self.conv_counter = 0
        self.residual_counter = 0
        self.maxpool_counter = 0
        self.activation_counter = 0
        self.specLB = []
        self.specUB = []
        self.original = []
        self.zonotope = []
        self.predecessors = []
        self.in_LB = specLB
        self.in_UB = specUB
        self.last_layer = None


    def calc_layerno(self):
        return self.ffn_counter + self.conv_counter + self.residual_counter + self.maxpool_counter + self.activation_counter


    def is_ffn(self):
        return not any(x in ['Conv2D', 'Conv2DNoReLU', 'Resadd', 'Resaddnorelu'] for x in self.layertypes)

    
    def check_layer_types(self):
        print('layer types:', self.layertypes)
