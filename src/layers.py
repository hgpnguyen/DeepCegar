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
        self.pool_counter = 0
        self.concat_counter = 0
        self.tile_counter = 0
        self.activation_counter = 0
        self.specLB = []
        self.specUB = []
        self.original = []
        self.zonotope = []
        self.predecessors = []
        self.in_LB = specLB
        self.in_UB = specUB
        self.last_layer = None
    
    def copy(self, layers:'Layers') -> None:
        self.layertypes = layers.layertypes.copy()
        self.weights = layers.weights.copy()
        self.biases = layers.biases.copy()
        self.filters = layers.filters.copy()
        self.numfilters = layers.numfilters.copy()
        self.filter_size = layers.filter_size.copy()
        self.input_shape = layers.input_shape.copy()
        self.strides = layers.strides.copy()
        self.padding = layers.padding.copy()
        self.out_shapes = layers.out_shapes.copy()
        self.pool_size = layers.pool_size.copy()
        self.numlayer = 0
        self.ffn_counter = 0
        self.conv_counter = 0
        self.residual_counter = 0
        self.pool_counter = 0
        self.concat_counter = 0
        self.tile_counter = 0
        self.activation_counter = 0
        self.specLB = layers.specLB.copy()
        self.specUB = layers.specUB.copy()
        self.original = layers.original.copy()
        self.zonotope = layers.zonotope.copy()
        self.predecessors = layers.predecessors.copy()
        self.in_LB = layers.in_LB.copy()
        self.in_UB = layers.in_UB.copy()
        self.last_layer = layers.last_layer   


    def calc_layerno(self):
        return self.ffn_counter + self.conv_counter + self.residual_counter + self.pool_counter + self.activation_counter + self.concat_counter + self.tile_counter


    def is_ffn(self):
        return not any(x in ['Conv2D', 'Conv2DNoReLU', 'Resadd', 'Resaddnorelu'] for x in self.layertypes)

    
    def check_layer_types(self):
        print('layer types:', self.layertypes)
    
    def set_last_weights(self, constraints):
        length = 0.0       
        last_weights = [0 for weights in self.weights[-1][0]]
        for or_list in constraints:
            for (i, j, cons) in or_list:
                if j == -1:
                    last_weights = [l + w_i + float(cons) for l,w_i in zip(last_weights, self.weights[-1][i])]
                else:
                    last_weights = [l + w_i + w_j + float(cons) for l,w_i, w_j in zip(last_weights, self.weights[-1][i], self.weights[-1][j])]
                length += 1
        self.last_weights = [w/length for w in last_weights]


    def back_propagate_gradiant(self, nlb, nub):
        #assert self.is_ffn(), 'only supported for FFN'

        grad_lower = self.last_weights.copy()
        grad_upper = self.last_weights.copy()
        last_layer_size = len(grad_lower)
        for layer in range(len(self.weights)-2, -1, -1):
            weights = self.weights[layer]
            lb = nlb[layer]
            ub = nub[layer]
            layer_size = len(weights[0])
            grad_l = [0] * layer_size
            grad_u = [0] * layer_size

            for j in range(last_layer_size):

                if ub[j] <= 0:
                    grad_lower[j], grad_upper[j] = 0, 0

                elif lb[j] <= 0:
                    grad_upper[j] = grad_upper[j] if grad_upper[j] > 0 else 0
                    grad_lower[j] = grad_lower[j] if grad_lower[j] < 0 else 0

                for i in range(layer_size):
                    if weights[j][i] >= 0:
                        grad_l[i] += weights[j][i] * grad_lower[j]
                        grad_u[i] += weights[j][i] * grad_upper[j]
                    else:
                        grad_l[i] += weights[j][i] * grad_upper[j]
                        grad_u[i] += weights[j][i] * grad_lower[j]
            last_layer_size = layer_size
            grad_lower = grad_l
            grad_upper = grad_u
        return grad_lower, grad_upper
