'''
@author: Adrian Hoffmann
@update: Li Jiaying
'''


from concrete_nodes import *
from zonotope_nodes import *
from poly_nodes import *
from layers import *
from functools import reduce
import numpy as np


# TODO replace all strings with constants

class Optimizer:
    def __init__(self, operations, resources):
        """
        Arguments
        ---------
        operations : list
            list of dicts, each dict contains a mapping from a domain (like deepzono, refinezono or deeppoly) to a tuple with resources (like matrices, biases ...)
        resources : list
            list of str, each one being a type of operation (like "MatMul", "Conv2D", "Add" ...)
        """
        self.operations = operations
        self.resources  = resources
    

    ##############################################################################################################################
    ######################################### Concrete ###########################################################################
    ##############################################################################################################################
    def get_concrete(self):
        output = []
        output_info = []
        domain = 'deepzono'
        nbr_op = len(self.operations)
        
        i = 0
        while i < nbr_op:
            if self.operations[i] == "Placeholder":
                # m_input_names, m_input_shapes, output_name, b_output_shape = self.resources[i]
                # in_len = len(input_names)
                # X = tf.Variable(tf.zeros([in_len], tf.float64), trainable=True, name = "X")
                # output.append(ConcreteInput(m_input_names, output_name, b_output_shape))
                # output.append(ConcreteInput(x, *self.resources[i]))
                output.append(ConcreteInput(*self.resources[i][domain]))
                i += 1
            elif self.operations[i] == "MatMul":
                matrix, m_input_names, m_input_shapes, _, _           = self.resources[i][domain]
                if i != nbr_op-1 and self.operations[i+1] in ["Add", "BiasAdd"]:
                    bias, _, _, output_name, b_output_shape = self.resources[i+1][domain]
                    output.append(ConcreteAffine(matrix, bias, m_input_names, m_input_shapes, output_name, b_output_shape))
                    i += 2
                else:
                    #self.resources[i].append(refine)
                    output.append(ConcreteMatmul(*self.resources[i][domain]))
                    i += 1
            elif self.operations[i] == "Gemm":
                matrix, bias, m_input_names, m_input_shapes, b_output_name, b_output_shape = self.resources[i][domain]
                output.append(ConcreteAffine(matrix, bias, m_input_names, m_input_shapes, b_output_name, b_output_shape))
                i += 1
            elif self.operations[i] == "Conv2D":
                if i != nbr_op-1 and self.operations[i+1] == "BiasAdd":
                    filters, image_shape, strides, pad_top, pad_left, c_input_names, c_input_shapes, _, _ = self.resources[i][domain]
                    bias, _, b_output_name, b_output_shape = self.resources[i+1][domain]
                    output.append(ConcreteConvbias(image_shape, filters, bias, strides, pad_top, pad_left, c_input_names, c_input_shapes, b_output_name, b_output_shape))
                    i += 2
                else:
                    filters, image_shape, strides, pad_top, pad_left, input_names, intput_shapes, output_name, output_shape = self.resources[i][domain]
                    output.append(ConcreteConv(image_shape, filters, strides, pad_top, pad_left, input_names, input_shapes, output_name, output_shape))
                    i += 1
            elif self.operations[i] == "Conv":
                filters, bias, image_shape, strides, pad_top, pad_left, c_input_names, c_input_shapes, output_name, b_output_shape = self.resources[i][domain]
                output.append(ConcreteConvbias(image_shape, filters, bias, strides, pad_top, pad_left, c_input_names, c_input_shapes, output_name, b_output_shape))
                i += 1
            elif self.operations[i] == "Add":
                #self.resources[i].append(refine)
                output.append(ConcreteAdd(*self.resources[i][domain]))
                i += 1
            elif self.operations[i] == "Sub":
                #self.resources[i].append(refine)
                output.append(ConcreteSub(*self.resources[i][domain]))
                i += 1
            elif self.operations[i] == "Mul":
                #self.resources[i].append(refine)
                output.append(ConcreteMul(*self.resources[i][domain]))
                i += 1
            elif self.operations[i] == "MaxPool":
                image_shape, window_size, strides, pad_top, pad_left, input_names, input_shapes, output_name, output_shape = self.resources[i][domain]
                output.append(ConcreteMaxpool(image_shape, window_size, strides, pad_top, pad_left, input_names, input_shapes, output_name, output_shape))
                i += 1
            elif self.operations[i] == "Resadd":
                #self.resources[i].append(refine)
                output.append(ConcreteResadd(*self.resources[i][domain]))
                i += 1
            elif self.operations[i] == "Relu":
                #self.resources[i].append(refine)
                output.append(ConcreteRelu(*self.resources[i][domain]))
                i += 1
            elif self.operations[i] == "Sigmoid":
                output.append(ConcreteSigmoid(*self.resources[i][domain]))
                i += 1
            elif self.operations[i] == "Tanh":
                output.append(ConcreteTanh(*self.resources[i][domain]))
                i += 1
            elif self.operations[i] == "Gather":
                image_shape, indexes, axis,  input_names, inpuy_shapes, output_name, output_shape = self.resources[i][domain]
                calculated_indexes = self.get_gather_indexes(image_shape, indexes, axis)
                output.append(ConcreteGather(calculated_indexes, input_names, input_shapes, output_name, output_shape))
                i += 1
            elif self.operations[i] == "Reshape":
                output.append(ConcreteGather(*self.resources[i][domain]))
                i += 1
            else:
                assert 0, "the optimizer for Concrete doesn't know of the operation type " + self.operations[i]

            # for testing, getting the corresponding layer in the tensorflow or onnx model
            output_info.append(self.resources[i-1][domain][-2:])
            # output_info.append(self.resources[i-1][-2:])
        return output, output_info




    ##############################################################################################################################
    ######################################### Deep Zono ##########################################################################
    ##############################################################################################################################
    def get_deepzono(self, nn, specLB, specUB = None):
        """
        This function will go through self.operations and self.resources and creates a list of Deepzono-Nodes which then can be run by an Analyzer object.
        It is assumed that self.resources[i]['deepzono'] holds the resources for the operation of type self.operations[i]                
        
        Arguments
        ---------
        spec_LB : numpy.ndarray
            1D array with the lower bound of the input spec
        spec_UB : numpy.ndarray
            1D array with the upper bound of the input spec
        
        Return
        ------
        output : list
            list of Deepzono-Nodes that can be run by an Analyzer object
        """     
        execute_list = []   
        output_info = []
        domain = 'deepzono'
        nbr_op = len(self.operations)
        if nn is None:
            nn = Layers()
        
        assert self.operations[0] == "Placeholder", "the optimizer for Deepzono cannot handle this network "
        input_names, _, output_name, output_shape = self.resources[0][domain]
        if specUB is None:
            execute_list.append(DeepzonoInputZonotope(specLB, input_names, output_name, output_shape))
        else:
            execute_list.append(DeepzonoInput(specLB, specUB, input_names, output_name, output_shape))
        output_info.append(self.resources[0][domain][-2:])

        self.get_abstract_element(nn, 1, execute_list, output_info, 'deepzono')

        use_dict = self.deepzono_get_dict(execute_list)
        self.set_predecessors(nn, execute_list)
        output   = self.deepzono_forward_pass(execute_list, use_dict)
        return execute_list, output_info


    def deepzono_get_dict(self, ir_list):
        """
        Returns a dict mapping output-names to the number of times that output is used by the nodes in the ir_list.
        This functions is a helper function for organizing the sections of an abstract elements when we have a ResNet or later an RNN.

        Arguments
        ---------
        ir_list : iterable
            list of Deepzono-Nodes

        Return
        ------
        use_dict : dict
            mapping from a name to the number of times that node-output is used
        """
        use_dict = {}
        for node in ir_list:
            for input_name in node.input_names:
                use_dict[input_name] += 1
            use_dict[node.output_name] = 0
        return use_dict


    def deepzono_forward_pass(self, ir_list, use_dict):
        """
        This function plans which Deepzono-Node-output occupies which section of an abstract element. 
        If a DeepzonoDuplicate-Node should be needed, then this function will add it.
        This is needed when we have a ResNet or later RNNs.

        Arguments
        ---------
        ir_list : list
            list of Nodes, where each node has the fields output_length, input_names, and output_name (see DeepzonoNodes.py for examples)
        use_dict : dict
            maps the output_name of each node in ir_list to the number of times the node's output will be used

        Return
        ------
        ir_list : list
            the ir_list with updated and potentionally added nodes
        """
        def get_index(active_abstracts, in_name, index_store):
            index = 0
            while True:
                index = index + active_abstracts[index:].index(in_name)
                if not index in index_store:
                    break
                index += 1
            return index

        active_abstracts = []
        abstract_length  = []

        i = 0
        while i < len(ir_list):
            node        = ir_list[i]
            index_store = []
            node.abstract_information = []
            for in_name in node.input_names:
                index  = get_index(active_abstracts, in_name, index_store)
                length = abstract_length[index]
                offset = reduce(lambda x,y: x+y, abstract_length[:index], 0)
                node.abstract_information += [offset, length]
                index_store.append(index)

            if len(index_store) != 0:
                active_abstracts[index_store[0]] = node.output_name
                abstract_length[index_store[0]]  = node.output_length
                for j in range(1,len(index_store)):
                    index = index_store[j]
                    del active_abstracts[index]
                    del abstract_length[index]
            else:
                active_abstracts.append(node.output_name)
                abstract_length.append(node.output_length)
                node.abstract_information = [0, node.output_length]
                
            i += 1
            if use_dict[node.output_name] > 1:
                for j in range(1, use_dict[node.output_name]):
                    ir_list.insert(i, DeepzonoDuplicate(node.abstract_information[0], node.output_length))
                    active_abstracts.append(node.output_name)
                    abstract_length.append(node.output_length)
                    i += 1

        return ir_list



    def set_predecessors(self, nn, output):
        output_index_store = {}
        index_o = 0
        for node in output:
            output_index_store[node.output_name] = index_o
            index_o += 1
        for node in output:
            predecessors = (c_size_t * len(node.input_names))()
            i = 0
            for input_name in node.input_names:
                predecessors[i] = output_index_store[input_name]
                i += 1
            node.predecessors = predecessors
            nn.predecessors.append(predecessors)



    def get_gather_indexes(self, input_shape, indexes, axis):
        size = np.prod(input_shape)
        base_indexes = np.arange(size).reshape(input_shape)
        return np.take(base_indexes, indexes, axis=axis)

##############################################################################################################################
######################################### Deep Poly ##########################################################################
##############################################################################################################################

    def get_deeppoly(self, nn, specLB, specUB, lexpr_weights=None, lexpr_cst=None, lexpr_dim=None, uexpr_weights=None, uexpr_cst=None, uexpr_dim=None, expr_size=0, spatial_constraints=None):
        """
        This function will go through self.operations and self.resources and create a list of Deeppoly-Nodes which then can be run by an Analyzer object.
        It is assumed that self.resources[i]['deeppoly'] holds the resources for an operation of type self.operations[i].
        self.operations should only contain a combination of the following 4 basic sequences:
            - Placholder         (only at the beginning)
                - MatMul -> Add -> Relu
                - Conv2D -> Add -> Relu    (not as last layer)
                - MaxPool/AveragePool         (only as intermediate layer)

        Arguments
        ---------
        specLB : numpy.ndarray
            1D array with the lower bound of the input spec
        specUB : numpy.ndarray
            1D array with the upper bound of the input spec

        Return
        ------
        execute_list : list
            list of Deeppoly-Nodes that can be run by an Analyzer object
        """
        execute_list = []
        output_info = []
        domain = 'deeppoly'
        assert self.operations[0] == "Placeholder", "the optimizer for Deeppoly cannot handle this network "
        input_names, _, output_name, output_shape = self.resources[0][domain]
        output_info.append(self.resources[0][domain][-2:])
        execute_list.append(DeeppolyInput(specLB, specUB, input_names, output_name, output_shape,
                                            lexpr_weights, lexpr_cst, lexpr_dim, uexpr_weights, uexpr_cst, uexpr_dim, expr_size, spatial_constraints))

        self.get_abstract_element(nn, 1, execute_list, output_info, 'deeppoly')
        self.set_predecessors(nn, execute_list)
        return execute_list, output_info
    

    def get_abstract_element(self, nn, i, execute_list, output_info, domain):
        assert domain == "deepzono" or domain == "deeppoly", "ERAN does not support" + domain + " abstraction"
        nbr_op = len(self.operations)
        while i < nbr_op:
            if self.operations[i] == "MatMul":
                nn.layertypes.append('FC')
                if i < nbr_op-1 and self.operations[i+1] in ["Add", "BiasAdd"]:
                    matrix,  m_input_names, _, _, _      = self.resources[i][domain]
                    bias, _, _, output_name, b_output_shape = self.resources[i+1][domain]
                    i += 2
                else:
                    #self.resources[i][domain].append(refine)
                    matrix, m_input_names , _, output_name , b_output_shape  = self.resources[i][domain]
                    
                    bias_length = reduce((lambda x, y: x*y), b_output_shape)
                    bias = nn.zeros(bias_length)

                    i += 1
                if domain == 'deepzono':
                    execute_list.append(DeepzonoAffine(matrix, bias, m_input_names, output_name, b_output_shape))
                elif domain == 'deeppoly':
                    execute_list.append(DeeppolyFCNode(matrix, bias, m_input_names, output_name, b_output_shape))
                nn.weights.append(matrix)
                nn.biases.append(bias)
                nn.numlayer+= 1
            elif self.operations[i] == "Gemm":
                matrix, bias, m_input_names, _, b_output_name, b_output_shape = self.resources[i][domain]

                nn.weights.append(matrix)
                nn.biases.append(bias)
                nn.layertypes.append('FC')
                nn.numlayer+= 1
                #print("Gemm Matrix ", matrix)
                if domain == 'deepzono':
                    execute_list.append(DeepzonoAffine(matrix, bias, m_input_names, b_output_name, b_output_shape))
                else:
                    execute_list.append(DeeppolyFCNode(matrix, bias, m_input_names, b_output_name, b_output_shape))
                i += 1
            
            elif self.operations[i] == "Conv2D":
                if i < nbr_op-1 and self.operations[i+1] == "BiasAdd":
                    filters, image_shape, strides, pad_top, pad_left, c_input_names, _, _, _ = self.resources[i][domain]
                    bias, _, _, b_output_name, b_output_shape = self.resources[i+1][domain]
                    i += 2
                else:
                    filters, image_shape, strides, pad_top, pad_left, c_input_names, _, b_output_name, b_output_shape = self.resources[i][domain]
                    bias_length = reduce((lambda x, y: x*y), output_shape)
                    bias = nn.zeros(bias_length)
                    i += 1
                nn.numfilters.append(filters.shape[3])
                nn.filter_size.append([filters.shape[0], filters.shape[1]])
                nn.input_shape.append([image_shape[0],image_shape[1],image_shape[2]])
                nn.strides.append([strides[0],strides[1]])
                nn.padding.append([pad_top, pad_left])
                nn.out_shapes.append(b_output_shape)
                nn.filters.append(filters)
                nn.biases.append(bias)
                nn.layertypes.append('Conv')
                if domain == 'deepzono':
                    execute_list.append(DeepzonoConvbias(image_shape, filters, bias, strides, pad_top, pad_left, c_input_names, b_output_name, b_output_shape))
                else:
                    execute_list.append(DeeppolyConv2dNode(filters, strides, pad_top, pad_left, bias, image_shape, c_input_names, b_output_name, b_output_shape))
                nn.numlayer+=1
            elif self.operations[i] == "Conv":
                filters, bias, image_shape, strides, pad_top, pad_left, c_input_names, _, output_name, b_output_shape = self.resources[i][domain]
                nn.numfilters.append(filters.shape[3])
                nn.filter_size.append([filters.shape[0], filters.shape[1]])
                nn.input_shape.append([image_shape[0],image_shape[1],image_shape[2]])
                nn.strides.append([strides[0],strides[1]])
                nn.out_shapes.append(b_output_shape)
                nn.padding.append([pad_top, pad_left])
                nn.filters.append(filters)

                nn.biases.append(bias)
                nn.layertypes.append('Conv')
                nn.numlayer+=1
                if domain == 'deepzono':
                    execute_list.append(DeepzonoConvbias(image_shape, filters, bias, strides, pad_top, pad_left, c_input_names, output_name, b_output_shape))
                else:
                    execute_list.append(DeeppolyConv2dNode(filters, strides, pad_top, pad_left, bias, image_shape, c_input_names, output_name, b_output_shape))
                i += 1    
            elif self.operations[i] == "Resadd":
                #self.resources[i][domain].append(refine)
                input_names, _, output_name, output_shape = self.resources[i][domain]
                if domain == 'deepzono':
                    execute_list.append(DeepzonoResadd(input_names, output_name, output_shape))
                else:
                    execute_list.append(DeeppolyResidualNode(input_names, output_name, output_shape))
                nn.layertypes.append('Resadd')
                nn.numlayer += 1
                i += 1
            #elif self.operations[i] == "Add":
                #self.resources[i][domain].append(refine)
           #     execute_list.append(DeepzonoAdd(*self.resources[i][domain]))
           #     nn.layertypes.append('Add')
           #     nn.numlayer += 1
           #     i += 1
            elif self.operations[i] == "Sub":
                #self.resources[i][domain].append(refine)
                bias, is_minuend, input_names, _, output_name, output_shape = self.resources[i][domain]
                if domain == 'deepzono':
                    execute_list.append(DeepzonoSub(bias, is_minuend, input_names, output_name, output_shape))
                else:
                    execute_list.append(DeeppolySubNode(bias, is_minuend, input_names, output_name, output_shape))
                nn.layertypes.append('Sub')
                nn.numlayer += 1
                i += 1
            elif self.operations[i] == "Mul":
                #self.resources[i][domain].append(refine)
                bias, input_names, _, output_name, output_shape = self.resources[i][domain]
                if domain == 'deepzono':
                    execute_list.append(DeepzonoMul(bias, input_names, output_name, output_shape))
                else:
                    execute_list.append(DeeppolyMulNode(bias, input_names, output_name, output_shape))
                nn.layertypes.append('Mul')
                nn.numlayer += 1
                i += 1
            elif self.operations[i] == "MaxPool" or self.operations[i] == "AveragePool" or self.operations[i] == "AvgPool":
                image_shape, window_size, strides, pad_top, pad_left, input_names, _, output_name, output_shape = self.resources[i][domain]
                nn.pool_size.append(window_size)
                nn.input_shape.append([image_shape[0],image_shape[1],image_shape[2]])
                nn.strides.append([strides[0],strides[1]])
                nn.out_shapes.append(output_shape)
                nn.padding.append([pad_top, pad_left])
                nn.numlayer+=1
                is_maxpool = (self.operations[i]=="MaxPool")
                if is_maxpool:
                    nn.layertypes.append('Maxpool')
                else:
                    nn.layertypes.append('Avgpool')
                if domain == 'deepzono':
                    execute_list.append(DeepzonoPool(image_shape, window_size, strides, pad_top, pad_left, input_names, output_name, output_shape, is_maxpool))
                else:
                    execute_list.append(DeeppolyPoolNode(image_shape, window_size, strides, pad_top, pad_left, input_names, output_name, output_shape, is_maxpool))
                i += 1
            elif self.operations[i] == "Relu":
                #self.resources[i][domain].append(refine)
                input_names, _, output_name, output_shape = self.resources[i][domain]
                nn.layertypes.append('ReLU')
                if domain == 'deepzono':
                    execute_list.append(DeepzonoRelu(input_names, output_name, output_shape))
                else:
                    execute_list.append(DeeppolyReluNode(input_names, output_name, output_shape))
                nn.numlayer += 1
                i += 1
            elif self.operations[i] == "Sign":
                input_names, _, output_name, output_shape = self.resources[i][domain]
                nn.layertypes.append('Sign')
                if domain == 'deeppoly':
                   execute_list.append(DeeppolySignNode(input_names, output_name, output_shape))
                nn.numlayer += 1
                i += 1
            elif self.operations[i] == "Sigmoid":
                input_names, _, output_name, output_shape = self.resources[i][domain]
                if domain == 'deepzono':
                    execute_list.append(DeepzonoSigmoid(input_names, output_name, output_shape))
                else:
                    execute_list.append(DeeppolySigmoidNode(input_names, output_name, output_shape))
                nn.layertypes.append('Sigmoid')
                nn.numlayer += 1
                i += 1
            elif self.operations[i] == "Tanh":
                input_names, _, output_name, output_shape = self.resources[i][domain]
                if domain == 'deepzono':
                    execute_list.append(DeepzonoTanh(input_names, output_name, output_shape))
                else:
                    execute_list.append(DeeppolyTanhNode(input_names, output_name, output_shape))
                nn.layertypes.append('Tanh')
                nn.numlayer += 1
                i += 1
            elif self.operations[i] == "Gather":
                image_shape, indexes, axis,  input_names, _, output_name, output_shape = self.resources[i][domain]
                calculated_indexes = self.get_gather_indexes(image_shape, indexes, axis)
                if domain == 'deepzono':
                    execute_list.append(DeepzonoGather(calculated_indexes, input_names, output_name, output_shape))
                else:
                    execute_list.append(DeeppolyGather(calculated_indexes, input_names, output_name, output_shape))
                nn.layertypes.append('Gather')
                nn.numlayer += 1
                i += 1
            elif self.operations[i] == "Reshape":
                indexes, input_names, _, output_name, output_shape = self.resources[i][domain]
                if domain == 'deepzono':
                    execute_list.append(DeepzonoGather(indexes, input_names, output_name, output_shape))
                else:
                    execute_list.append(DeeppolyGather(indexes, [input_names[0]], output_name, output_shape))
                nn.layertypes.append('Gather')
                nn.numlayer += 1
                i += 1
            elif self.operations[i] == "Concat":
                assert domain == "deeppoly", "Only DeepPoly currently supports concatenation"
                width, height, channels, input_names, _, output_name, output_shape = self.resources[i][domain]
                execute_list.append(DeeppolyConcat(width, height, channels, input_names, output_name, output_shape))
                nn.layertypes.append('Concat')
                nn.numlayer += 1
                i += 1
            elif self.operations[i] == "Tile":
                assert domain == "deeppoly", "Only DeepPoly currently supports tiling"
                repeats, input_names, _, output_name, output_shape = self.resources[i][domain]
                execute_list.append(DeeppolyTile(repeats, input_names, output_name, output_shape))
                nn.layertypes.append('Tile')
                nn.numlayer += 1
                i += 1
            else:
                assert 0, "the optimizer for" + domain + " doesn't know of the operation type " + self.operations[i]
            output_info.append(self.resources[i-1][domain][-2:])