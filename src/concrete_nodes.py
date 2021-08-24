"""
@author: Li Jiaying
"""

import numpy as np
from functools import reduce
from config import config
import tensorflow.compat.v1 as tf 
tf.disable_v2_behavior() 


last_output_length=0

def numel(x):
    return reduce((lambda x,y: x*y), x.shape)

def add_input_output_information(self, input_names, input_shapes, output_name, output_shape):
    """
    sets for an object the three fields:
        - self.output_length
        - self.input_names
        - self.output_name
    which will mainly be used by the Optimizer, but can also be used by the Nodes itself
    
    Arguments
    ---------
    self : Object
        will be a ConcreteNode, but could be any object
    input_names : iterable
        iterable of strings, each one being the name of another Concrete-Node
    output_name : str
        name of self
    output_shape : iterable
        iterable of ints with the shape of the output of this node
        
    Return
    ------
    None 
    """
    # global last_output_length
    assert len(input_shapes)<=1
    self.input_shapes = input_shapes
    self.output_shape = output_shape
    input_lengths = [reduce((lambda x,y: x*y), input_shape, 1) for input_shape in input_shapes]
    self.input_length = reduce((lambda x, y: x+y), input_lengths, 0)
    # self.input_length = reduce((lambda x, y: x+y), [[reduce((lambda x,y: x*y), input_shape) for input_shape in input_shapes]])
    self.output_length = reduce((lambda x, y: x*y), output_shape)
    self.input_names   = input_names
    self.output_name   = output_name
    # print('->input:', end=' ')
    # for name, shape in zip(input_names, input_shapes):
    #     print(shape, name, end=' ')
    # print(' length:', self.input_length)
    # print('<-ouput:', output_shape, output_name)
    input_names = [input_name.split('/')[1].split(':')[0].replace('_', '') for input_name in input_names]
    # print('input_names: ', input_names)
    xname = 'input_' + reduce((lambda x, y: x+'_'+y), input_names, 'for')
    self.x = tf.Variable(tf.zeros([self.input_length, 1], tf.float64), trainable=True, name=xname)
    # print(self.x)
    # print()


def get_bounds(man, element, nlb, nub, num_vars, start_offset, is_refine_layer = False):
    dimension = elina_abstract0_dimension(man, element)
    var_in_element = dimension.intdim + dimension.realdim
    bounds = elina_abstract0_to_box(man, element)
    itv = [bounds[i] for i in range(start_offset, num_vars+start_offset)]
    lbi = [x.contents.inf.contents.val.dbl for x in itv]
    ubi = [x.contents.sup.contents.val.dbl for x in itv]
    elina_interval_array_free(bounds, var_in_element)
    if is_refine_layer:
        nlb.append(lbi)
        nub.append(ubi)
    else:
        return lbi, ubi


class ConcreteInput:
    def __init__(self, input_names, input_shapes, output_name, output_shape):
        """
        Arguments
        ---------
        specLB : numpy.ndarray
            1D array with the lower bound of the input spec
        specUB : numpy.ndarray
            1D array with the upper bound of the input spec
        output_name : str
            name of this node's output
        output_shape : iterable
            iterable of ints with the shape of the output of this node
        """
        # add_input_output_information(self, input_names, input_shapes, [0], output_name, output_shape)
        add_input_output_information(self, input_names, input_shapes, output_name, output_shape)
        # self.spec = x

    # def get_spec(self):
    #     return self.spec

    def get_fresh_input(self):
        # xname = 'input_' + reduce((lambda x, y: x+'_'+y), self.input_names, 'for')
        # self.x = tf.Variable(tf.zeros([self.input_length], tf.float64), trainable=True, name=xname)
        return None
        # return self.x

    def transformer(self, sess, x=None):
        """
        creates an abstract element from the input spec
        """
        # return tf.Variable(tf.zeros([output_shape], tf.float64), trainable=True, name='x')
        return self.x




class ConcreteMatmul:
    def __init__(self, matrix, input_names, input_shapes, output_name, output_shape):
        """
        Arguments
        ---------
        matrix : numpy.ndarray
            2D matrix for the matrix multiplication
        input_names : iterable
            iterable with the name of the vector for the matrix multiplication
        output_name : str
            name of this node's output
        output_shape : iterable
            iterable of ints with the shape of the output of this node
        """
        input_shape = [matrix.shape[1]]
        # add_input_output_information(self, input_names, input_shapes, input_shape, output_name, output_shape)
        add_input_output_information(self, input_names, input_shapes, output_name, output_shape)
        # self.w = tf.constant(matrix, dtype=tf.float64)
        self.w = tf.constant(matrix, dtype=tf.float64)
    
    def get_fresh_input(self):
        return self.x
    
    def transformer(self, sess, x):
        """
        transforms element with ffn_matmult_without_bias_zono
        """
        # if self.w.shape[1]!=x.shape[0]:
        #     x=tf.transpose(x)
        y = tf.matmal(self.w, x)
        return y





class ConcreteAdd:
    def __init__(self, bias, input_names, input_shapes, output_name, output_shape):
        """
        Arguments
        ---------
        bias : numpy.ndarray
            the values of the first addend
        input_names : iterable
            iterable with the name of the second addend
        output_name : str
            name of this node's output
        output_shape : iterable
            iterable of ints with the shape of the output of this node
        """
        # add_input_output_information(self, input_names, input_shapes, bias.shape, output_name, output_shape)
        add_input_output_information(self, input_names, input_shapes, output_name, output_shape)
        b = tf.constant(bias, dtype=tf.float64)
        self.b = tf.reshape(b, [numel(b), 1])

    def get_fresh_input(self):
        return self.x
    
    def transformer(self, sess, x, testing):
        y = x + self.b
        return y


class ConcreteSub:
    def __init__(self, bias, input_names, input_shapes, output_name, output_shape):
        """
        Arguments
        ---------
        bias : numpy.ndarray
            the values of the first addend
        input_names : iterable
            iterable with the name of the second addend
        output_name : str
            name of this node's output
        output_shape : iterable
            iterable of ints with the shape of the output of this node
        """
        add_input_output_information(self, input_names, input_shapes, output_name, output_shape)
        b = tf.constant(bias, dtype=tf.float64)
        self.b = tf.reshape(b, [numel(b), 1])

    def get_fresh_input(self):
        return self.x

    def transformer(self, sess, x):
        y = x - self.b
        return y


class ConcreteMul:
    def __init__(self, matrix, input_names, input_shapes, output_name, output_shape):
        """
        Arguments
        ---------
        bias : numpy.ndarray
            the values of the first addend
        input_names : iterable
            iterable with the name of the second addend
        output_name : str
            name of this node's output
        output_shape : iterable
            iterable of ints with the shape of the output of this node
        """
        # add_input_output_information(self, input_names, input_shapes, bias.shape, output_name, output_shape)
        add_input_output_information(self, input_names, input_shapes, output_name, output_shape)
        self.w = tf.constant(matrix, dtype=tf.float64)

    def get_fresh_input(self):
        return self.x

    def transformer(self, sess, x):
        # if self.w.shape[1] != x.shape[0]:
        #     x=tf.transpose(x)
        x = tf.reshape(x, [numel(x), 1])
        assert self.w.shape[1] == x.shape[0]
        y = tf.matmul(self.w, x)
        return y




class ConcreteAffine(ConcreteMatmul):
    def __init__(self, matrix, bias, input_names, input_shapes, output_name, output_shape):
        """
        Arguments
        ---------
        matrix : numpy.ndarray
            2D matrix for the matrix multiplication
        bias : numpy.ndarray
            the values of the bias
        input_names : iterable
            iterable with the name of the other addend of the addition
        output_name : str
            name of this node's output
        output_shape : iterable
            iterable of ints with the shape of the output of this node
        """
        # add_input_output_information(self, input_names, input_shapes, [matrix.shape[1]], output_name, output_shape)
        add_input_output_information(self, input_names, input_shapes, output_name, output_shape)
        self.w = tf.constant(matrix, dtype=tf.float64)
        # self.b = tf.constant(bias, dtype=tf.float64)
        b = tf.constant(bias, dtype=tf.float64)
        self.b = tf.reshape(b, [numel(b), 1])

    def get_fresh_input(self):
        return self.x
    
    def transformer(self, sess, x):
        # x = tf.reshape(x, [numel(x), 1])
        # print('assume(input:', self.input_shapes, 'output:', self.output_shape, ')')
        # print('w:', self.w)
        # print('x:', x)
        # print('b:', self.b)
        # if self.w.shape[1] != x.shape[0]:
        #     x=tf.transpose(x)
        #     print('transpose(x):', x)
        # assert self.w.shape[1] == x.shape[0]
        # wx=self.w * x;
        # wx=tf.matmul(self.w, x)
        # if wx.shape != self.b.shape:
        #     self.b=tf.reshape(self.b, wx.shape)
            # wx=tf.reshape(wx, self.b.shape)
            # print('transpose(b):', self.b)
        # y = tf.nn.bias_add(wx, self.b)
        # y = tf.matadd(wx, self.b)
        y = tf.matmul(self.w, x) + self.b
        # y = wx + self.b
        # print('y:', y)
        # y = wx + b
        # print('w shape: ', self.w.shape)
        # print('x shape: ', x.shape)
        # print('wx shape: ', wx.shape)
        # print('b shape: ', self.b.shape)
        # if y.shape != self.output_shape:
        #     y=tf.transpose(y)
        #     print('transpose(y):', y)
        # assert self.output_shape==y.shape
        return y


class ConcreteConv:
    def __init__(self, image_shape, filters, strides, pad_top, pad_left, input_names, input_shapes, output_name, output_shape):
        """
        Arguments
        ---------
        image_shape : numpy.ndarray
            of shape [height, width, channels]
        filters : numpy.ndarray
            the 4D array with the filter weights
        strides : numpy.ndarray
            of shape [height, width]
        padding : str
            type of padding, either 'VALID' or 'SAME'
        input_names : iterable
            iterable with the name of the second addend
        output_name : str
            name of this node's output
        output_shape : iterable
            iterable of ints with the shape of the output of this node
        """
        # add_input_output_information(self, input_names, input_shapes, image_shape, output_name, output_shape)
        add_input_output_information(self, input_names, input_shapes, output_name, output_shape)
        self.image_size = tf.constant(image_shape, dtype=tf.float64)
        self.filters    = tf.constant(filters, dtype=tf.float64)
        self.strides    = tf.constant(strides, dtype=np.uintp)
        self.output_shape = (c_size_t * 3)(output_shape[1], output_shape[2], output_shape[3])
        self.pad_top    = pad_top
        self.pad_left   = pad_left

    def get_fresh_input(self):
        return self.x
    
    def transformer(self, sess, x):
        """
        transforms element with conv_matmult_zono, without bias
        """
        offset, old_length  = self.abstract_information
        element = conv_matmult_zono(*self.get_arguments(man, element))
        return remove_dimensions(man, element, offset, old_length)



class ConcreteConvbias(ConcreteConv):
    def __init__(self, image_shape, filters, bias, strides, pad_top, pad_left, input_names, input_shapes, output_name, output_shape):
        """
        Arguments
        ---------
        image_shape : numpy.ndarray
            of shape [height, width, channels]
        filters : numpy.ndarray
            the 4D array with the filter weights
        bias : numpy.ndarray
            array with the bias (has to have as many elements as the filter has out channels)
        strides : numpy.ndarray
            of shape [height, width]
        padding : str
            type of padding, either 'VALID' or 'SAME'
        input_names : iterable
            iterable with the name of the second addend
        output_name : str
            name of this node's output
        output_shape : iterable
            iterable of ints with the shape of the output of this node
        """
        ConcreteConv.__init__(self, image_shape, filters, strides, pad_top, pad_left, input_names, input_shapes, output_name, output_shape)
        # self.bias = tf.constant(bias, dtype=tf.float64)
        b = tf.constant(bias, dtype=tf.float64)
        self.b = tf.reshape(b, [numel(b), 1])
    
    
    def transformer(self, sess, x):
        """
        transforms element with conv_matmult_zono, with bias

        Arguments
        ---------
        man : ElinaManagerPtr
            man to which element belongs
        element : ElinaAbstract0Ptr
            abstract element onto which the transformer gets applied
        
        Return
        ------
        output : ElinaAbstract0Ptr
            abstract element after the transformer
        """
        offset, old_length  = self.abstract_information
        man, destructive, element, start_offset, filters, bias, input_size, expr_offset, filter_size, num_filters, strides, out_size, pad_top, pad_left, has_bias = self.get_arguments(man, element)
        bias     = self.bias
        has_bias = True
        element = conv_matmult_zono(man, destructive, element, start_offset, filters, bias, input_size, expr_offset, filter_size, num_filters, strides, out_size, pad_top, pad_left, has_bias)

        nn.last_layer='Conv2D'
        relu_groups.append([])
        get_bounds(man, element, nlb, nub, self.output_length, offset+old_length, is_refine_layer=True)
        if testing:
            return remove_dimensions(man, element, offset, old_length), nlb[-1], nub[-1]
        return remove_dimensions(man, element, offset, old_length)




class ConcreteNonlinearity:
    def __init__(self, input_names, input_shapes, output_name, output_shape):
        """
        Arguments
        ---------
        input_names : iterable
            iterable with the name of the vector you want to apply the non-linearity to
        output_name : str
            name of this node's output
        output_shape : iterable
            iterable of ints with the shape of the output of this node
        """
        # add_input_output_information(self, input_names, input_shapes, output_shape, output_name, output_shape)
        add_input_output_information(self, input_names, input_shapes, output_name, output_shape)

    def get_fresh_input(self):
        return self.x
    
    

class ConcreteRelu(ConcreteNonlinearity):
    def transformer(self, sess, x):
        return tf.nn.relu(x)

class ConcreteSigmoid(ConcreteNonlinearity):
    def transformer(self, sess, x):
        return tf.nn.sigmoid(x)

class ConcreteTanh(ConcreteNonlinearity):
    def transformer(self, sess, x):
        return tf.nn.tanh(x)


class ConcreteMaxpool:
    def __init__(self, image_shape, window_size, strides, pad_top, pad_left, input_names, input_shapes, output_name, output_shape):
        """
        Arguments
        ---------
        image_shape : numpy.ndarray
            1D array of shape [height, width, channels]
        window_size : numpy.ndarray
            1D array of shape [height, width] representing the window's size in these directions
        strides : numpy.ndarray
            1D array of shape [height, width] representing the stride in these directions
        padding : str
            type of padding, either 'VALID' or 'SAME'
        input_names : iterable
            iterable with the name of node output we apply maxpool on
        output_name : str
            name of this node's output
        output_shape : iterable
            iterable of ints with the shape of the output of this node
        """
        # add_input_output_information(self, input_names, input_shapes, image_shape, output_name, output_shape)
        add_input_output_information(self, input_names, input_shapes, output_name, output_shape)
        self.window_size = tf.constant(window_size, dtype=np.uintp)
        self.input_shape = tf.constant(image_shape, dtype=np.uintp)
        self.stride      = tf.constant(strides, dtype=np.uintp)
        self.pad_top     = pad_top
        self.pad_left    = pad_left
        self.output_shape = (c_size_t * 3)(output_shape[1], output_shape[2], output_shape[3])

    def get_fresh_input(self):
        return self.x
    
    def transformer(self, sess, x):
        """
        transforms element with maxpool_zono
        """
        offset, old_length = self.abstract_information
        h, w    = self.window_size
        H, W, C = self.input_shape
        element = maxpool_zono(man, True, element, (c_size_t * 3)(h,w,1), (c_size_t * 3)(H, W, C), 0, (c_size_t * 2)(self.stride[0], self.stride[1]), 3, offset+old_length, self.pad_top, self.pad_left, self.output_shape)

        if refine or testing:
            get_bounds(man, element, nlb, nub, self.output_length, offset + old_length, is_refine_layer=True)
        nn.maxpool_counter += 1

        relu_groups.append([])
        element = remove_dimensions(man, element, offset, old_length)
        if testing:
            return element, nlb[-1], nub[-1]
        return element




class ConcreteDuplicate:
    def __init__(self, src_offset, num_var):
        """
        Arguments
        ---------
        src_offset : int
            the section that need to be copied starts at src_offset
        num_var : int
            how many dimensions should be copied
        """
        self.src_offset = src_offset
        self.num_var    = num_var
        
        
    def transformer(self, sess, x):
        """
        adds self.num_var dimensions to element and then fills these dimensions with zono_copy_section
        """
        dst_offset = elina_abstract0_dimension(man, element).realdim
        add_dimensions(man, element, dst_offset, self.num_var)
        zono_copy_section(man, element, dst_offset, self.src_offset, self.num_var)
        return element




class ConcreteResadd:
    def __init__(self, input_names, input_shapes, output_name, output_shape):
        """
        Arguments
        ---------
        input_names : iterable
            iterable with the names of the two nodes you want to add
        output_name : str
            name of this node's output
        output_shape : iterable
            iterable of ints with the shape of the output of this node
        """
        # add_input_output_information(self, input_names, input_shapes, output_name, output_shape)
        add_input_output_information(self, input_names, input_shapes, output_name, output_shape)

    def get_fresh_input(self):
        return self.x
        
    
    def transformer(self, sess, x):
        """
        uses zono_add to add two sections from element together and removes the section that is defined by self.abstract_information[2]
        the result of the addition is stored in the section defined by self.abstract_information[:2]
        """
        dst_offset, num_var = self.abstract_information[:2]
        src_offset = self.abstract_information[2]
        zono_add(man, element, dst_offset, src_offset, num_var)

        if refine or testing:
            get_bounds(man, element, nlb, nub, self.output_length, dst_offset, is_refine_layer=True)
            relu_groups.append([])

        nn.residual_counter += 1

        if dst_offset != src_offset:
            element = remove_dimensions(man, element, src_offset, num_var)

        if testing:
            return element, nlb[-1], nub[-1]
        else:
            return element


class ConcreteGather:
    def __init__(self, indexes, input_names, input_shapes, output_name, output_shape):
        """
        collects the information needed for the handle_gather_layer transformer and brings it into the required shape

        Arguments
        ---------
        indexes : numpy.ndarray
            array of ints representing the entries of the of the input that are passed to the next layer
        """
        # add_input_output_information(self, [input_names[0]], output_name, output_shape)
        add_input_output_information(self, input_names, input_shapes, output_name, output_shape)
        self.indexes = tf.constant(indexes, dtype=np.uintp)

    def get_fresh_input(self):
        return self.x

    def transformer(self, sess, x):
        handle_gather_layer(man, True, element, self.indexes)
        return element
