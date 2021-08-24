'''
@author: Li Jiaying
'''
import numpy as np
# from elina_abstract0 import *
# from elina_manager import *

from colors import *
from zonotope_funcs import *
import sys



class AffineFormGenerator(object):
    def __init__(self):
        self.no = 0
        self.cst = 0.0
        self.coefs = []
        self.vars = []
        self.var_indices = []


    # ε: \u03B5
    def __str__(self):
        # ret = 'Generator for X[' + str(self.no) + ']: '
        ret = 'X[' + str(self.no) + '] := '
        for i in range(len(self.vars)):
            ret = ret + str(self.coefs[i]) + '*\u03B5[' + str(self.vars[i]) + '] + '
        ret = ret + str(self.cst)
        return ret


    def set_var_indices(self, all_vars):
        # print(' o- all variables:', all_vars)
        # print('    selected variables:', self.vars)
        self.var_indices = []
        for var in self.vars:
            i = all_vars.index(var)
            self.var_indices.append(i)
        # print('    selected indices:', self.var_indices)
    
    
    def eval(self, values):  # values are of all the variables
        res = self.cst
        for index, coef in zip(self.var_indices, self.coefs):
            # index = self.var_indices[var]
            res += coef * values[index]
        return res

        
        



class ZonotopeGenerator(object):
    def __init__(self, man):
        self.all_vars = [] # eplison
        self.funcs = []
        self.man = man
        self.z = None
        
    
    #def get_variables(self, z):
        # z is elina_abstact0_t*
        # self.z = z
        #noises = zonotope_noise_symbols(self.man, z)
        # print(' '*12, 'plain noise:', noises)
        #size = noises[0]
        #all_vars = noises[1:size]
        # print(' '*12, 'get noise_symbols: (size:', size-1, ') ', all_vars, sep='')
        #all_vars = sorted(all_vars)
        #return all_vars
    
    def get_variables(self, z):
        dimension = elina_abstract0_dimension(self.man, z)
        num_dims = dimension.intdim + dimension.realdim
        indexs = [0]
        for i in range(num_dims):
            affine = get_affine_form_for_dim(self.man, z, i)
            size = int(affine[0]-1)
            index = [int(affine[i+size]) for i in range(2, size + 2)]
            if index:
                indexs.append(max(index))
        max_size = max(indexs) + 1
        res = list(range(max_size))
        return res
    
    def test_new_introduced_vars_for_dim(self, coefs, vars, focused_vars=None, dprint=False):
        # X[dim] = ?*new_set
        # x0 = coef_0_0 * v0 + coef_0_1 * v1 + b0 * 1
        if focused_vars is None:
            focused_vars = vars
        res = dict()
        # print('X[', dim, ']=', sep='', end='')
        # print('x dim=', dim, ' <={', end='')
        for coef, var in zip(coefs, vars):
            if dprint:
                if (coef!=0.0):
                    if var in focused_vars:
                        cprint('[', var, ']', color='green', end=' ')
                    else:
                        cprint('[', var, ']', end=' ')
                    print(coef, end=' | ')
            # if (coef!=0.0):
                # print('+', coef, '*g', var, sep='', end='')
            if (coef!=0.0) and (var in focused_vars):
                res[var] = coef
        if dprint:
            print()
        return res 
        
        
    def get_involved_vars_for_dim(self, coefs, vars):
        res = []
        for coef, var in zip(coefs, vars):
            # if (coef!=0.0):
                # print('+', coef, '*g', var, sep='', end='')
            if (coef!=0.0):
                res.append(var)
        return res
    
    
    # def test_new_introduced_vars(self, matrix, last_vars, all_vars):
    #     for dim in range(len(matrix)):
    #         dim_res = self.test_new_introduced_vars_for_dim(matrix, dim, all_vars, new_vars)
    #         # print('[', dim, ']:', len(dim_res), sep='', end='')
    #         print(len(dim_res), sep='', end=' ')
    #         if len(dim_res)<=0: continue
    #         if len(dim_res)==1:
    #             pass 
    #             # print(dim_res.keys(), end='')
    #         else:
    #             print('@@Error! dim=', dim, 'has at least two new_introduced noise symbols:')
    #             # print(' ->', dim_res)
    #     #         print(' $$', dim_res.keys())
    #     print()
    #     return True
    
        
        
    # def get_generator_for_dim(self, z, dim):
    #     array = get_affine_form_for_dim(self.man, z, dim)
    #     size = int(array[0]-1)
    #     afg = AffineFormGenerator()
    #     afg.no = dim
    #     afg.cst = array[1]
    #     for i in range(2, size+2):
    #         val = array[i]
    #         var = int(array[size+i])
    #         afg.vars.append(var)
    #         afg.coefs.append(val)
    #     return afg
    
    
    # def get_generators(self, z, all_vars):
    #     dimension = elina_abstract0_dimension(self.man, z)
    #     dims = dimension.intdim + dimension.realdim
    #     ret = []
    #     # print(' '*12, 'dimension:', dims)
    #     for i in range(dims):
    #         gen = self.get_generator_for_dim(z, i)
    #         gen.set_var_indices(all_vars)
    #         # print(' '*12, ' X[', i, ']:= ', gen, sep='')
    #         ret.append(gen)
    #     return ret
        
        
    def get_affine_for_dim(self, z, dim):
        array = get_affine_form_for_dim(self.man, z, dim)
        size = int(array[0]-1)
        afg = AffineFormGenerator()
        afg.no = dim
        afg.cst = array[1]
        for i in range(2, size+2):
            val = array[i]
            var = int(array[size+i])
            afg.vars.append(var)
            afg.coefs.append(val)
        return afg
    
    
    def test_new_introduced_vars(self, matrix, last_vars, this_vars):
        new_vars = list(set(this_vars).difference(set(last_vars)))
        # print('@@last layer variables:', last_vars)
        # print('@@this layer variables:', this_vars)
        # print('@@new variables since last layer:', new_vars)
        if len(new_vars)<=0:
            # print('xxxx NO new introduced vars. It should be an affine layer.')
            return
        else:
            # cprint('√√√√ new introduced vars(size:', len(new_vars), '):', new_vars, color='lgreen', sep='')
            for dim in range(len(matrix)):
                coefs = matrix[dim]
                # new introduced epsilons for dim:
                new_epsilons = self.test_new_introduced_vars_for_dim(coefs, this_vars, new_vars)
                if len(new_epsilons)>=2: 
                    cprint('@@Error! dim=', dim, 'has n=', len(new_epsilons), 'new introduced noise symbols:', background=True, color='red', end='')
                    cprint(' symbols:', new_epsilons, background=True, color='green')
                    # for detail information to debug
                    # self.test_new_introduced_vars_for_dim(coefs, this_vars, new_vars, dprint=True)
                    # affine = self.get_affine_for_dim(self.z, dim)
                    # print('detailed generator', affine)
            return True
    
    
    # matrix, bias = [], []
    # matrix * all_vars + bias
    # |coef_0_0 coef_0_1 |     |v0|   |b0|   |x0|
    # |coef_1_0 coef_1_1 |  *  |v1| + |b1| = |x1|
    # |coef_2_0 coef_2_1 |            |b2|   |x2|
    ## dim: [3, 2] * [2, 1] + [3, 1] = [3, 1]
    ###############################################
    ## x0 = coef_0_0 * v0 + coef_0_1 * v1 + b0 * 1
    # |coef_0_0 coef_0_1 b0|     |v0|   |x0|
    # |coef_1_0 coef_1_1 b1|  *  |v1| = |x1|
    # |coef_2_0 coef_2_1 b2|     |01|   |x2|
    def get_matrix_generator(self, z):
        self.z = z
        vars = self.get_variables(z)
        dimension = elina_abstract0_dimension(self.man, z)
        num_dims = dimension.intdim + dimension.realdim
        num_vars = len(vars)
        matrix = [[0.0]*(num_vars+1) for i in range(num_dims)]
        for dim in range(num_dims):
            array = get_affine_form_for_dim(self.man, z, dim)
            size = int(array[0]-1)
            matrix[dim][-1] = array[1]
            for i in range(2, size+2):
                val = array[i]
                var = int(array[size+i])
                index = all_vars.index(var)
                matrix[dim][index] = val
        return matrix
    
    
    # matrix, bias = [], []
    # matrix * all_vars + bias
    # |coef_0_0 coef_0_1 |     |v0|   |b0|   |x0|
    # |coef_1_0 coef_1_1 |  *  |v1| + |b1| = |x1|
    # |coef_2_0 coef_2_1 |            |b2|   |x2|
    ## dim: [3, 2] * [2, 1] + [3, 1] = [3, 1]
    ###############################################
    ## x0 = coef_0_0 * v0 + coef_0_1 * v1 + b0 * 1
    def get_generator(self, z):
        self.z = z
        vars = self.get_variables(z)
        dimension = elina_abstract0_dimension(self.man, z)
        num_dims = dimension.intdim + dimension.realdim
        num_vars = len(vars)
        matrix = [[0.0]*(num_vars) for i in range(num_dims)]
        bias = [[0.0] for i in range(num_dims)]
        for dim in range(num_dims):
            array = get_affine_form_for_dim(self.man, z, dim)
            size = int(array[0]-1)
            bias[dim][0] = array[1]
            for i in range(2, size+2):
                val = array[i]
                var = int(array[size+i])
                index = vars.index(var)
                matrix[dim][index] = val
        # m = np.array(matrix)
        return vars, matrix, bias
    
    
    ## WE NEED TO UPDATE LB, UB TO FIX ISSUE #19.
    ## in principle, we should extract the scope of each ε from the given zonotope...
    def get_constrained_bounds(self, z, vars):
        #size, es, elb, eub = zono_constrained_spec(self.man, z)
        # reindex the epsilons, according to all the variables in z
        size = 0
        res_es = []
        res_elb = []
        res_eub = []
        for i in range(size):
            e = es[i]
            lb = elb[i]
            ub = eub[i]
            if e in vars:
                # we reindex the epsilons, the new index is the position in the matrix, as the position in vars
                new_index = vars.index(e) 
                res_es.append(new_index)
                res_elb.append(lb)
                res_eub.append(ub)
        return res_es, res_elb, res_eub
            
        # try:
        #     # the reversed is used to keep the remaining list is not effected even after element removal
        #     for i, e in reversed(list(enumerate(es))):
        #         if e in vars:
        #             # print('e:', e, 'new_index:', vars.index(e))
        #             es[i] = vars.index(e)
        #         else:
        #             # here we remove the constrained epsilons that are not involved in all_vars.
        #             # it is strange, but we do not know why this condition could happen.
        #             # print('---- remove', e, '----')
        #             es.pop(i)
        #             elb.pop(i)
        #             eub.pop(i)
        # except:
        #     print('all_vars:', vars)
        #     print('cons_eps:', es)
        #     elina_abstract0_simple_print(self.man, z)
        #     sys.exit(-1)
        #     # print('reindex_of_e:', es)
        # return es, elb, eub
        
    
    @staticmethod
    def py_generate(x, matrix, bias):
        # x: a list
        # matrix: a numpy matrix
        # output: a list.
        assert isinstance(x, list)
        assert isinstance(matrix, list)
        assert isinstance(bias, list)
        # print('-----------generate---------')
        # print('inputx:', x)
        mx = np.matmul(np.array(matrix), np.array(x))
        y = np.add(mx, np.array(bias))
        return list(np.reshape(y, (-1)))
    
    
    @staticmethod
    def np_generate(np_x, np_matrix, np_bias):
        # np_x: a numpy array
        # np_matrix: a numpy matrix
        # output: np_y: a numpy array
        assert isinstance(np_x, (np.ndarray, np.generic))
        assert isinstance(np_matrix, (np.ndarray, np.generic))
        assert isinstance(np_bias, (np.ndarray, np.generic))
        # print('-----------generate_np---------')
        # print('inputx:', list(np.reshape(np_x, (-1))))
        np_x = np.reshape(np_x, (-1, 1))
        np_mx = np.matmul(np_matrix, np_x)
        np_y = np.add(np_mx, np_bias)
        # print('outputy:', list(np.reshape(np_y, (-1))))
        return np_y
        
        
    @staticmethod
    def tf_generate(tf_x, tf_matrix, tf_bias):
        # tf_x: a tf tensor, array
        # matrix: a tf tensor, matrix
        # output: y: a tf tensor, array
        assert isinstance(x, (np.ndarray, np.generic))
        assert isinstance(np_matrix, (np.ndarray, np.generic))
        assert isinstance(np_bias, (np.ndarray, np.generic))
        # print('-----------generate_np---------')
        # print('inputx:', list(tf.reshape(tf_x, (-1))))
        tf_x = tf.reshape(tf_x, (-1, 1))
        tf_mx = tf.matmul(tf_matrix, tf_x)
        tf_y = tf.add(tf_mx, np_bias)
        # print('outputy:', list(np.reshape(tf_y, (-1))))
        return tf_y