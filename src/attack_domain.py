'''
@author: Li Jiaying
'''

import numpy as np
from numpy.lib.function_base import disp
from scipy.optimize.optimize import OptimizeResult
from scipy.sparse.linalg import dsolve
# from fppoly import *
from elina_interval import *
from elina_abstract0 import *
from elina_manager import *
from analyzer import *
from executor import *
import scipy.optimize as op
import tensorflow.compat.v1 as tf 
tf.disable_v2_behavior()
from network_property import *
from attack_result import *
from colors import *


def robust_distance(output, target):
    # we would like to make the label be not the target
    # so, the other output should be greater than the target output
    # "output[target]-max(output[other])": the smaller the better
    # if it is less than 0, attack succeed.
    output = np.reshape(output, -1)
    target_value = output[target]
    output[target] = np.nan
    other_value = np.nanmax(output)
    output[target] = target_value # this leads to the critical bug
    #print("Target", target, "Target value:", target_value, "Other value:", other_value)
    return target_value - other_value


def print_optimize_result(opt_res, enable=False):
    if enable:
        print(' '*10, 'scipy optimize result---------------------')
        #print(' '*10, '    msg:', opt_res.message.decode('utf-8'))
        print(' '*10, '    msg:', opt_res.message)
        print(' '*10, '    nfev times:', opt_res.nfev)
        print(' '*10, '    x:', list(np.reshape(opt_res.x, (-1))))
        print(' '*10, '    fun:', opt_res.fun)

def check_in_bound(x, bound_spec):
    lb, ub = np.array(bound_spec[0]), np.array(bound_spec[1])
    x = x.flatten()
    assert x.shape == lb.shape and x.shape == ub.shape
    return ((x>=lb)&(x<=ub)).all()

class AbstractDomainAttack:
    def __init__(self, executor):
        self.ir_list = executor.ir_list
        self.exe = executor
        self.distances = []
        #self.sess = executor.sess
        self.property = None
        self.pgd_cache = {'xent':{}, 'cw':{}}


    def target_at_property(self, p): # p is of NNProperty
        assert p.type == 1 # 'robustness'
        self.property = p
        # we would like to find examples with label != target
        # in other words, output_vector[label] is not the maximum
        f1 = lambda x: robust_distance(x, p.target)
        self.distances.append(f1)


    # ---------------------------------------------------
    # | attack_mode      | arguments
    # ---------------------------------------------------
    # | bound_attack     | [lb, ub]
    # | domain_attack    | [domain_name, domain_manager, domain_element] # this is virtual...
    # | generator_attack | [variables, generate_matrix, genearate_bias]
    # ---------------------------------------------------
    def configure(self, start, domain, attack_mode='bound', args=None, attack_method='pgd', dprint=False):
        self.dprint = dprint
        # if dprint: 
        #     print(' '*10, 'attack configuration')
        #     print(' '*10, '----- attacking: start from ir', start, ' (ir1 is the input layer)-----')
        #     print(' '*10, ' attack_mode:[', attack_mode, ']  attack_method:[', attack_method, ']', sep='')
        assert attack_method in ['scipy', 'pgd', 'sgd']
        assert attack_mode in ['bound', 'domain', 'generator']
        self.attack_mode=attack_mode
        self.attack_method=attack_method
        self.domain = domain

        if attack_mode=='bound':
            # self.lb, self.ub = args
            self.lb = np.reshape(args[0], (-1, 1))
            self.ub = np.reshape(args[1], (-1, 1))
            self.seed = np.random.uniform(self.lb, self.ub, self.lb.shape)
            self.bound_spec = tuple(zip(self.lb, self.ub))
            # if self.dprint:
                # print('use bounds to launch attack')
                # print('lb:', *self.lb, '...')
                # print('ub:', *self.ub, '...')
        elif attack_mode=='domain':
            self.domain_name, self.domain_manager, self.domain_element = args
            # if self.dprint:
                # print('use abstract_domain_element to launch attack')
                # elina_abstract0_simple_print(self.domain_manager, self.domain_element)
        elif attack_mode=='generator':
            if domain == "deepzono":
                self.vars, self.matrix, self.bias, es, elb, eub, xlb, xub = args
                ## WE NEED TO UPDATE LB, UB TO FIX ISSUE #19.
                ## in principle, we should extract the scope of each ε from the given zonotope...
                lb = np.array([-1.0]*len(self.vars))
                ub = np.array([1.0]*len(self.vars))
                for i, e in enumerate(es):
                    lb[es[i]] = elb[i]
                    ub[es[i]] = eub[i]
            elif domain == "deeppoly":
                self.matrix, self.bias, xlb, xub = args
                lb, ub = xlb, xub
                
            # if len(es)>0:
            #     print('******lb,ub changes******')
            #     print(' lb:', lb)
            #     print(' ub:', ub)
            self.lb = np.reshape(lb, (-1, 1))
            self.ub = np.reshape(ub, (-1, 1))
            self.seed = np.random.uniform(self.lb, self.ub, self.lb.shape)
            self.bound_spec = tuple(zip(self.lb, self.ub))
            self.xbound_spec = [xlb, xub]
            # if self.dprint:
                # print('use generator to launch attack')
                # print('generator matrix:', self.matrix)

        self.start_layer = start
        self.attack_mode = attack_mode
        self.attack_method = attack_method
        self.args = args
        self.mask = self.property.get_mask()
        self.xmask = self.property.get_xmask()
        self.nr_remaining_layers = len(self.exe.ir_list) - self.start_layer
        


    #========================================================================================
    def scipy_bound_attack(self):
        def obj_func(x):
            x = np.reshape(x, (-1,1))
            output = self.Y.eval(feed_dict={self.X:x})
            res = self.distances[0](output)
            # for dist in self.distances:
                # res += dist(output)
            return res

        spec = tuple(zip(self.lb, self.ub))
        seed = np.random.uniform(lb, ub, self.lb.shape)
        opt_res = op.minimize(obj_func, np.array(seed), bounds=spec)
        if self.dprint: print_optimize_result(opt_res, opt_res.fun<0)

        y = model_output.eval(feed_dict={model_input:np.reshape(opt_res.x, (-1,1))})
        # validation
        assert opt_res.fun>=0 or np.argmax(y)!=self.property.target
        return res.x, y, res.fun, 0.0



    #========================================================================================
    def sgd_bound_attack(self): # like 'cw' mode in pgd attack        
        # label_mask = self.property.get_target_mask()
        this = tf.reduce_sum(self.Y * self.tf_mask)
        other = tf.reduce_max(self.Y * self.tf_xmask)
        # output_loss measures how far it is from current prediction to a target attack
        output_loss = tf.maximum(this-other, 0.0)
    
        x_perturb = tf.reduce_max(tf.maximum(lb-self.X, self.X-ub))
        input_loss = tf.maximum(x_perturb, 0.0)

        c = tf.placeholder(tf.float64, [])
        loss = output_loss + c * input_loss
        
        optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.002)
        train = optimizer.minimize(loss)

        init_op = tf.initialize_all_variables()
        self.sess.run(init_op)
        _, x, y, xloss, yloss = self.sess.run([train, self.X, self.Y, input_loss, output_loss], feed_dict={c:10.0})
        # validation
        assert yloss>=0 or np.argmax(y)!=self.property.target
        return x, y, yloss



    #========================================================================================
    def pgd_bound_attack(self, loss_func = 'cw'):
        assert loss_func in ['xent', 'cw']
        mult_attack_mode = False
        step_size = 0.01
        max_iter = 100
        max_trials = 5 + 2 * self.nr_remaining_layers
        # label_mask = self.property.get_target_mask()
        # tf_mask = tf.reshape(self.tf_mask, (-1,1))

        if loss_func == 'cw':
            correct_logit = tf.reduce_sum(self.tf_mask * self.Y, axis=0)
            wrong_logit = tf.reduce_max(self.tf_xmask * self.Y
                                        - 1e4*self.tf_mask, axis=0)
            loss = -tf.nn.relu(correct_logit - wrong_logit + 50)
        else:
            if loss_func != 'xent':
                print('Unknown loss function. Defaulting to cross-entropy')
            y_xent = tf.nn.softmax_cross_entropy_with_logits(labels=self.tf_mask, logits=self.Y)
            loss = tf.reduce_sum(y_xent)

        model_grad = tf.gradients(loss, self.X)[0]
        label = tf.argmax(self.Y)
        # print('grad:', model_grad)
        
        if mult_attack_mode:
            xs = []
            # x_y_and_loss_set = []
        for trial in range(max_trials):
            x = np.random.uniform(self.lb, self.ub, self.lb.shape) # seed
            for i in range(max_iter):
                grad, y, lab, lss = self.sess.run([model_grad, self.Y, label, loss], feed_dict={self.X:x})
                # in this case, we got an adversary example. We store the value and end this attack
                if lab != self.property.target: 
                    print('!@'*20, ' From random seed iteration:', trial, ' gradient step:', i, ' label:', lab, ' loss:', lss)
                    if mult_attack_mode: 
                        xs.append(x)
                    else: 
                        assert lss>=0 or np.argmax(y)!=self.property.target
                        return x, y, lss, grad
                    break # end of this attack
                x += step_size * np.sign(grad)
                # x += step_size * grad
                x = np.clip(x, self.lb, self.ub)
        if mult_attack_mode:
            return xs
        else:
            return None
     
        
        
    #========================================================================================
    def scipy_domain_attack(self):
        assert False, '"scipy_domain_attack" is not avaiable in AbstractDomainAttack. Try "scipy_bound_attack" or "scipy_generator_attack" instead.'
    
    
    
   #========================================================================================
    def pgd_domain_attack(self, loss_func = 'cw'):
        assert False, '"pgd_domain_attack" is not avaiable in AbstractDomainAttack. Try "pgd_bound_attack" or "pgd_generator_attack" instead.'
    
    
    
   #========================================================================================
    def sgd_domain_attack(self, loss_func = 'cw'):
        assert False, '"sgd_domain_attack" is not avaiable in AbstractDomainAttack. Try "sgd_bound_attack" or "sgd_generator_attack" instead.'
    
    
    
   #========================================================================================
    def scipy_generator_attack(self):
        # print(' '*10, 'do scipy generator attack.......')
        found_violation = False
        n_iter = 0
        n_eval_in_last_iter = 0 
        np_matrix = np.array(self.matrix)
        np_bias = np.array(self.bias)
        
        def obj_func(gene):
            nonlocal n_iter, n_eval_in_last_iter, found_violation
            n_eval_in_last_iter += 1
            # print(' -->input gene:', list(gene), end='')
            x = np.add(np.matmul(np_matrix, np.reshape(gene, (-1, 1))), np_bias)
            output = self.Y.eval(feed_dict={self.X:x})
            dist = self.distances[0](output)
            if (dist<0 or not check_in_bound(x, self.xbound_spec)):# print('Distance<0, so attack succeed.')
                found_violation = True
                # assert np.argmax(output)!=self.property.target
            return dist

        # callback(xk, OptimizeResult state) -> bool
        def early_stop_callback(gene):
            nonlocal n_iter, n_eval_in_last_iter, found_violation
            print(' '*12, '  [scipy_opt_info] Iteration:', n_iter, ' Evaluated times with iteration:', n_eval_in_last_iter, ' Found_violation:', found_violation)
            n_eval_in_last_iter=0
            n_iter += 1
            if found_violation:
                return True
            return False


        if not self.seed.any(): #No variable. The layer is all 0
            temp_x = np.add(np.matmul(np_matrix, np.reshape(self.seed, (-1, 1))), np_bias) #x = np.bias
            temp_y = self.Y.eval(feed_dict={self.X:temp_x})
            temp_dist = self.distances[0](temp_y)
            opt_res = OptimizeResult({"x":self.seed, "fun":temp_dist, "message":"CONVERGENCE:SEED IS EMPTY", "nfev":1})
        else:
            opt_res = op.minimize(obj_func, self.seed, bounds=self.bound_spec, callback=early_stop_callback)
        
        gene = opt_res.x
        x = np.add(np.matmul(np_matrix, np.reshape(gene, (-1, 1))), np_bias)
        opt_res.x = x
        y = self.Y.eval(feed_dict={self.X:x})
        label = np.argmax(y)
        if self.dprint:
            print_optimize_result(opt_res, opt_res.fun<0)
            if opt_res.fun<0:
                print(' '*10, '    ε:', list(gene))
                print(' '*10, '    y:', list(np.reshape(y, (-1))))
                print(' '*10, '    label:', label)
        # validation
        assert opt_res.fun>=0 or np.argmax(y)!=self.property.target
        if not check_in_bound(x, self.xbound_spec):
            return opt_res.x, y, 1, 0.0, gene #x out of bound so attack failse
        return opt_res.x, y, opt_res.fun, 0.0, gene

    
    
   #========================================================================================
    def pgd_generator_attack(self, loss_func = 'cw'):
        assert loss_func in ['xent', 'cw']
        mult_attack_mode = False
        step_size = 2.0
        max_iter = 30
        max_trials = 5 + self.nr_remaining_layers//2
        #print("Shape gene:", self.lb.shape)
        #print("Lower bound:", self.xbound_spec[0])
        #print("Upper bound:", self.xbound_spec[1])
        model_grad, model_grad_x, X, Y, label, loss, tf_gene, tf_mask, tf_xmask = self.get_pgd_para_cache(loss_func)
        # if mult_attack_mode:
        #     xs = []
        #     # x_y_and_loss_set = []
        x_v = None
        y_v = -1
        loss_v = 99999.9
        sess = tf.Session()
        for trial in range(max_trials):
            gene = np.random.uniform(self.lb, self.ub, self.lb.shape) # seed
            for i in range(max_iter+trial*2):
                input_dict = {tf_gene:gene, tf_mask:self.mask, tf_xmask:self.xmask}
                grad_v, grad_x_v, x_v, y_v, label_v, loss_v = sess.run([model_grad, model_grad_x, X, Y, label, loss], feed_dict=input_dict)

                if not check_in_bound(x_v, self.xbound_spec):
                    x_v = None
                    y_v = -1
                    loss_v = 99999.9
                    break
                # in this case, we got an adversary example. We store the value and end this attack
                # attack succeeds.... when label_v != self.property.target: 
                if loss_v < 0:
                    print(' [trial:', trial, ' iteration:', i, '] ', sep='', end='')
                    if self.dprint:
                        print('  gene:', list(np.reshape(gene, (-1))))
                        print('  matrix:', self.matrix)
                        print('  bias:', self.bias)
                    # print(' '*10, ' @->@->@-> From random seed iteration:', trial, ' gradient step:', i, ' label:', label_v, ' loss:', loss_v)
                    # if mult_attack_mode: 
                    #     xs.append(x)
                    # else:
                    # if True:
                    assert loss_v>=0 or np.argmax(y_v)!=self.property.target
                    sess.close()
                    return x_v, y_v, loss_v, grad_x_v, gene
                    # break # end of this attack
                # gene += step_size * np.sign(grad_v)
                step_size = max((loss_v+0.1), 3.0)
                gene -= step_size * grad_v 
                gene = np.clip(gene, self.lb, self.ub)
        # if mult_attack_mode:
        #     return xs
        # else:
        sess.close()
        return x_v, y_v, loss_v, grad_x_v, gene
    
    
    
   #========================================================================================
    def sgd_generator_attack(self, loss_func = 'cw'):
        pass
    
    
    
   #========================================================================================
    functions = {'bound':{'scipy':scipy_bound_attack, 'pgd':pgd_bound_attack, 'sgd':sgd_bound_attack},
            'domain':{'scipy':scipy_domain_attack, 'pgd':pgd_domain_attack, 'sgd':sgd_domain_attack},
            'generator':{'scipy':scipy_generator_attack, 'pgd':pgd_generator_attack, 'sgd':sgd_generator_attack}
            }
    
    def pgd_update(self, gene, loss_func="cw"):
        tf_gene = tf.Variable(tf.zeros(self.lb.shape, tf.float64), name = "gene")
        tf_matrix = tf.convert_to_tensor(self.matrix, dtype=tf.float64)
        tf_bias = tf.convert_to_tensor(self.bias, dtype=tf.float64)
        X = tf.add(tf.matmul(tf_matrix, tf_gene), tf_bias)
        Y = self.exe.get_parametric_model(self.start_layer, X)
        
        if loss_func == 'cw':
            correct_logit = tf.reduce_sum(self.tf_mask * Y, axis=0)
            wrong_logit = tf.reduce_max(tf.boolean_mask(Y, self.tf_xmask), axis=0)
            loss = correct_logit - wrong_logit # 
        else:
            if loss_func != 'xent':
                print('Unknown loss function. Defaulting to cross-entropy')
            y_xent = tf.nn.softmax_cross_entropy_with_logits(labels=self.tf_mask, logits=Y)
            loss = tf.reduce_sum(y_xent)
            
        model_grad = tf.gradients(loss, tf_gene)[0]
        model_grad_x = tf.gradients(loss, X)[0]
        grad_v, grad_x_v, x_v, y_v, loss_v = self.sess.run([model_grad, model_grad_x, X, Y, loss], feed_dict={tf_gene:gene})
        ares = AttackResult()
        ares.success = True
        ares.x = np.reshape(x_v, (-1))
        ares.gene = gene
        ares.distance = loss_v
        ares.grad = list(np.reshape(grad_x_v, (-1)))
        return ares

    def get_attack_result(self, gene):
        ares = AttackResult()
        if self.attack_method == "scipy":
            self.X = self.exe.get_fresh_input(self.start_layer)
            self.Y = self.exe.get_concrete_model(self.start_layer)
            np_matrix = np.array(self.matrix)
            np_bias = np.array(self.bias)
            x = np.add(np.matmul(np_matrix, np.reshape(gene[:len(self.vars)], (-1, 1))), np_bias)
            y = self.Y.eval(feed_dict={self.X:x})
            ares.success = True
            ares.x = np.reshape(x, (-1))
            ares.gene = gene[np.array[self.vars]]
            ares.distance = self.distances[0](y)
            ares.grad = [0.0]
            return ares
        else:
            return self.pgd_update(gene[:len(self.vars)])
    
    def get_pgd_para_cache(self, loss_func):
        para = self.pgd_cache[loss_func].get(self.start_layer)
        if para and self.domain == "deeppoly":
            return para
        # rebuild the model
        # generating input X for layer start of the neural network
        tf_gene = tf.Variable(tf.zeros(self.lb.shape, tf.float64), trainable=True)
        tf_matrix = tf.convert_to_tensor(self.matrix, dtype=tf.float64)
        tf_bias = tf.convert_to_tensor(self.bias, dtype=tf.float64)
        tf_mask = tf.placeholder(tf.float64, shape=(self.property.nr_labels,1))
        tf_xmask = tf.placeholder(tf.bool, shape=self.property.nr_labels)
        X = tf.add(tf.matmul(tf_matrix, tf_gene), tf_bias)
        Y = self.exe.get_parametric_model(self.start_layer, X)
        
        if loss_func == 'cw':
            correct_logit = tf.reduce_sum(tf_mask * Y, axis=0)
            wrong_logit = tf.reduce_max(tf.boolean_mask(Y, tf_xmask), axis=0)
            loss = correct_logit - wrong_logit # 
        else:
            if loss_func != 'xent':
                print('Unknown loss function. Defaulting to cross-entropy')
            y_xent = tf.nn.softmax_cross_entropy_with_logits(labels=self.tf_mask, logits=Y)
            loss = tf.reduce_sum(y_xent)
            
        model_grad = tf.gradients(loss, tf_gene)[0]
        model_grad_x = tf.gradients(loss, X)[0]
        # print(' '*10, 'grad:', model_grad)
        label = tf.argmax(Y)
        new_para = (model_grad, model_grad_x, X, Y, label, loss, tf_gene, tf_mask, tf_xmask)
        if self.domain == "deeppoly":
            self.pgd_cache[loss_func][self.start_layer] = new_para
        return new_para

    
    def last_layer_attack(self):
        if self.attack_mode == 'generator':
            np_matrix = np.array(self.matrix)
            np_bias = np.array(self.bias)
        nr_samples = 5
        x_len = np_bias.shape[0]
        # nr_samples = len(self.lb)*4
        for i in range(nr_samples):
            gene = np.random.uniform(self.lb, self.ub, self.lb.shape)
            if self.attack_mode == 'generator':
                x = np.add(np.matmul(np_matrix, np.reshape(gene, (-1, 1))), np_bias)
            label = np.argmax(x)
            if label != self.property.target:
                return x, x, x[self.property.target][0] - x[label][0], [1.0]*x_len, gene
        return x, x, x[self.property.target][0], 0.0, gene
                
                
        
    
    # start=1  <==>  start=2
    # this is because from start1 to start2, we do not have non-linear transformation
    # therefore, the output vector should be the same
    # but there is no guarantee that the optimisation will result in the same output   
    def attack(self):
        # return x in 1d np.array format, e.g. [784], if attack succeed, None otherwise
        # note: the shape of models are in column vector, e.g. [784, 1]
        #       we need to take care of the format very carefully.
        #       This is due to the design of the original deeppoly IR format.
        # print('in attack: attack target is', self.property.target)
        ares = AttackResult()
        if self.nr_remaining_layers == 0:
            ## sample points in the domain, and check the label directly.
            x, y, dist, grad, gene = self.last_layer_attack()
        else:
            self.X = self.exe.get_fresh_input(self.start_layer)
            self.Y = self.exe.get_concrete_model(self.start_layer)
            
            real_attack_function = self.functions[self.attack_mode][self.attack_method]
            # print('real attack function: ', real_attack_function)
            x, y, dist, grad, gene = real_attack_function(self)
            
        x = np.reshape(x, (-1))
        label = np.argmax(y)
        
        if dist >= 0:
            print('[x] No counterexample found. Continue.', end='')
        else:
            print(red, bold, '[√] Counterexample(s) found. Refine.', nonbold, reset, end='')            
            ares.success = True
            ares.x = x
            ares.distance = dist
            ares.grad = list(np.reshape(grad, (-1)))
            ares.gene = gene

        if self.dprint and dist < 0:
            np.set_printoptions(precision=4, linewidth=200)
            print('\n', ' '*12, ' -->x(0:20):', list(x)[:20], '...', end='')
            print('\n', ' '*12, ' -->y:', np.reshape(y, (-1)), end='')
            print('\n', ' '*12, ' -->label:', label, '(target label:', str(self.property.target) + ')', end='')
            print('\n', ' '*12, ' -->dist:', dist, end='')
            
        return ares