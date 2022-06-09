'''
@author: Adrian Hoffmann
@update: Li Jiaying
'''

from types import new_class
import numpy as np
import gc
import tracemalloc
from random import *
from elina_abstract0 import *
from elina_manager import *

from concrete_nodes import *
from zonotope_nodes import *
from poly_nodes import *
from attack_domain import *
from causality import Causality

from layers import *
from task_manager import *
from zonotope_generator import *
from zonotope_funcs import *
from colors import *
from copy import *
from enum import Enum
from ai_milp import milp_callback, create_model
from gurobipy import *
import time

only_split_not_add_new_task = False
split_check_time = 0
label_len = 30

class Verified_Result(Enum):
    Safe = 1
    UnSafe = -1 
    Unknow = 0

class Analyzer:
    def __init__(self, ir_list, nn, domain, specnumber):
        """
        Arguments
        ---------
        ir_list: list
            list of Node-Objects (e.g. from DeepzonoNodes), first one must create an abstract element
        domain: str
            either 'deepzono'
        """
        self.domain = domain
        self.specnumber = specnumber
        self.ir_list = ir_list     
        self.refine = True if 'refine' in domain else False   
        self.concrete_executor = None
        self.abstract_attacker = None
        self.causality = None
        self.man = None
        self.property = None
        self.nr_classes = self.ir_list[-1].output_length
        self.target_label = -1
        self.task_manager = TaskManager()
        self.k = 1
        self.task_manager.setLimit(self.k)
        self.timeout_lp = config.timeout_lp
        self.timeout_milp = config.timeout_milp
        self.nn = nn
        self.relu_groups = []

        if domain == 'deepzono':
            self.man = zonoml_manager_alloc()
            self.is_greater = is_greater_zono
            #self.is_greater_or_equal = is_greater_or_equal_zono
            self.zono_gen = ZonotopeGenerator(self.man)
        elif domain == 'deeppoly' or domain == 'refinepoly':
            self.man = fppoly_manager_alloc()
            self.is_greater = is_greater
        else:
            assert 0

    
    
    def set_target_label(self, target_label):
        self.target_label = target_label
        
    
    def __del__(self):
        if self.man is not None:
            elina_manager_free(self.man)
        
        
    def equip_with_concrete_executor(self, exe):
        self.concrete_executor = exe
        
        
    def equip_with_target_property(self, prop):
        self.property = prop
    
    def equip_with_abstract_attacker(self, abstract_attacker):
        self.abstract_attacker = abstract_attacker
        self.abstract_attacker.target_at_property(self.property)

    def update_attack_result(self, start_layer, element, attack_result):
        attack_mode = 'generator'
        vars, matrix, bias = self.zono_gen.get_generator(element)
        es, elb, eub = self.zono_gen.get_constrained_bounds(element, vars)
        xlb, xub = zono_bounds(self.man, element) 
        args = [vars, matrix, bias, es, elb, eub, xlb, xub]

        self.abstract_attacker.configure(start_layer, attack_mode=attack_mode, args=args, attack_method=self.attack_method, dprint=debug_mode)

        return self.abstract_attacker.get_attack_result(attack_result.gene)

    def attack_abstract_domain(self, start_layer, element, bounds=None):
        if self.abstract_attacker is None:
            self.abstract_attacker = AbstractDomainAttack(self.concrete_executor)
            self.abstract_attacker.target_at_property(self.property)
        if self.causality is None:
            self.causality = Causality(self.concrete_executor, self.target_label)
            
        if debug_mode:
            print('target property:', self.property)
            # elina_abstract0_simple_print(self.man, element)
        print(' |-attack....', end='')

        assert self.domain in ['deepzono', 'deeppoly', 'refinepoly']
        attack_mode = 'generator'
        if self.domain == "deepzono":
            vars, matrix, bias = self.zono_gen.get_generator(element)
            # self.zono_gen.test_new_introduced_vars(matrix, old_vars, vars)
            es, elb, eub = self.zono_gen.get_constrained_bounds(element, vars)
            xlb, xub = zono_bounds(self.man, element) 
            # print(' '*12, 'generator matrix: ', matrix)
            args = [vars, matrix, bias, es, elb, eub, xlb, xub]
            
            # scipy attack is too slow, we should avoid it.
        else:
            vars = []
            if start_layer != 1:
                xlb, xub = poly_bound(self.man, element, start_layer-2)
            else:
                xlb, xub = self.ir_list[0].specLB, self.ir_list[0].specUB
            matrix, bias = np.eye(len(xlb), dtype=float), np.zeros((len(xlb), 1), dtype=float)
            args = [matrix, bias, xlb, xub]
        self.abstract_attacker.configure(start_layer, self.domain, attack_mode=attack_mode, args=args, attack_method=self.attack_method, dprint=debug_mode)
        res = self.abstract_attacker.attack()
        print()
        return res

        
        
        
        
    def get_first_k_dimensions_to_split(self, attack_result, k, strategy, neurons, lb=None, ub=None, is_relu=False):
        strategies = ['grad_and_scale', 'causality', 'lb_similar_with_ub_first', 'grad_larger_first', 'input_ub_larger_first', 'input_lb_smaller_first', 'input_interval_larger_first', 'output_smaller_first', 'output_smaller_first_only_when_neg']
        assert strategy in strategies
        x = attack_result.x
        grad = attack_result.grad
        causality = attack_result.causality
        dims = len(x)        
        
        all_indicies = range(dims)
        # for relu function, we only concern the dimension that span across '0'
        # since other dimensions are precise
        if is_relu:
            if lb == ub:
                print('!!lb=ub!!')
            zono_debug_info_print(self.man, None, '', lb, ub)
            all_indicies = [i for i in all_indicies if lb[i]<0 and ub[i]>0]
        else:
            all_indicies = [i for i in all_indicies if i not in neurons]
        print(' **cross_0_indicies:', all_indicies, reset)

        if strategy == 'grad_and_scale':
            print("GRAD AND SCALE")
            ranked_list = sorted(all_indicies, key=lambda s: abs(grad[s])*(ub[s]-lb[s]))
        elif strategy == 'lb_similar_with_ub_first':
            # idea: if lb=ub, then the interval is even span across 0.0. Then we split the interval can drop more area
            # this only works for relu, lb<0 and ub>0
            all_indicies = [i for i in all_indicies if ub[i]!=lb[i]]
            ranked_list = sorted(all_indicies, key=lambda s: (-lb[s]*ub[s])/((ub[s]-lb[s])**2))
        elif strategy == 'causality':
            print("CAUSALITY")
            ranked_list = sorted(all_indicies, key=lambda s: causality[s])
        elif strategy == 'grad_larger_first':
            ranked_list = sorted(all_indicies, key=lambda s: abs(grad[s]))
        elif strategy == 'output_smaller_first':
            ranked_list = sorted(all_indicies, key=lambda s: 1-abs(x[s]/(ub[s]-lb[s])) if ub[s]!=lb[s] else 0.0)
        elif strategy == 'output_smaller_first_only_when_neg':
            neg_output_indicies = [i for i in all_indicies if x[i]<0]   
            # print(' '*12, back_green, '-->neg_indices:', neg_output_indicies, reset)
            ranked_list = sorted(neg_output_indicies, key=lambda i: -x[i])
            # print(' '*12, back_green, '-->rank index list:', rank_index_list, reset)
            # print(' '*12, back_green, '-->ranked x values:', [x[i] for i in rank_index_list], reset)
        elif strategy == 'input_ub_larger_first':
            ranked_list = sorted(all_indicies, key=lambda i: ub[i])
        elif strategy == 'input_lb_smaller_first':
            ranked_list = sorted(all_indicies, key=lambda i: -lb[i])
        elif strategy == 'input_interval_larger_first':
            ranked_list = sorted(all_indicies, key=lambda i: (ub[i]-lb[i]))
        return ranked_list[:k]
        assert 0
        
        
        
        
        
        

    def refine_abstract_domain(self, attack_result, layer_no, element, parent_hid, neurons, relu_groups):
        if self.domain == "deepzono":
            nn = None
        else:
            element, nn = element[0], element[1]
        lb, ub = get_bound(self.man, element, self.domain, layer_no-2)
        if debug_mode:
            print(gray, bold, 'refine the last domain: ', nonbold, reset, end='')
            debug_info_print(self.man, element, self.domain, layer_no-2, 'to refine')

        if config.strategy == "causality":
            causal = self.causality.get_ie(layer_no, lb, ub)
            attack_result.causality = causal

        # print(' - on refine -')
        # ir_class = type(self.ir_list[layer]).__name__
        # is_affine = ('Affine' in ir_class)
        this_layer_op = self.ir_list[layer_no].op()
        is_affine = this_layer_op == 'Affine'
        is_activate = this_layer_op in ['Relu', 'Sigmoid', 'Tanh']
        is_relu = this_layer_op == 'Relu'
        is_sigmoid = this_layer_op == 'Sigmoid'
        is_tanh = this_layer_op == 'Tanh'
        
        x = attack_result.x
        x_grad = attack_result.grad
        refined_elements = []
        k = self.k
        if is_activate:
            if is_relu:
                print('   |-refine [operation:Relu, action: i)0-splitting first, ii) s-splitting]') # grad:', x_grad[:10], '...')
                # print('[RELU refine] (s-splitting) Can not refine the domain via splitting only one dimension due to no negative y coordinate.')
                # split_dims = self.get_first_k_dimensions_to_split(attack_result, k, 'grad_larger_first', lb, ub, is_relu=True)
                #split_dims = self.get_first_k_dimensions_to_split(attack_result, k, 'grad_and_scale', neurons, lb, ub, is_relu=True)
                split_dims = self.get_first_k_dimensions_to_split(attack_result, k, config.strategy, neurons, lb, ub, is_relu=True)
                if len(split_dims) == 0:
                    print('Relu approximation is already eliminated. Since attack succeeds, over-approximation must be happened in the last layer. Should back propagate!')
                    return False
                    # sys.exit(0)
                dim = split_dims[0]
                print('     pick up dim', dim, ' bound:[', lb[dim], ',', ub[dim], ']', sep='')
                split_values = [0.0]*k
                # split_values = [x[i] for i in split_dims]
                #refined_elements = zono_specified_split(self.man, element, k, split_dims, split_values)
            else: # elif is_tanh or is_sigmoid:
                print('   |-refine [operation:Tanh/Sigmoid, action: s-splitting].')
                #split_dims = self.get_first_k_dimensions_to_split(attack_result, k, 'output_smaller_first', neurons, lb, ub)
                split_dims = self.get_first_k_dimensions_to_split(attack_result, k, config.strategy, neurons, lb, ub)
                if len(split_dims) == 0:
                    return False
                for dim in split_dims:
                    neurons.add(dim)
                split_values = [(lb[i] + ub[i])/2 for i in split_dims]
                #refined_elements = zono_specified_split(self.man, element, k, split_dims, split_values)
            
            refined_elements = abstract_split(self.man, self.domain, element, self.ir_list, nn, k, split_dims, split_values)    
            print('      |-split the dim', split_dims, 'into', len(refined_elements), 'subtasks: ') # , end='')
            for i, e in enumerate(refined_elements):
                if self.domain == "deepzono":
                    new_nn = None
                else:
                    e, new_nn = e[0], e[1]
                self.task_manager.add_task(layer_no, e, new_nn, parent_hid, i, neurons.copy(), relu_groups.copy())
                print(yellow, '         |->', i, ')', self.task_manager.last_task, reset, sep='', end=' ') 
                if debug_mode:
                    debug_info_print(self.man, e, self.domain, layer_no-2, 'splitted', end='')
                    print(' --dim', dim, ' bound:[', lb[dim], ',', ub[dim], ']  ', sep='', end='')
                print()
        else: # non-activate layers...
            print('   |-refine abstract domain: [operation:Affine, action:ignore] FOUND approximation error. opt(x)=∅, while opt(affine(x))≠∅. We should take inverse, ignore here for now.')
        
        return True
            
    
    
    def weak_traceback(self, num_layer, attack_result, list_elements):
        #Return if this is the start of a branch
        if num_layer == 0 or len(list_elements) == 1:
            return num_layer
        vars, matrix, bias = self.zono_gen.get_generator(list_elements[-1])

        #from list_element[-2] to list_element[0]
        for i in range(num_layer-1, num_layer-len(list_elements), -1):
            element = list_elements[i - num_layer - 1]
            vars, matrix, bias = self.zono_gen.get_generator(element)
            np_matrix, np_bias = np.array(matrix), np.array(bias)
            lb, ub = zono_bounds(self.man, element)


            X = self.concrete_executor.get_fresh_input(i + 1)
            Y = self.concrete_executor.get_concrete_model(i + 1)
            x = np.add(np.matmul(np_matrix, np.reshape(attack_result.gene, (-1, 1))[:len(vars)]), np_bias)
            y = Y.eval(feed_dict={X:x})
            if np.argmax(y) == self.property.target or not check_in_bound(x, [lb, ub]):
                return i + 1
        
        return num_layer-len(list_elements)+1
    
    
    def attack_condition(self, task, i, method=1):
        if method == 1:
            if task.start_ir == i:
                return False
        elif method == 2:
            return self.task_manager.checkLimit(i)
        else:
            if task.start_ir != i:
                task.neurons_reset()
            return self.task_manager.checkLimit(i, 7)
        return True
    
    def analyze_task(self, task, refine):
        # we do the initialization here because it might be re-used. Every time we re-use it, we get a new object of Layers()
        i=task.start_ir
        element = task.element
        taskhid = task.get_hid()
        nn = task.nn
        relu_groups = task.relu_groups.copy()
        #list_elements = [zonotope_deepcopy(self.man, element)]
        if self.domain == "deepzono":
            task.element, task.nn = element_copy(self.man, self.domain, element, self.ir_list, i-1, nn), None
        else:
            task.element, task.nn = element_copy(self.man, self.domain, element, self.ir_list, i-1, nn)
        list_elements = [element_copy(self.man, self.domain, element, self.ir_list, i-1, nn)] 
        # print('* Given abstraction (before layer', i, '): ', sep='', end='')
        #zono_debug_info_print(self.man, element, '↓in')
        use_krelu = not self.relu_groups
        while i<len(self.ir_list):
            last_status = (i, list_elements[-1], taskhid, task.get_split_neurons(), relu_groups.copy())
            this_layer = self.ir_list[i]
            print(back_blue, taskhid.ljust(label_len+(2*i), '┈'), ' data @', i-1, ' ⟿ ⟿  ', this_layer.op().center(6,' '), ' ⟿ ⟿  data @', i, reset, sep='')    
            element = this_layer.transformer(nn, self.man, element, nn.specLB, nn.specUB, self.relu_groups, self.refine, use_krelu)
            #print("ReLu group:", self.relu_groups)
            list_elements.append(element_copy(self.man, self.domain, element, self.ir_list, i, nn))

            if debug_mode:
                # print('* Resulted abstraction (after layer', i, '): ', sep='', end='')
                debug_info_print(self.man, element, self.domain, i-1, 'out')
                #poly_split(self.man, element, self.ir_list, nn, 1, [1])

            if self.use_abstract_attack and this_layer.op() in ['Relu', 'Sigmoid', 'Tanh'] and refine and self.attack_condition(task, i, 3):
                attack_result = self.attack_abstract_domain(i + 1, element)
                if attack_result.success:
                    #new_layer = self.weak_traceback(i, attack_result, list_elements)
                    new_layer = i
                    if new_layer == 0:
                        print("Counterexample found for the input layer, and thus the property does not hold. This task fails.")
                        elina_abstract0_free(self.man, element)
                        for e in list_elements:
                            if self.domain == "deeppoly" or self.domain == "refinepoly":
                                e = e[0]
                            elina_abstract0_free(self.man, e)
                        gc.collect()
                        return False
                    elif new_layer != i:
                        if new_layer == task.start_ir-1:
                            new_layer += 1
                        last_status = (new_layer, list_elements[new_layer-task.start_ir], taskhid, task.get_split_neurons(), relu_groups.copy())
                        attack_result = self.update_attack_result(new_layer+1, list_elements[new_layer-task.start_ir+1], attack_result)
 

                    # print(back_magenta, 'attack succeeded, and thus start refining.....', reset)
                    if self.use_abstract_refine:
                        if self.refine_abstract_domain(attack_result, *last_status) is False:
                            i += 1
                            continue
                            print('** The domain can not be refined anymore. **')
                            return False  # in this case, we can not refine the domain any more, and thus return false
                    elina_abstract0_free(self.man, element)
                    for e in list_elements:
                        if self.domain == "deeppoly" or self.domain == "refinepoly":
                            e = e[0]
                        elina_abstract0_free(self.man, e)
                    gc.collect()
                    return None
            i+=1
        
        #counter, var_list, model = create_model(nn, nn.in_LB, nn.in_UB, nn.specLB, nn.specUB, self.relu_groups, nn.numlayer, config.complete==True)
        cprint(back_blue, taskhid.ljust(label_len+(2*i), '┈'), ' data @', i-1, ' ⟿ ⟿  Compare ⟿ ⟿  ', reset, sep='', end='')
        dominant_class = -1
        label_failed = []
        candidate_labels = []
        if self.target_label == -1:
            candidate_labels = list(range(self.nr_classes))
        else:
            candidate_labels.append(self.target_label)
        for i in candidate_labels:
            flag = True
            label = i
            # if i is the label, then i!=j ==> element[i]>element[j]
            for j in range(self.nr_classes):
                if(self.domain == "deepzono"):
                    args = (self.man, element, i, j)
                else:
                    args = (self.man, element, i, j, True) 
                if i!=j and not self.is_greater(*args):
                    flag = False	
                    break
            if flag:	
                dominant_class = i	
                break
        elina_abstract0_free(self.man, element)
        for e in list_elements:
            if self.domain == "deeppoly" or self.domain == "refinepoly":
                e = e[0]
            elina_abstract0_free(self.man, e)
        gc.collect()
        if dominant_class == self.target_label:
            print(back_green, bold, 'label:', dominant_class, '  (target:', self.target_label, ')', reset, nonbold, sep='')
            return True
        else:
            print(back_red, bold, 'label:', dominant_class, '  (target:', self.target_label, ')', reset, nonbold, sep='')
            return False
    
    
    
    
    
    
    
    def analyze_abstract(self, specLB, specUB):
        """
        processes self.ir_list and returns the resulting abstract element
        """
        root = '⡇'
        prefix_len = len(self.ir_list)+4 if len(self.ir_list)>20 else 20
        cprint(back_blue, root.ljust(label_len, '┈'), ' input ⟿ ⟿  Abstract ⟿ ⟿  data @0', reset, sep='')
        element = self.ir_list[0].transformer(self.man)
        start = time.time()
        global split_check_time
        if (split_check_time>0):
            for i in range(split_check_time): # check the domain splitter
                print(back_green, '----------------  orginal ----------------', reset)
                # elina_abstract0_simple_print(self.man, element)
                zono_constrained_spec(self.man, element, True)
                k = 2
                nr_dims = len(specLB)
                split_dims = sample(range(0, nr_dims), k)
                print('split_dims:', split_dims)
                split_values = [0.0]*k
                print('========= zono_specified_split at 0 =========')
                elements_k = zono_specified_split(self.man, element, k, split_dims, split_values)
                for (i, e) in enumerate(elements_k):
                    print(back_blue, '----------------', i, '----------------', reset)
                    # elina_abstract0_simple_print(self.man, e)
                    zono_constrained_spec(self.man, e, True)
                print('========= zono_isotonic_split (n=3) =========')
                elements_nk = zono_isotonic_split(self.man, element, 3, k, split_dims)
                for (i, e) in enumerate(elements_nk):
                    print(back_green, '----------------', i, '----------------', reset)
                    # elina_abstract0_simple_print(self.man, e)
                    zono_constrained_spec(self.man, e, True)
                    # self.task_manager.add_task(layer_no+1, e, root+'test', layer_no)
                    # print(red, '∂∂ add a new task....', self.tasks[-1], reset)
            sys.exit(0)
        
        layer_no=1
        if self.use_abstract_attack: # this is a shortcut for early termination
            attack_result = self.attack_abstract_domain(layer_no, element)
            #print("Debug")
            #debug_info_print(self.man, element, self.domain, 0)
            if attack_result.success:
                #dominant_class, nb = self.concrete_executor.forward(attack_result.x)
                #print("Dominent class", dominant_class, check_in_bound(attack_result.x, [specLB, specUB]))
                print('Counterexample found for the input layer, and thus the property does not hold. This task fails.')
                return Verified_Result.UnSafe
        
        self.task_manager.add_task(layer_no, element, self.nn, '', '◈', set(), self.relu_groups)
                
        ### SCHEDULE AND RUN THE TASKS ###
        task_verified = Verified_Result.Safe
        # max_num_tasks = 20
        while self.task_manager.size()>0: # and max_num_tasks>0:
            print('\n')
            print('-'*100)
            this_task = self.task_manager.pop_task()
            taskid = this_task.get_id()
            print(bold, '<', taskid, '> ---> analyzing....', reset, ' ', self.task_manager, sep='')
            # print(lgreen, bold, '<', taskid, '> ---> on analyzing task ', this_task, reset, '  {remaining ', self.task_manager, '}', sep='')
            status = self.analyze_task(this_task, False)
            if not status and self.use_abstract_attack and self.task_manager.size() < 450 and self.task_manager.cid < 4000 and time.time()-start < 1200:
                print(lgreen, bold, "\nFailed. Starting to attack and refine. \n", nonbold, reset)
                timeout_lp, timeout_milp = config.timeout_lp, config.timeout_milp
                config.timeout_lp, config.timeout_milp = 1, 1
                status = self.analyze_task(this_task, True)
                config.timeout_lp, config.timeout_milp = timeout_lp, timeout_milp
            this_task.destroy(self.man)
            
            if status is None:
                print(lblue, bold, '<', taskid, '> <--- this task is splitted.', nonbold, reset, sep='')
            elif status is True:
                print(lgreen, bold, '<', taskid, '> <--- this task is verified.', nonbold, reset, sep='')
            elif status is False:
                print(lred, bold, '<', taskid, '> <--- this task is falstified.', nonbold, reset, sep='')
                print(lred, bold, 'This tasks fails and thus system aborts.', nonbold, reset, sep='')
                task_verified = Verified_Result.Unknow
                self.task_manager.destroy(self.man)
                break
            else:
                assert(0)
            # max_num_tasks-=1
        return task_verified

    def get_abstract0(self):
        """
        processes self.ir_list and returns the resulting abstract element
        """
        element = self.ir_list[0].transformer(self.man)
        nlb = []
        nub = []
        self.causality = Causality(self.concrete_executor, self.target_label)
        #self.causality = None
        for i in range(1, len(self.ir_list)):
            element_test_bounds = self.ir_list[i].transformer(self.nn, self.man, element, nlb, nub, self.relu_groups, self.refine)
            element = element_test_bounds
        if self.domain == "refinepoly":
            gc.collect()
        return element, nlb, nub

    def analyze_refine(self):
        #from ai_milp import milp_callback, create_model
        element, nlb, nub = self.get_abstract0()
        output_size = 0
        output_size = self.ir_list[-1].output_length
        dominant_class = -1
        self.nn.ffn_counter = 0
        self.nn.conv_counter = 0
        self.nn.pool_counter = 0
        self.nn.concat_counter = 0
        self.nn.tile_counter = 0
        self.nn.residual_counter = 0
        self.nn.activation_counter = 0  
        counter, var_list, model = create_model(self.nn, self.nn.in_LB, self.nn.in_UB, nlb, nub,self.relu_groups, self.nn.numlayer, config.complete==True)
        if config.complete==True:
            model.setParam(GRB.Param.TimeLimit,self.timeout_milp)
        else:
            model.setParam(GRB.Param.TimeLimit,self.timeout_lp)
        num_var = len(var_list)
        output_size = num_var - counter
        label_failed = []
        x = None
        candidate_labels = []
        if self.target_label == -1:
            for i in range(output_size):
                candidate_labels.append(i)
        else:
            candidate_labels.append(self.target_label)
        adv_labels = []
        for i in range(output_size):
            adv_labels.append(i)
        for i in candidate_labels:
            flag = True
            label = i
            for j in adv_labels:
                if label!=j and not self.is_greater(self.man, element, label, j, True):
                    obj = LinExpr()
                    obj += 1*var_list[counter+label]
                    obj += -1*var_list[counter + j]
                    model.setObjective(obj,GRB.MINIMIZE)
                    if config.complete == True:
                        model.optimize(milp_callback)
                        if not hasattr(model,"objbound") or model.objbound <= 0:
                            flag = False
                            if self.target_label!=-1:
                                label_failed.append(j)
                            if model.solcount > 0:
                                x = model.x[0:len(self.nn.in_LB)]
                            break    
                    else:
                        model.optimize()
                        print("objval ", j, model.objval)
                        if model.Status!=2:
                            print("model was not successful status is", model.Status)
                            model.write("final.mps")
                            flag = False
                            break
                        elif model.objval < 0:
                    
                            flag = False
                            if model.objval != math.inf:
                                x = model.x[0:len(self.nn.in_LB)]
                            break
            if flag:
                dominant_class = i
                break
        elina_abstract0_free(self.man, element)
        if dominant_class == self.target_label:
            is_verified = Verified_Result.Safe
        else:
            is_verified = Verified_Result.Unknow
        model.reset()
        gc.collect()
        output_info = (nlb, nub, label_failed, x)
        return is_verified, output_info


    ## This function directly return True or False as the final analysis result.
    def analyze(self, specLB, specUB=None, pid=None, use_abstract_attack=False, attack_method='scipy', use_abstract_refine=False):
        """
        analyses the network with the given input
        -------
        output: int
            index of the dominant class. If no class dominates then returns -1
        """
        if self.refine or self.domain=="deeppoly":
            self.task_manager.init(pid, len(self.ir_list))
            self.use_abstract_attack = use_abstract_attack
            self.attack_method = attack_method
            self.use_abstract_refine = use_abstract_refine 
            if use_abstract_attack:
                assert self.concrete_executor, "use abstract_attack must equip the analyzer with a concrete executor ahead"
                assert self.property, "use abstract_attack must provide the analyzer with desired property"
                
            # print('************ analyzebox.analyze ***********')
            
            is_verified = self.analyze_abstract(specLB, specUB)
            output_info = (self.task_manager.get_num_all_task(), self.task_manager.get_largest_size())
            return is_verified, output_info
        else:
            return self.analyze_refine()
