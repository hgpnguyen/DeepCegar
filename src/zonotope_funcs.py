'''
@author: Adrian Hoffmann
@update: Li Jiaying
'''

import math
import numpy as np
from elina_abstract0 import *
from elina_manager import *
from elina_interval import *
from elina_lincons0 import *
from elina_scalar import *
from elina_coeff import *
from opt_pk import *
from zonoml import *
from fppoly import *
#from fppoly_gpu import *
from colors import *
from ctypes.util import *
from layers import *
from poly import *

#debug_mode = False
debug_mode = True

libc = CDLL(find_library('c'))
printf = libc.printf
cstdout = c_void_p.in_dll(libc, 'stdout')

def to_str(str):
    return bytes(str, 'utf-8')

def print_c(str):
    printf(to_str(str))

# --- the following assume we would like to split first k ranked dimensions at split_values ---
def zono_specified_split(man, element, k, split_dims, split_values=None): #, nlb, nub):
    assert len(split_dims) >= k
    if split_values is None:
        split_values = [0.0]*k
    assert len(split_values) >= k
    # print(back_red, 'specified split configuration(k=', k, '): [', sep='', end='')
    # for i in range(k):
    #     print(i, '(dim:', split_dims[i], ' split:', split_values[i], ')', sep='', end='')
    #     if i!=k-1:
    #         print(', ', end='')
    # print(']', reset)

    c_split_dims = (ctypes.c_size_t * k)(*split_dims)
    c_split_values = (ctypes.c_double * k)(*split_values)
    c_elements = zonotope_specified_split(man, element, k, c_split_dims, c_split_values)
    elements = [c_elements[i] for i in range(1<<k)]
    # print('size:', len(elements))
    # for i, element in enumerate(elements):
    #     print('->', i, ':', sep='', end='')
    #     elina_abstract0_simple_print(man, element)
    # print(back_red, '===============', reset)
    return elements
    



# --- the following assume we would like to split first k ranked dimensions at n-even splitter---
def zono_isotonic_split(man, element, n, k, split_dims):
    assert len(split_dims) >= k
    print(back_red, 'n-isotonic-k-split configuration(n=', n, ' k=', k, ')  operating on dims:', split_dims[:k], reset, sep='')

    c_split_dims = (ctypes.c_size_t * k)(*split_dims)
    c_elements = zonotope_isotonic_split(man, element, n, k, c_split_dims)
    elements = [c_elements[i] for i in range(n**k)]
    # print('size:', len(elements))
    # for i, element in enumerate(elements):
    #     print('->', i, ':', sep='', end='')
    #     elina_abstract0_simple_print(self.man, element)
    # print(back_red, '===============', reset)
    return elements




def zono_constrained_spec(man, element, dprint=False):
    noise_cons = zonotope_constrained_spec(man, element)
    size = int(noise_cons[0])
    # int_dim = noise_cons[1]
    # real_dim = noise_cons[2]
    es = []
    elb = []
    eub = []
    for i in range(size):
        base = 3*(i+1)
        es.append(int(noise_cons[base]))
        elb.append(noise_cons[base+1])
        eub.append(noise_cons[base+2])
    if dprint:
        if size>0: 
            cprint(back_red, "With constrained noise symbols: (size:", size, ')', reset)
            for i in range(size):
                print('ε', es[i], ' [', elb[i], ', ', eub[i], ']', sep='')
        else: 
            cprint(back_gray, "WithOut constrained noise symbols ", reset)
    return size, es, elb, eub


def debug_info_print(man, element, domain, layerno, prefix = '', lb=None, ub=None, end='\n'):
    if domain == "deepzono":
        zono_debug_info_print(man, element, prefix, lb, ub, end)
    else:
        poly_debug_info_print(man, element, layerno)


def zono_bounds(man, element, num_vars=0, start_offset=0):
    assert(start_offset==0)
    dimension = elina_abstract0_dimension(man, element)
    var_in_element = dimension.intdim + dimension.realdim
    if num_vars==0:
        num_vars = var_in_element
    bounds = elina_abstract0_to_box(man, element)
    itv = [bounds[i] for i in range(start_offset, num_vars+start_offset)]
    lbi = [x.contents.inf.contents.val.dbl for x in itv]
    ubi = [x.contents.sup.contents.val.dbl for x in itv]
    elina_interval_array_free(bounds, var_in_element)
    return lbi, ubi




def zono_debug_info_print(man, element, prefix, lb=None, ub=None, end='\n'):
    if debug_mode:
        if element is not None:
            print('* ', sep='', end='')
            #elina_abstract0_simple_print(man, element, end='')
            cstdout = c_void_p.in_dll(libc, 'stdout')
            elina_abstract0_fprint(cstdout, man, element, None)
            zlb, zub = zono_bounds(man, element)
            print_c("\n")
            print(back_llgray, 'box:', end='')
            for i, (l,u) in enumerate(zip(zlb, zub)):
                print(i, ':[', l, ', ', u, ']', sep='', end=' ')
            print(reset)
            
        if lb!=None and ub!=None:
            print(gray, '  (', prefix, ') ', sep='', end='')
            print(' lb,ub:', end='')
            for i, (l,u) in enumerate(zip(lb, ub)):
                print(i, ':[', l, ', ', u, ']', sep='', end=' ')
            print(reset, end=end)

def poly_bound(man, element, layerno):
    bounds = box_for_layer(man, element, layerno)
    num_neurons = get_num_neurons_in_layer(man, element, layerno)
    itv = [bounds[i] for i in range(num_neurons)]
    lbi = [x.contents.inf.contents.val.dbl for x in itv]
    ubi = [x.contents.sup.contents.val.dbl for x in itv]
    elina_interval_array_free(bounds,num_neurons)
    return lbi, ubi

def pk_bound(man, element):
    dimension = elina_abstract0_dimension(man, element)
    var_in_element = dimension.intdim + dimension.realdim
    lbi, ubi = [], []
    for i in range(var_in_element):
        interval = elina_abstract0_bound_dimension(man, element, i)
        down = c_double()
        elina_double_set_scalar(down, interval.contents.inf, MpfrRnd.MPFR_RNDD)
        up = c_double()
        elina_double_set_scalar(up, interval.contents.sup, MpfrRnd.MPFR_RNDU)
        lbi.append(down.value)
        ubi.append(up.value)
        elina_interval_free(interval)
    return lbi, ubi


def get_bound(man, element, domain, layerno):
    if domain == "deepzono":
        return zono_bounds(man, element)
    else:
        return poly_bound(man, element, layerno)

def poly_debug_info_print(man, element, layerno):
    num_neurons = get_num_neurons_in_layer(man, element, layerno)
    lexpr = [get_lexpr_for_output_neuron(man, element, i) for i in range(num_neurons)]
    uexpr = [get_uexpr_for_output_neuron(man, element, i) for i in range(num_neurons)]
    for i, (l,u) in enumerate(zip(lexpr, uexpr)):
        print_c("{}: ".format(str(i)))
        elina_linexpr0_print(l, None)
        print_c(' <= x <= ') 
        elina_linexpr0_print(u, None)
        print_c('\n')

    plb, pub = poly_bound(man, element, layerno)
    print_c('box:')
    for i, (l,u) in enumerate(zip(plb, pub)):
        print_c("{}:[{}, {}]".format(i,l,u))
    print_c('\n')
    print_c(reset)

def element_copy(man, domain, element, ir_list, stop, nn):
    assert domain == "deepzono" or domain == "deeppoly" or domain == "refinepoly"
    refine = True if 'refine' in domain else False  
    if domain == "deepzono":
        return elina_abstract0_copy(man, element)
    else:
        #copy = elina_abstract0_copy(man, element)
        return poly_copy(man, ir_list, stop, nn, refine)

def poly_copy(man, ir_list, stop, nn, refine):
    assert stop <= len(ir_list)
    element = ir_list[0].transformer(man)
    new_nn = Layers(nn.in_LB, nn.in_UB)
    new_nn.copy(nn)
    nlb_, nub_, relu_groups = [], [], []
    for i in range(1, stop + 1):
        element = ir_list[i].transformer(new_nn, man, element, nlb_, nub_, relu_groups, False)
        lb , ub = poly_bound(man, element, i-1)
        nlb, nub = nn.specLB[i-1], nn.specUB[i-1]
        assert len(lb) == len(nlb) and len(ub) == len(nub)
        for j in range(len(lb)):
            if lb[j] != nlb[j] or ub[j] != nub[j]:
                update_bounds_for_neuron(man, element, i-1, j, nlb[j], nub[j])
                #new_nn.specLB[-1][j] = nlb[j]
                #new_nn.specUB[-1][j] = nub[j]    
    return element, new_nn

def zonotope_dim_split_check(man, z0, z1, dim, split):
    lb1, ub1 = zono_bounds(man, z0)
    lb2, ub2 = zono_bounds(man, z1)
    if lb2[dim] >= split or ub2[dim] <= split:
        return True
    else:
        cprint("validation fails.", color="red")
        print("split:{} interval change:[{},{}]==>[{},{}] ".format(split, lb1[dim], ub1[dim], lb2[dim], ub2[dim]))
        return False

def zono_abstract_split(man, element, k, split_dims, split_values=None):
    assert len(split_dims) >= k
    if split_values is None:
        split_values = [0.0]*k
    assert len(split_values) >= k

    total_size = 1<<k

    z0 = element
    res = []
    linexprs = [[], []]

    for i in range(k):
        linexpr = elina_linexpr0_alloc(ElinaLinexprDiscr.ELINA_LINEXPR_SPARSE, 1)
        linexprs[0].append(linexpr)
        cst = pointer(linexpr.contents.cst)
        elina_scalar_set_double(cst.contents.val.scalar, split_values[i])
        linterm = pointer(linexpr.contents.p.linterm[0])
        linterm.contents.dim = ElinaDim(split_dims[i])
        coeff = pointer(linterm.contents.coeff)
        elina_scalar_set_int(coeff.contents.val.scalar, -1)


        linexpr = elina_linexpr0_alloc(ElinaLinexprDiscr.ELINA_LINEXPR_SPARSE, 1)
        linexprs[1].append(linexpr)
        cst = pointer(linexpr.contents.cst)
        elina_scalar_set_double(cst.contents.val.scalar, -split_values[i])
        linterm = pointer(linexpr.contents.p.linterm[0])
        linterm.contents.dim = ElinaDim(split_dims[i])
        coeff = pointer(linterm.contents.coeff)
        elina_scalar_set_int(coeff.contents.val.scalar, 1)

    lincons0_array = elina_lincons0_array_make(k)
    #cstdout = c_void_p.in_dll(libc, 'stdout')
    for i in range(total_size):
        for j in range(k):
            is_right = i>>j&1
            print("is right:", is_right)
            lincons0_array.p[j].linexpr0 = linexprs[is_right][j]
            lincons0_array.p[j].constyp = ElinaConstyp.ELINA_CONS_SUPEQ

        z1 = elina_abstract0_meet_lincons_array(man, False, z0, lincons0_array)
        for j in range(k):
            is_pass = zonotope_dim_split_check(man, z0, z1, split_dims[j], split_values[j])
            if not is_pass:
                print("result index {}, dim: {} ".format(i, j))
			
        res.append(z1)
    
    for i in range(k):
        elina_linexpr0_free(linexprs[0][i])
        elina_linexpr0_free(linexprs[1][i])

    return res

def poly_split(man, element, ir_list, nn, k, split_dims, split_values=None, meet_lincons_arr=True):
    if len(split_dims) < k:
        k = len(split_dims)
    assert len(split_dims) >= k
    if split_values is None:
        split_values = [0.0]*k
    assert len(split_values) >= k

    if nn.calc_layerno() - 2 != -1:
        last_neuron = len(nn.specLB[nn.calc_layerno() - 2])
    else:
        last_neuron = len(nn.in_LB)

    total_size = 1<<k
    res = []
    lb, ub = poly_bound(man, element, nn.calc_layerno()-1)
    if meet_lincons_arr:
        o1, pk_man = fpoly_to_poly(man, element, nn, 1)

        linexprs = create_k_lincons_array(k, [last_neuron+i for i in split_dims], split_values)
        lincons0_array = elina_lincons0_array_make(k)
        
    for i in range(total_size):
        new_e, new_nn = poly_copy(man, ir_list, nn.calc_layerno(), nn, False)
        for j in range(k):
            is_right = i>>j&1
            if meet_lincons_arr:
                lincons0_array.p[j].linexpr0 = linexprs[is_right][j]
                lincons0_array.p[j].constyp = ElinaConstyp.ELINA_CONS_SUPEQ
            if is_right:
                update_bounds_for_neuron(man, new_e, new_nn.calc_layerno()-1, split_dims[j], split_values[j], ub[split_dims[j]])
                new_nn.specLB[-1][split_dims[j]] = split_values[j]
            else:
                update_bounds_for_neuron(man, new_e, new_nn.calc_layerno()-1, split_dims[j], lb[split_dims[j]], split_values[j])
                new_nn.specUB[-1][split_dims[j]] = split_values[j]

        if meet_lincons_arr:
            o2 = elina_abstract0_meet_lincons_array(pk_man, False, o1, lincons0_array)
            if elina_abstract0_is_bottom(pk_man, o2):
                res.append((new_e, new_nn))
                continue
            lbi, ubi = pk_bound(pk_man, o2)
            lbi, ubi = lbi[last_neuron:], ubi[last_neuron:]
            for j in range(len(new_nn.specUB[-1])):
                violate = lbi[j] > ubi[j] or lbi[j] > new_nn.specUB[-1][j] or ubi[j] < new_nn.specLB[-1][j]
                new_low = max(lbi[j], new_nn.specLB[-1][j]) if not math.isinf(lbi[j]) and not violate else new_nn.specLB[-1][j]
                new_up = min(ubi[j], new_nn.specUB[-1][j]) if not math.isinf(ubi[j]) and not violate else new_nn.specUB[-1][j]
                update_bounds_for_neuron(man, new_e, new_nn.calc_layerno()-1, j, new_low, new_up)
                new_nn.specLB[-1][j], new_nn.specUB[-1][j] = new_low, new_up
            elina_abstract0_free(pk_man, o2)

        #print("Split:", i)
        #poly_debug_info_print(man, new_e, new_nn.calc_layerno()-1)
        #print("New Layer bound:", new_nn.specLB[-1], new_nn.specUB[-1])
        res.append((new_e, new_nn))

    if meet_lincons_arr:
        for i in range(k):
            elina_linexpr0_free(linexprs[0][i])
            elina_linexpr0_free(linexprs[1][i])
        elina_abstract0_free(pk_man, o1)
        elina_manager_free(pk_man)
    return res

def abstract_split(man, domain, element, ir_list, nn, k, split_dims, split_values=None):
    if domain == "deepzono":
        return zono_abstract_split(man, element, k, split_dims, split_values)
    else:
        return poly_split(man, element, ir_list, nn, k, split_dims, split_values, False)

def create_k_lincons_array(k, split_dims, split_values):
    linexprs = [[], []]

    for i in range(k):
        linexpr = elina_linexpr0_alloc(ElinaLinexprDiscr.ELINA_LINEXPR_SPARSE, 1)
        linexprs[0].append(linexpr)
        elina_linexpr0_set_cst_scalar_double(linexpr, split_values[i])
        elina_linexpr0_set_coeff_scalar_int(linexpr, split_dims[i], -1)

        linexpr = elina_linexpr0_alloc(ElinaLinexprDiscr.ELINA_LINEXPR_SPARSE, 1)
        linexprs[1].append(linexpr)
        elina_linexpr0_set_cst_scalar_double(linexpr, -split_values[i])
        elina_linexpr0_set_coeff_scalar_int(linexpr, split_dims[i], 1)
    return linexprs

def fpoly_to_poly(man, element, nn, k=1):
    layer_idx = nn.calc_layerno()-1
    curr_neuron = len(nn.specLB[layer_idx])
    if layer_idx - k != -1:
        last_lb, last_ub = nn.specLB[layer_idx-k], nn.specUB[layer_idx-k]
    else:
        last_lb, last_ub = nn.in_LB, nn.in_UB
    last_neuron = len(last_lb)

    lincons0_array = elina_lincons0_array_make((curr_neuron + last_neuron)*2)
    #lst_le, lst_ge = element_to_poly(man, element, layer_idx-k+1, layer_idx, (last_lb, last_ub), nn)
    #num_dimension = lst_le[0].shape[0] - 1
    for i in range(last_neuron):
        lb_neuron = create_linexpr([1, -last_lb[i]], [i])
        ub_neuron = create_linexpr([-1, last_ub[i]], [i])
        lincons0_array.p[i*2].constyp = ElinaConstyp.ELINA_CONS_SUPEQ
        lincons0_array.p[i*2].linexpr0 = lb_neuron
        lincons0_array.p[i*2+1].constyp = ElinaConstyp.ELINA_CONS_SUPEQ
        lincons0_array.p[i*2+1].linexpr0 = ub_neuron

    for i in range(curr_neuron):
        ub_expr = to_sparseExpr(get_uexpr_for_output_neuron(man, element, i), True)
        lb_expr = to_sparseExpr(get_lexpr_for_output_neuron(man, element, i), True)
        #lb_expr = to_dense(lst_le[i])
        #ub_expr = to_dense(lst_ge[i])
        addDimension(ub_expr, last_neuron+i, -1)
        negLin(lb_expr)
        addDimension(lb_expr, last_neuron+i, 1)
        lincons0_array.p[(i+last_neuron)*2].constyp = ElinaConstyp.ELINA_CONS_SUPEQ
        lincons0_array.p[(i+last_neuron)*2].linexpr0 = lb_expr
        lincons0_array.p[(i+last_neuron)*2+1].constyp = ElinaConstyp.ELINA_CONS_SUPEQ
        lincons0_array.p[(i+last_neuron)*2+1].linexpr0 = ub_expr
    
    pk_man = opt_pk_manager_alloc(False)
    top = elina_abstract0_top(pk_man,0,curr_neuron+last_neuron)
    o1 = elina_abstract0_meet_lincons_array(pk_man,False,top, lincons0_array)
    elina_lincons0_array_clear(lincons0_array)
    return o1, pk_man

def addDimension(linexpr0, dim, coeff):
    size = elina_linexpr0_size(linexpr0)
    if linexpr0.contents.discr == ElinaLinexprDiscr.ELINA_LINEXPR_DENSE:
        new_size = dim + 1
    else:
        new_size = size + 1
    elina_linexpr0_realloc(linexpr0, new_size)
    elina_linexpr0_set_coeff_scalar_double(linexpr0, dim, coeff)

def negLin(linexpr0):
    size = elina_linexpr0_size(linexpr0)
    for i in range(size):
        if linexpr0.contents.discr == ElinaLinexprDiscr.ELINA_LINEXPR_DENSE:
            dim = i
        else:
            dim = linexpr0.contents.p.linterm[i].dim
        coeff0 = elina_linexpr0_coeffref(linexpr0, dim)
        elina_coeff_neg(coeff0, coeff0)
    const0 = elina_linexpr0_cstref(linexpr0)
    elina_coeff_neg(const0, const0)

def create_linexpr(coeffs, dims = None):
    if not dims:
        dims = list(range(len(coeffs)-1))
    assert len(dims) + 1 == len(coeffs)
    linexpr0 = elina_linexpr0_alloc(ElinaLinexprDiscr.ELINA_LINEXPR_SPARSE, len(coeffs))
    elina_linexpr0_set_cst_scalar_double(linexpr0, coeffs[-1])
    for i in range(len(coeffs[:-1])):
        elina_linexpr0_set_coeff_scalar_double(linexpr0, dims[i], coeffs[i])
    return linexpr0

def to_sparseExpr(linexpr, destroy = False):
    size = elina_linexpr0_size(linexpr)
    linexpr0 = elina_linexpr0_alloc(ElinaLinexprDiscr.ELINA_LINEXPR_SPARSE, size)
    cst = elina_coeff_alloc(ElinaCoeffDiscr.ELINA_COEFF_INTERVAL)
    elina_linexpr0_get_cst(cst, linexpr)
    elina_linexpr0_set_cst_scalar_double(linexpr0, coeff_to_db(cst))
    for i in range(size):
        coeff = elina_coeff_alloc(ElinaCoeffDiscr.ELINA_COEFF_INTERVAL)
        if linexpr.contents.discr == ElinaLinexprDiscr.ELINA_LINEXPR_DENSE:
            dim = i
        else:
            dim = linexpr.contents.p.linterm[i].dim
        elina_linexpr0_get_coeff(coeff, linexpr, dim)
        elina_linexpr0_set_coeff_scalar_double(linexpr0, dim, coeff_to_db(coeff))
    #print_c("old expr: ")
    #elina_linexpr0_print(linexpr, None)
    #print_c(" new expr: ")
    #elina_linexpr0_print(linexpr0, None)
    #print_c("\n")
    if destroy:
        elina_linexpr0_free(linexpr)
    return linexpr0

def to_dense(expr):
    size = expr.shape[0]
    linexpr0 = elina_linexpr0_alloc(ElinaLinexprDiscr.ELINA_LINEXPR_DENSE, size)
    elina_linexpr0_set_cst_scalar_double(linexpr0, expr[-1])
    for i in range(size-1):
        elina_linexpr0_set_coeff_scalar_double(linexpr0, i, expr[i])
    return linexpr0

def interval_to_db(interval):
    up = c_double()
    elina_double_set_scalar(up, interval.contents.sup, MpfrRnd.MPFR_RNDU)
    down = c_double()
    elina_double_set_scalar(down, interval.contents.inf, MpfrRnd.MPFR_RNDD)
    return (up.value + down.value) / 2

def coeff_to_db(coeff):
    if coeff.contents.discr == ElinaCoeffDiscr.ELINA_COEFF_SCALAR:
        db = c_double()
        elina_double_set_scalar(db, coeff.contents.val.scalar, MpfrRnd.MPFR_RNDNA)
        return db.value
    return interval_to_db(coeff.contents.val.interval)

def expr_to_np(expr, length):
    expr_np = np.zeros(length, dtype=np.float)
    size = elina_linexpr0_size(expr)
    cst = elina_coeff_alloc(ElinaCoeffDiscr.ELINA_COEFF_INTERVAL)
    elina_linexpr0_get_cst(cst, expr)
    expr_np[-1] = coeff_to_db(cst)

    for i in range(size):
        coeff = elina_coeff_alloc(ElinaCoeffDiscr.ELINA_COEFF_INTERVAL)
        if expr.contents.discr == ElinaLinexprDiscr.ELINA_LINEXPR_DENSE:
            dim = i
        else:
            dim = expr.contents.p.linterm[i].dim
        elina_linexpr0_get_coeff(coeff, expr, dim)
        expr_np[i] = coeff_to_db(coeff)

    return expr_np

def element_to_poly(man, element, start, end, input_bound, nn):
    x0_poly = Poly()
    lw, up = input_bound
    no_neurons = len(lw)
    x0_poly.lw, x0_poly.up = np.array(lw), np.array(up)
    x0_poly.le = np.eye(no_neurons + 1)[:-1]
    x0_poly.ge = np.eye(no_neurons + 1)[:-1]
    lst_poly = [x0_poly]
    last_lay_no_nerons = no_neurons
    for idx in range(start, end+1):
        no_neurons = get_num_neurons_in_layer(man, element, idx)
        le, ge = [], []
        idx_poly = Poly()
        for i in range(no_neurons):
            if idx == end:
                uexpr = get_uexpr_for_output_neuron(man, element, i)
                lexpr = get_lexpr_for_output_neuron(man, element, i)
            else:
                uexpr = get_output_uexpr_defined_over_previous_layers(man,element, i, idx)
                lexpr = get_output_lexpr_defined_over_previous_layers(man,element, i, idx)
            uexpr_np = expr_to_np(uexpr, last_lay_no_nerons+1)
            lexpr_np = expr_to_np(lexpr, last_lay_no_nerons+1)
            ge.append(uexpr_np)
            le.append(lexpr_np)
        idx_poly.ge = np.array(ge)
        idx_poly.le = np.array(le)
        idx_poly.lw = np.array(nn.specLB[idx])
        idx_poly.up = np.array(nn.specUB[idx])
        last_lay_no_nerons = no_neurons 
        lst_poly.append(idx_poly)
    
    last_poly = lst_poly.pop()
    list_ge, list_le = last_poly.back_substitute(lst_poly, True)

    return list_ge, list_le
