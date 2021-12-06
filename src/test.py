from posixpath import join
import sys
sys.path.insert(0, '../elina/python_interface/')
sys.path.insert(0, '../os/deepg/code/')
from elina_abstract0 import *
from elina_manager import *
from elina_lincons0 import *
from elina_scalar import *
from elina_coeff import *
from opt_pk import *
from zonoml import *
from elina_interval import *
from ctypes.util import find_library

libc = CDLL(find_library('c'))
printf = libc.printf
cstdout = c_void_p.in_dll(libc, 'stdout')

def to_str(str):
    return bytes(str, 'utf-8')

def print_c(str):
    printf(to_str(str))

def create_linexpr(coeffs, dims = None):
    if not dims:
        dims = list(range(len(coeffs)-1))
    linexpr0 = elina_linexpr0_alloc(ElinaLinexprDiscr.ELINA_LINEXPR_SPARSE, len(coeffs))
    elina_linexpr0_set_cst_scalar_double(linexpr0, coeffs[-1])
    for i in range(len(coeffs[:-1])):
        elina_linexpr0_set_coeff_scalar_double(linexpr0, dims[i], coeffs[i])
    return linexpr0

def getInfo(linexpr0):
    size = elina_linexpr0_size(linexpr0)
    coeffs = [elina_linexpr0_coeffref(linexpr0, i) for i in range(size)]
    const = elina_linexpr0_cstref(linexpr0)
    print("Coeffs:", coeffs)
    print("Constant:", const)
    return coeffs, const

def addDimension(linexpr0, dim):
    size = elina_linexpr0_size(linexpr0)
    elina_linexpr0_realloc(linexpr0, size + 1)
    elina_linexpr0_set_coeff_scalar_double(linexpr0, dim, -1)

    return linexpr0

def negLin(linexpr0):
    linexpr1 = elina_linexpr0_copy(linexpr0)
    size = elina_linexpr0_size(linexpr0)
    for i in range(size):
        coeff0 = elina_linexpr0_coeffref(linexpr0, i)
        coeff1 = elina_linexpr0_coeffref(linexpr1, i)
        elina_coeff_neg(coeff1, coeff0)
    const0 = elina_linexpr0_cstref(linexpr0)
    const1 = elina_linexpr0_cstref(linexpr1)
    elina_coeff_neg(const0, const1)
    elina_linexpr0_print(linexpr1, None)
    print_c("\n")
    return linexpr1

def createLinconsArr(coeffs, lb, ub):
    size = len(coeffs)
    num = 2
    lincons0_array = elina_lincons0_array_make(size*4)
    for i in range(size):
        linexpr0 = create_linexpr(coeffs[i])
        newDim = len(coeffs[i])-1+i
        addDimension(linexpr0, newDim)
        linexpr1 = negLin(linexpr0)
        lincons0_array.p[i*num].constyp = ElinaConstyp.ELINA_CONS_SUPEQ
        lincons0_array.p[i*num].linexpr0 = linexpr0
        lincons0_array.p[i*num+1].constyp = ElinaConstyp.ELINA_CONS_SUPEQ
        lincons0_array.p[i*num+1].linexpr0 = linexpr1
    for i in range(len(lb)):
        ub_coeff = create_linexpr([-1.0, ub[i]], [i])
        lb_coeff = create_linexpr([1.0, -lb[i]], [i])
        lincons0_array.p[size*2 + i*num].constyp = ElinaConstyp.ELINA_CONS_SUPEQ
        lincons0_array.p[size*2 + i*num].linexpr0 = lb_coeff
        lincons0_array.p[size*2 + i*num+1].constyp = ElinaConstyp.ELINA_CONS_SUPEQ
        lincons0_array.p[size*2 + i*num+1].linexpr0 = ub_coeff

    elina_lincons0_array_print(lincons0_array,None)
    print_c("\n")
    return lincons0_array

def createSimpleLinconsArr(coeffs, dims):
    size = len(coeffs)
    lincons0_array = elina_lincons0_array_make(size)
    for i in range(size):
        lincons0_array.p[i].constyp = ElinaConstyp.ELINA_CONS_SUPEQ
        lincons0_array.p[i].linexpr0 = create_linexpr(coeffs[i], dims[i])
    return lincons0_array


def doSomething(lincons0_array, dim, nbcons):
    man = opt_pk_manager_alloc(False)
    top = elina_abstract0_top(man,0,dim)
    o1 = elina_abstract0_meet_lincons_array(man,False,top, lincons0_array)

    dimension = elina_abstract0_dimension(man, o1)
    print("Dimention", dimension.intdim, dimension.realdim)
    for i in range(dimension.intdim + dimension.realdim):
        interval = elina_abstract0_bound_dimension(man, o1, i)
        print_c("Dimension {}:".format(i))
        elina_interval_fprint(cstdout, interval)
        elina_interval_free(interval)
        print_c("\n")
    
    join_expr = createSimpleLinconsArr([[1,0]], [[2]])
    elina_lincons0_array_print(join_expr,None)
    o2 = elina_abstract0_meet_lincons_array(man,False,o1, join_expr)
    dimension = elina_abstract0_dimension(man, o2)
    print("Dimention", dimension.intdim, dimension.realdim)
    for i in range(dimension.intdim + dimension.realdim):
        interval = elina_abstract0_bound_dimension(man, o2, i)
        print_c("Dimension {}:".format(i))
        elina_interval_fprint(cstdout, interval)
        print_c("\n")
        d = c_double()
        elina_double_set_scalar(d, interval.contents.inf, MpfrRnd.MPFR_RNDD)
        print("Py Down:", d.value)
        elina_double_set_scalar(d, interval.contents.sup, MpfrRnd.MPFR_RNDU)
        print("Py Up:", d.value)
        elina_interval_free(interval)

    return None

if __name__ == "__main__":
    linexpr0 = create_linexpr([-2, 2, 0])
    linexpr0 = addDimension(linexpr0, 2)
    linexpr1 = negLin(linexpr0)
    arr = createLinconsArr([[-2,2,1],[2,2,2]], [0,0], [2,1])
    doSomething(arr, 4, 0)
    