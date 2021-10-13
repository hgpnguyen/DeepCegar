import autograd.numpy as np
import multiprocessing
import os
from ctypes import *

clib = CDLL('../clib/clib.so')

class Poly():
    def __init__(self):
        self.lw, self.up = None, None
        self.le, self.ge = None, None


    def copy(self):
        new_poly = Poly()

        new_poly.lw = self.lw.copy()
        new_poly.up = self.up.copy()

        new_poly.le = None if self.le is None else self.le.copy()
        new_poly.ge = None if self.ge is None else self.ge.copy()


        return new_poly

    def back_substitute(self, lst_poly, get_ineq=False):
        no_neurons = len(self.lw)
        if no_neurons <= 100 or len(lst_poly) <= 2:
            lst_ge, lst_le = [], []
            for i in range(no_neurons):
                args = (i, self.le[i], self.ge[i], lst_poly)
                _, lw_i, up_i, lst_le_i, lst_ge_i = back_substitute0(args)
                self.lw[i], self.up[i] = lw_i, up_i
                print("List lei:", lst_le_i)
                print("List gei:", lst_ge_i)
                lst_le.append(lst_le_i[0])
                lst_ge.append(lst_ge_i[0])
                # get_ineq only happens at the last step
                # no_neurons in this case always be 1
            if get_ineq: lst_ge, lst_le
        else:
            lst_ge, lst_le = [None]*no_neurons, [None]*no_neurons
            clones = []

            for i in range(no_neurons):
                clones.append(lst_poly)

            num_cores = os.cpu_count()
            pool = multiprocessing.Pool(num_cores)
            zz = zip(range(no_neurons), self.le, self.ge, clones)
            for i, lw_i, up_i, lst_le_i, lst_ge_i in pool.map(back_substitute1, zz):
                self.lw[i], self.up[i] = lw_i, up_i
                lst_le.append(lst_le_i[0])
                lst_ge.append(lst_ge_i[0])
            pool.close()

        print("List le:", lst_le)
        print("List ge:", lst_ge)
        if get_ineq: return lst_le, lst_ge


def back_substitute0(args):
    idx, le_curr, ge_curr, lst_poly = args

    le_curr = np.array(le_curr, dtype=np.float64)
    ge_curr = np.array(ge_curr, dtype=np.float64)

    lst_le, lst_ge = [le_curr], [ge_curr]
    best_lw, best_up = -1e9, 1e9

    for k, e in reversed(list(enumerate(lst_poly))):
        no_e_ns = len(e.lw)

        max_le_curr = np.maximum(le_curr[:-1], 0)
        min_le_curr = np.minimum(le_curr[:-1], 0)

        max_ge_curr = np.maximum(ge_curr[:-1], 0)
        min_ge_curr = np.minimum(ge_curr[:-1], 0)

        max_le_n0id = np.array(np.nonzero(max_le_curr)[0], dtype=np.int32)
        min_le_n0id = np.array(np.nonzero(min_le_curr)[0], dtype=np.int32)

        max_ge_n0id = np.array(np.nonzero(max_ge_curr)[0], dtype=np.int32)
        min_ge_n0id = np.array(np.nonzero(min_ge_curr)[0], dtype=np.int32)
        print(max_ge_n0id)
        print(max_ge_curr, e.lw)
        lw, up = ge_curr[-1], le_curr[-1]

        if len(max_ge_n0id) > 0:
            lw += np.sum(max_ge_curr[max_ge_n0id] * e.lw[max_ge_n0id])
        if len(min_ge_n0id) > 0:
            lw += np.sum(min_ge_curr[min_ge_n0id] * e.up[min_ge_n0id])

        if len(max_le_n0id) > 0:
            up += np.sum(max_le_curr[max_le_n0id] * e.up[max_le_n0id])
        if len(min_le_n0id) > 0:
            up += np.sum(min_le_curr[min_le_n0id] * e.lw[min_le_n0id])

        best_lw = max(best_lw, lw)
        best_up = min(best_up, up)

        if k > 0:
            no_coefs = len(e.le[0])

            le = np.zeros([no_coefs], dtype=np.float64)
            ge = np.zeros([no_coefs], dtype=np.float64)

            le_curr_ptr = le_curr.ctypes.data_as(POINTER(c_double))
            ge_curr_ptr = ge_curr.ctypes.data_as(POINTER(c_double))

            e_le, e_ge = e.le, e.ge

            le_ptr = e_le.ctypes.data_as(POINTER(POINTER(c_double)))
            ge_ptr = e_ge.ctypes.data_as(POINTER(POINTER(c_double)))

            max_le_n0id_ptr = max_le_n0id.ctypes.data_as(POINTER(c_int))
            min_le_n0id_ptr = min_le_n0id.ctypes.data_as(POINTER(c_int))

            max_ge_n0id_ptr = max_ge_n0id.ctypes.data_as(POINTER(c_int))
            min_ge_n0id_ptr = min_ge_n0id.ctypes.data_as(POINTER(c_int))

            clib.array_mul_c.argtypes = (POINTER(POINTER(c_double)), POINTER(c_double), POINTER(c_int), c_int, c_int)
            clib.array_mul_c.restype = POINTER(c_double * no_coefs)

            clib.free_array.argtype = POINTER(c_double * no_coefs)

            result_ptr = clib.array_mul_c(le_ptr, le_curr_ptr, max_le_n0id_ptr, len(max_le_n0id), no_coefs)
            le += np.frombuffer(result_ptr.contents)
            clib.free_array(result_ptr)

            result_ptr = clib.array_mul_c(ge_ptr, le_curr_ptr, min_le_n0id_ptr, len(min_le_n0id), no_coefs)
            le += np.frombuffer(result_ptr.contents)
            clib.free_array(result_ptr)

            result_ptr = clib.array_mul_c(ge_ptr, ge_curr_ptr, max_ge_n0id_ptr, len(max_ge_n0id), no_coefs)
            ge += np.frombuffer(result_ptr.contents)
            clib.free_array(result_ptr)

            result_ptr = clib.array_mul_c(le_ptr, ge_curr_ptr, min_ge_n0id_ptr, len(min_ge_n0id), no_coefs)
            ge += np.frombuffer(result_ptr.contents)
            clib.free_array(result_ptr)

            le[-1] = le[-1] + le_curr[-1]
            ge[-1] = ge[-1] + ge_curr[-1]

            le_curr, ge_curr = le, ge

            lst_le.insert(0, le_curr)
            lst_ge.insert(0, ge_curr)

    return idx, best_lw, best_up, lst_le, lst_ge

def back_substitute1(args):
    idx, le_curr, ge_curr, lst_poly = args

    lst_le, lst_ge = [le_curr], [ge_curr]
    best_lw, best_up = -1e9, 1e9

    for k, e in reversed(list(enumerate(lst_poly))):
        no_e_ns = len(e.lw)
        lw, up = 0, 0

        if k > 0:
            no_coefs = len(e.le[0])

            le = np.zeros([no_coefs])
            ge = np.zeros([no_coefs])

            for i in range(no_e_ns):
                if le_curr[i] > 0:
                    up = up + le_curr[i] * e.up[i]
                    le = le + le_curr[i] * e.le[i]
                elif le_curr[i] < 0:
                    up = up + le_curr[i] * e.lw[i]
                    le = le + le_curr[i] * e.ge[i]

                if ge_curr[i] > 0:
                    lw = lw + ge_curr[i] * e.lw[i]
                    ge = ge + ge_curr[i] * e.ge[i]
                elif ge_curr[i] < 0:
                    lw = lw + ge_curr[i] * e.up[i]
                    ge = ge + ge_curr[i] * e.le[i]

            lw = lw + ge_curr[-1]
            up = up + le_curr[-1]

            le[-1] = le[-1] + le_curr[-1]
            ge[-1] = ge[-1] + ge_curr[-1]

            best_lw = max(best_lw, lw)
            best_up = min(best_up, up)

            le_curr, ge_curr = le, ge

            lst_le.insert(0, le_curr)
            lst_ge.insert(0, ge_curr)
        else:
            for i in range(no_e_ns):
                if le_curr[i] > 0:
                    up = up + le_curr[i] * e.up[i]
                elif le_curr[i] < 0:
                    up = up + le_curr[i] * e.lw[i]

                if ge_curr[i] > 0:
                    lw = lw + ge_curr[i] * e.lw[i]
                elif ge_curr[i] < 0:
                    lw = lw + ge_curr[i] * e.up[i]

            lw = lw + ge_curr[-1]
            up = up + le_curr[-1]

            best_lw = max(best_lw, lw)
            best_up = min(best_up, up)

    return idx, best_lw, best_up, lst_le, lst_ge