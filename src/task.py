'''
@author: LI Jiaying
'''
import sys

from elina_abstract0 import elina_abstract0_add_dimensions, elina_abstract0_free

gid = 0

class Task:
    def __init__(self, pid, tid, start_ir, element, nn, hid):
        # if nn is None:
        #     nn = Layers()
        global gid
        gid+=1
        self.tid = str(tid)
        self.pid = pid
        self.element = element
        self.start_ir = start_ir
        self.nn = nn
        self.hid = str(hid)
        
    
    @staticmethod
    def create(pid, tid, start_ir, element, nn, parent_hid, partition_no):
        hid = parent_hid + '⡇' + str(partition_no)
        created = Task(pid, tid, start_ir, element, nn, hid)
        return created
    
    
    def __str__(self):
        return 'P' + self.pid + 'T' + self.tid

    
    def get_id(self):
        return 'P' + self.pid + 'T' + self.tid

    
    def get_hid(self):
        return self.hid

    def destroy(self, man):
        elina_abstract0_free(man, self.element)