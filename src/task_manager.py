'''
@author: LI Jiaying
'''
import sys
from task import *


class TaskManager:
    def __init__(self):
        self.tasks = None
        self.task_sizes = None
        self.limit = 0
       
       
    def init(self, pid, nr_layers):
        self.cid = 0
        self.pid = pid
        self.tasks = [[] for i in range(nr_layers)]
        self.task_sizes = [0]*nr_layers
        self.last_task = None
        self.total_size = 0
        self.front = None
        self.rear = None
        self.largest_size = 0
        
        
    def add_task(self, start_ir, element, nn, parent_hid, partition_no, neurons, relu_groups):
        self.cid += 1
        self.last_task = Task.create(self.pid, self.cid, start_ir, element, nn, parent_hid, partition_no, neurons, relu_groups)
        self.tasks[start_ir].append(self.last_task)
        self.task_sizes[start_ir] += 1
        self.total_size += 1
        self.largest_size = max(self.largest_size, self.total_size)
        #if self.total_size > 1024:
        #    print('Too many tasks (size>=', self.total_size, ') spawned. System exits.', sep='')
        #    sys.exit(-1)
        if self.front is None or start_ir<self.front:
            self.front = start_ir
        if self.rear is None or start_ir>self.rear:
            self.rear = start_ir
    
    
    def pop_task(self, start_ir=None):
        if start_ir is None:
            start_ir = self.rear
        if start_ir<self.front or start_ir>self.rear or self.task_sizes[start_ir]<=0:
            return None
        task = self.tasks[start_ir].pop(0)
        self.task_sizes[start_ir] -= 1
        self.total_size -= 1
        if self.total_size == 0:
            self.front = self.rear = None
        elif self.task_sizes[start_ir] == 0:
            if start_ir == self.front:
                while self.task_sizes[self.front]==0:
                    self.rear += 1
            if start_ir == self.rear:
                while self.task_sizes[self.rear]==0:
                    self.rear -= 1
        return task
    
    
    def size(self):
        return self.total_size
    
    def get_num_all_task(self):
        return self.cid

    def get_largest_size(self):
        return self.largest_size
    
    def checkLimit(self, layer_idx):
        return self.task_sizes[layer_idx] < self.limit
    
    def checkLimit(self, layer_idx, k):
        return self.task_sizes[layer_idx] < 2**k-1
    
    def destroy(self, man):
        for i in self.tasks:
            for j in i:
                j.destroy(man)

    def setLimit(self, k):
        self.limit = 2**k-1
    
    def __str__(self):
        return '[Problem:' + self.pid + '] layer_tasks:' + str(self.task_sizes) + ' total_size:' + str(self.total_size)