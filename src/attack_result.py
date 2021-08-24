'''
@author: Li Jiaying
'''


class AttackResult(object):
    def __init__(self):
        self.success = False
        self.x = None
        self.distance = 9999.9
        self.grad = 0.0
        self.n_eval = 0
        self.n_init = 0
        self.x_set = []
        self.distance_set = []
        self.gene = None