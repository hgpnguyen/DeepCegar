import sys
sys.path.insert(0, '../elina/python_interface/')
from executor import *
from causality import Causality
from optimizer import *

from tensorflow_translator import *
from read_net_file import *


def testCausility(causality) -> None:
    x0s, y = causality.gen_x0()
    #print(np.reshape(x0s, (-1, x0s.shape[0])))
    #print(np.reshape(y, (-1, y.shape[0])))
    mie = causality.get_ie(0, 2, 0, -4, 2)
    print(mie)
    print(causality.get_ie(0, 2, 1, 0, 1))


def main():
    netname = "../nets/spec.tf"
    num_pixels = 2
    model, is_conv, means, stds = read_tensorflow_net(netname, num_pixels, False)
    translator = TFTranslator(model, None)
    operations, resources = translator.translate()
    optimizer = Optimizer(operations, resources)
    execute_list, _ = optimizer.get_concrete()
    executor = Executor(execute_list)
    causality = Causality(executor, ([-1, -1], [1, 1]))
    testCausility(causality)

if __name__ == "__main__":
    main()