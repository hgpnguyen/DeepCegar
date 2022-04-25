'''
@author: Adrian Hoffmann
'''

from attack_domain import AbstractDomainAttack
from tensorflow_translator import *
from onnx_translator import *
from optimizer import *
from analyzer import *
from executor import *
from layers import *
from network_property import NetworkProperty

global_problem_id = 0

class ERAN:
    def __init__(self, model, session=None, is_onnx = False):
        """
        This constructor takes a reference to a TensorFlow Operation, TensorFlow Tensor, or Keras model. 
        The two TensorFlow functions graph_util.convert_variables_to_constants and 
            graph_util.remove_training_nodes will be applied to the graph to cleanse it of any nodes that are linked to training.
        In the resulting graph there should only be tf.Operations left that have 
            one of the following types [Const, MatMul, Add, BiasAdd, Conv2D, Reshape, MaxPool, Placeholder, Relu, Sigmoid, Tanh]
        If the input should be a Keras model we will ignore operations with type 
            Pack, Shape, StridedSlice, and Prod such that the Flatten layer can be used.
        
        Arguments
        ---------
        model : tensorflow.Tensor or tensorflow.Operation or tensorflow.python.keras.engine.sequential.Sequential or 
                keras.engine.sequential.Sequential
            if tensorflow.Tensor: model.op will be treated as the output node of the TensorFlow model. 
                Make sure that the graph only contains supported operations after applying graph_util.
                convert_variables_to_constants and graph_util.remove_training_nodes with [model.op.name] as output_node_names
            if tensorflow.Operation: model will be treated as the output of the TensorFlow model. 
                Make sure that the graph only contains supported operations after applying graph_util.
                convert_variables_to_constants and graph_util.remove_training_nodes with [model.op.name] as output_node_names
            if tensorflow.python.keras.engine.sequential.Sequential: x = model.layers[-1].output.op.inputs[0].op will be treated 
                as the output node of the Keras model. Make sure that the graph only contains supported operations after applying 
                graph_util.convert_variables_to_constants and graph_util.remove_training_nodes with [x.name] as output_node_names
            if keras.engine.sequential.Sequential: x = model.layers[-1].output.op.inputs[0].op will be treated as the output node 
                of the Keras model. Make sure that the graph only contains supported operations after applying graph_util.
                convert_variables_to_constants and graph_util.remove_training_nodes with [x.name] as output_node_names
        session : tf.Session
            session which contains the information about the trained variables. If session is None the code will 
            take the Session from tf.get_default_session(). If you pass a keras model you don't have to provide a session, 
            this function will automatically get it.
        """
        if is_onnx:
            translator = ONNXTranslator(model)
        else:
            translator = TFTranslator(model, session)
        operations, resources = translator.translate()
        self.optimizer  = Optimizer(operations, resources)
        self.concrete_analyzer = None
        self.abstract_analyzer = None
        self.abstract_deepbox_analyzer = None
        self.abstract_deepzono_analyzer = None
        self.abstract_deeppoly_analyzer = None
        self.abstract_powerset_analyzer = None
        self.abstract_attacker = None
        # self.powerset_box_analyzer = None
        # self.powerset_zonotope_analyzer = None
        self.executor = None


    def get_concrete_executor(self):
        if self.executor is None:
            execute_list, output_info = self.optimizer.get_concrete()
            self.executor = Executor(execute_list)
        return self.executor


    def get_output_size(self):
        exe = self.get_concrete_executor()
        return exe.ir_list[-1].output_length
    
    def get_abstract_attacker(self):
        if self.abstract_attacker is None:
            self.abstract_attacker = AbstractDomainAttack(self.get_concrete_executor())
        return self.abstract_attacker


    def get_abstract_analyzer(self, domain, specnumber, specLB, specUB):
        # assert domain in ['powerset@box', 'powerset@zonotope', 'deepbox', 'refinebox', 'deepzono', 'refinezono', 'deeppoly', 'refinepoly'], "domain is invalid, must be 'deepbox', 'deepzono' or 'deeppoly'"
        # print('Get/Init the abstract domain analyzer')
        # print('domain: ', domain)
        specLB = np.reshape(specLB, (-1,))
        specUB = np.reshape(specUB, (-1,))
        nn = Layers(specLB, specUB)
        if domain == 'deepzono' or domain == 'refinezono':
            #if self.abstract_deepzono_analyzer is None:
            execute_list, output_info = self.optimizer.get_deepzono(nn,specLB, specUB)
            self.abstract_deepzono_analyzer = Analyzer(execute_list, nn, domain, specnumber)
            return self.abstract_deepzono_analyzer
        elif domain == 'deeppoly' or domain == 'refinepoly':
            execute_list, output_info = self.optimizer.get_deeppoly(nn,specLB, specUB)
            self.abstract_deeppoly_analyzer = Analyzer(execute_list, nn, domain, specnumber)
            return self.abstract_deeppoly_analyzer
        else:
            assert 0
    

    
    def analyze_box(self, specLB, specUB, domain, specnumber=0, problem_id=None, use_abstract_attack=False, attack_method='scipy', use_abstract_refine=False, target_label=-1):
        """
        This function runs the analysis with the provided model and session from the constructor, 
        the box specified by specLB and specUB is used as input. Currently we have three domains, 
        'deepzono', 'refinezono' and 'deeppoly'.
        
        Arguments
        ---------
        specLB : numpy.ndarray
            ndarray with the lower bound of the input box
        specUB : numpy.ndarray
            ndarray with the upper bound of the input box
        domain : str
            either 'concrete', 'powerset@box', 'powerset@zonotope', 'deepbox', 'refinebox', 'deepzono', 'refinezono', 'deeppoly', or 'refinepoly', decides which set of abstract transformers is used.
            
        Return
        ------
        dominant_class : int
            if the analysis is succesfull (it could prove robustness for this box) then the index of the class that dominates is returned
            if the analysis couldn't prove robustness then -1 is returned
        """
        # print('domain is', domain)
        assert domain in ['concrete', 'powerset@box', 'powerset@zonotope', 'deepbox', 'refinebox', 'deepzono', 'refinezono', 'deeppoly', 'refinepoly'], "domain isn't valid, must be 'concrete', 'powersete@box', 'powersete@zonotope', 'deepbox', 'deepzono' or 'deeppoly'"
        is_concrete = (domain == 'concrete') or np.array_equal(specLB, specUB)
        
        ############## concrete propagation ##################
        if is_concrete:
            executor = self.get_concrete_executor()
            dominant_class, nb = executor.forward(specLB)
            # print('** output vector: ', np.reshape(nb, (-1)).tolist())
            # print('** output label: ', label)
            is_verified = (dominant_class == target_label)
            return is_verified, dominant_class

        ############## abstract propagation ##################
        analyzer = self.get_abstract_analyzer(domain, specnumber, specLB, specUB)
        analyzer.set_target_label(target_label)
        if use_abstract_attack:
            # print('use abstract domain attack')
            nr_labels = self.get_output_size()
            target_property = NetworkProperty(nr_labels)
            target_property.set_robust(target_label)
            analyzer.equip_with_concrete_executor(self.get_concrete_executor())
            analyzer.equip_with_target_property(target_property)
            analyzer.equip_with_abstract_attacker(self.get_abstract_attacker())
            
        if problem_id == None:
            global global_problem_id
            problem_id = '@'+str(global_problem_id)
            global_problem_id += 1
        
        is_verified, output_info = analyzer.analyze(specLB, specUB, pid=str(problem_id), use_abstract_attack=use_abstract_attack, attack_method=attack_method, use_abstract_refine=use_abstract_refine)
        return is_verified, output_info
