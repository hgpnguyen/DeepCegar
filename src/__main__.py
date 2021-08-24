import sys

sys.path.insert(0, '../elina/python_interface/')
sys.path.insert(0, '../os/deepg/code/')
import os
import numpy as np
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
from analyzer import Verified_Result
from eran import ERAN
from read_net_file import *
from ast import literal_eval
import tensorflow.compat.v1 as tf 
tf.disable_v2_behavior()
import csv
import time
from tqdm import tqdm
import argparse
from config import config
from colors import *



def str2bool(v):
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')



def isnetworkfile(fname):
    _, ext = os.path.splitext(fname)
    if ext not in ['.pyt', '.meta', '.tf','.onnx']:
        raise argparse.ArgumentTypeError('only .pyt, .tf, .onnx, and .meta formats supported')
    return fname



def normalize(image, means, stds, dataset, is_conv):
    if(dataset=='mnist'):
        for i in range(len(image)):
            image[i] = (image[i] - means[0])/stds[0]
    elif(dataset=='cifar10'):
        tmp = np.zeros(3072)
        for i in range(3072):
            tmp[i] = (image[i] - means[i % 3]) / stds[i % 3]

        if(is_conv):
            for i in range(3072):
                image[i] = tmp[i]
        else:
            count = 0
            for i in range(3072):
                image[i] = (tmp[i % 1024 + (i % 3) * 1024] - means[i % 3]) / stds[i % 3]



def normalize_poly(num_params, lexpr_cst, lexpr_weights, lexpr_dim, uexpr_cst, uexpr_weights, uexpr_dim, means, stds, dataset):
    if dataset == 'mnist' or dataset == 'fashion':
        for i in range(len(lexpr_cst)):
            lexpr_cst[i] = (lexpr_cst[i] - means[0]) / stds[0]
            uexpr_cst[i] = (uexpr_cst[i] - means[0]) / stds[0]
        for i in range(len(lexpr_weights)):
            lexpr_weights[i] /= stds[0]
            uexpr_weights[i] /= stds[0]
    else:
        for i in range(len(lexpr_cst)):
            lexpr_cst[i] = (lexpr_cst[i] - means[i % 3]) / stds[i % 3]
            uexpr_cst[i] = (uexpr_cst[i] - means[i % 3]) / stds[i % 3]
        for i in range(len(lexpr_weights)):
            lexpr_weights[i] /= stds[(i // num_params) % 3]
            uexpr_weights[i] /= stds[(i // num_params) % 3]



def denormalize(image, means, stds, dataset):
    if(dataset=='mnist'):
        for i in range(len(image)):
            image[i] = image[i]*stds[0] + means[0]
    elif(dataset=='cifar10'):
        count = 0
        tmp = np.zeros(3072)
        for i in range(1024):
            tmp[count] = image[count]*stds[0] + means[0]
            count = count + 1
            tmp[count] = image[count]*stds[1] + means[1]
            count = count + 1
            tmp[count] = image[count]*stds[2] + means[2]
            count = count + 1

        if(is_conv):
            for i in range(3072):
                image[i] = tmp[i]
        else:
            count = 0
            for i in range(1024):
                image[i] = tmp[count]
                count = count+1
                image[i+1024] = tmp[count]
                count = count+1
                image[i+2048] = tmp[count]
                count = count+1



def get_tests(dataset, geometric):
    if (dataset == 'acasxu'):
        specfile = '../data/acasxu/specs/acasxu_prop' + str(specnumber) + '_spec.txt'
        tests = open(specfile, 'r').read()
    # elif (dataset == 'test'):
    #     tests = [[0, 0.0, 0.0]]
    else:
        if geometric:
            csvfile = open('../deepg/code/datasets/{}_test.csv'.format(dataset), 'r')
        else:
            csvfile = open('../data/{}_test.csv'.format(dataset), 'r')
        tests = csv.reader(csvfile, delimiter=',')
    return tests

def read(text):
    if os.path.isfile(text):
        return open(text, 'r').readline()
    else:
        return text

def get_dataset(x_path, y_path):
    x0s = literal_eval(read(x_path))
    y0s = literal_eval(read(y_path))
    assert len(x0s) == len(y0s)
    
    for i in range(len(y0s)):
        x0s[i].insert(0, y0s[i])
    return x0s

def output_to_csv(filename, output_info):
    with open(filename, 'w') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(['Testcase', 'Verify result', 'time', 'Total number of task', 'Largest number of task'])
        writer.writerows(output_info)

def add_row_to_file(filename, output_info):
    out = []
    if not os.path.isfile(filename):
        out.append(['Testcase', 'Verify result', 'time', 'Total number of task', 'Largest number of task'])
    out.append(output_info)
    with open(filename, 'a') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerows(out)


def main(config):
    netname = config.netname
    filename, file_extension = os.path.splitext(netname)

    is_trained_with_pytorch = file_extension==".pyt"
    is_saved_tf_model = file_extension==".meta"
    is_pb_file = file_extension==".pb"
    is_tensorflow = file_extension== ".tf"
    is_onnx = file_extension == ".onnx"
    assert is_trained_with_pytorch or is_saved_tf_model or is_pb_file or is_tensorflow or is_onnx, "file extension not supported"

    epsilon = config.epsilon
    assert (epsilon >= 0) and (epsilon <= 1), "epsilon can only be between 0 and 1"

    domain = config.domain

    if not config.geometric:
        assert domain in ['powerset@box', 'powerset@zonotope', 'deepbox', 'refinebox', 'deepzono', 'refinezono', 'deeppoly', 'refinepoly'], "domain name can be either deepzono, refinezono, deeppoly or refinepoly"

    dataset = config.dataset
    assert dataset in ['mnist','cifar10','acasxu', 'test'], "only mnist, cifar10, and acasxu datasets are supported"

    specnumber = 9
    if(dataset=='acasxu' and (specnumber!=9)):
        print("currently we only support property 9 for acasxu")

    # if(dataset=='acasxu'):
    #     print("netname ", netname, " specnumber ", specnumber, " domain ", domain, " dataset ", dataset)
    # else:
    #     print("netname ", netname, " epsilon ", epsilon, " domain ", domain, " dataset ", dataset)
    assert config.attack_method in ['scipy', 'pgd', 'sgd']
    #### end of program argument manipulation

    #### start of loading the model from net file
    is_conv = False
    non_layer_operation_types = ['NoOp', 'Assign', 'Const', 'RestoreV2', 'SaveV2', 'PlaceholderWithDefault', 'IsVariableInitialized', 'Placeholder', 'Identity']
    if is_saved_tf_model or is_pb_file:
        print('#### is saved tf model or is pb file')
        netfolder = os.path.dirname(netname)
        tf.logging.set_verbosity(tf.logging.ERROR)

        sess = tf.Session()
        if is_saved_tf_model:
            saver = tf.train.import_meta_graph(netname)
            saver.restore(sess, tf.train.latest_checkpoint(netfolder+'/'))
        else:
            with tf.gfile.GFile(netname, "rb") as f:
                graph_def = tf.GraphDef()
                graph_def.ParseFromString(f.read())
                sess.graph.as_default()
                tf.graph_util.import_graph_def(graph_def, name='')
        ops = sess.graph.get_operations()
        last_layer_index = -1
        while ops[last_layer_index].type in non_layer_operation_types:
            last_layer_index -= 1
        eran = ERAN(sess.graph.get_tensor_by_name(ops[last_layer_index].name + ':0'), sess)
    else:
        if(dataset=='mnist'):
            num_pixels = 784
        elif (dataset=='cifar10'):
            num_pixels = 3072
        elif(dataset=='acasxu'):
            num_pixels = 5
        elif(dataset=='test'):
            num_pixels = 2
        if is_onnx:
            model, is_conv = read_onnx_net(netname)
            # this is to have different defaults for mnist and cifar10
        else:
            model, is_conv, means, stds = read_tensorflow_net(netname, num_pixels, is_trained_with_pytorch)
        # print('model:', model)
        eran = ERAN(model, is_onnx=is_onnx)
        

    if not is_trained_with_pytorch:
        if dataset == 'cifar10':
            means = [0.485, 0.456, 0.406]
            stds = [0.225, 0.225, 0.225]
        else:
            means = [0]
            stds = [1]
    if config.mean:
        means = config.mean
    if config.std:
        stds = config.std


    #### ignore some exmpales for now.
    if dataset=='acasxu':
        print ('acasxu omit here.')
        return 0
    elif config.geometric:
        print ('Geometric omit here.')
        return 0

    is_trained_with_pytorch = is_trained_with_pytorch or is_onnx
    correctly_classified_images = 0
    verified_images = 0
    verified_unsafe_images = 0
    verified_test = []
    verified_unsafe = []
    num_graphs = []
    output_infos = []
    if dataset:
        if not config.x_input_dataset:
            tests = get_tests(dataset, config.geometric)
        else:
            tests = get_dataset(config.x_input_dataset, config.y_input_dataset)
    
    start = config.start
    # tf.InteractiveSession().as_default()
    for i, test in enumerate(tests[start:100], start):
        num_graphs.append(len(tf.Session().graph._nodes_by_name.keys()))
        #if i not in [80]:
        #    continue
        if config.test_idx is not None and i != config.test_idx: continue
        # if config.from_test and i < config.from_test: continue
        if config.num_tests is not None and i >= config.num_tests: break

        if(dataset=='mnist') and not config.x_input_dataset:
            image= np.float64(test[1:len(test)])/np.float64(255)
            # print('image:', list(np.ndarray.flatten(image)))
        elif(dataset=='test') or config.x_input_dataset:
            image=np.float64(test[1:len(test)])
        else:
            if is_trained_with_pytorch:
                image= (np.float64(test[1:len(test)])/np.float64(255))
            else:
                image= (np.float64(test[1:len(test)])/np.float64(255)) - 0.5
                
        if i>0:
            print ('\n'*3)
        print(lgreen, '‣'*56, ' test ', i, ' ', '‣'*56, reset, sep='')
        # print('========= new test:', [int(i) for i in test][:40], " ===========")
        print('#'*37, 'concrete model evaluation', '#'*36)
        normalized_image = np.reshape(np.copy(image), (-1))
        if is_trained_with_pytorch:
            normalize(normalized_image, means, stds, dataset, is_conv)
        # print('========= normalized :', normalized_image, " ===========")
        # print('== input upper bound: ', np.ndarray.flatten(specUB)[:10], '...')
        target_label = int(test[0])
        print('==> target label: ', target_label)

        
        cstart = time.time()
        is_verified, dominant_class = eran.analyze_box(normalized_image, normalized_image, 'concrete', target_label=target_label)
        cend = time.time()
        if not is_verified:
            print("img",i,"not considered, correct_label", target_label, "classified label ", dominant_class)
            continue
        print("img", i, "concretely labeled correctly")
        correctly_classified_images +=1
        print("time ", cend - cstart, ' seconds\n\n')

        
        print('#'*38, 'abstract interpretation', '#'*37)
        sys.stdout.flush()
        perturbed_label = None
        if(dataset=='mnist'):
            specLB = np.clip(image - epsilon,0,1)
            specUB = np.clip(image + epsilon,0,1)
        elif(dataset=='test'):
            specLB = np.clip(image - epsilon,-1,1)
            specUB = np.clip(image + epsilon,-1,1)
        else:
            if(is_trained_with_pytorch):
                 specLB = np.clip(image - epsilon,0,1)
                 specUB = np.clip(image + epsilon,0,1)
            else:
                 specLB = np.clip(image-epsilon,-0.5,0.5)
                 specUB = np.clip(image+epsilon,-0.5,0.5)
        if(is_trained_with_pytorch):
            normalize(specLB, means, stds, dataset, is_conv)
            normalize(specUB, means, stds, dataset, is_conv)
        # print('image:', np.ndarray.flatten(image)[:30], '...')
        # print(' ε:', epsilon)
        # print('++ input lower bound:', list(np.ndarray.flatten(specLB))[:10], '...')
        # print('++ input upper bound:', list(np.ndarray.flatten(specUB))[:10], '...')
        sys.stdout.flush()

        start = time.time()
        is_verified, output_info = eran.analyze_box(specLB, specUB, domain, problem_id=i, use_abstract_attack=config.use_abstract_attack, attack_method=config.attack_method, use_abstract_refine=config.use_abstract_refine, target_label=target_label)
        end = time.time()
        output_infos.append([i, is_verified, end-start, *output_info])
        if config.output:
            add_row_to_file(config.output, [i, is_verified, end-start, *output_info])
        if is_verified == Verified_Result.Safe:
            print("img", i, "Verified.")
            verified_images += 1 
            verified_test.append(i)
        elif is_verified == Verified_Result.Unknow:
            print("img", i, "Failed (may false negative, as abstract interpretation is only sound)")
        else:
            print("img", i, "Verified as Unsafe")
            verified_unsafe_images += 1
            verified_unsafe.append(i)
        print(end - start, "seconds")

    #if config.output:
    #    output_to_csv(config.output, output_infos)
    print('analysis precision ', verified_images,'/ ', correctly_classified_images)
    print('analysis unsafe ', verified_unsafe_images,'/ ', correctly_classified_images)
    print('Verified images:', verified_test)
    print('Verified unsafe images:', verified_unsafe)
    print('Total task born:', np.sum(np.array(output_infos)[:,-1]), 'Max task born:', np.max(np.array(output_infos)[:,-1]))





if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='ERAN Example',  formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--netname', type=isnetworkfile, default=config.netname, help='the network name, the extension can be only .pyt, .tf and .meta')
    parser.add_argument('--epsilon', type=float, default=config.epsilon, help='the epsilon for L_infinity perturbation')
    # parser.add_argument('--specnumber', type=int, default=9, help='the property number for the acasxu networks')
    parser.add_argument('--domain', type=str, default=config.domain, help='the domain name can be either powerset@box, powerset@zonotope, deepbox, refinebox, deepzono, refinezono, deeppoly or refinepoly')
    parser.add_argument('--dataset', type=str, default=config.dataset, help='the dataset, can be either mnist, cifar10, or acasxu')
    parser.add_argument('--use_area_heuristic', type=str2bool, default=config.use_area_heuristic,  help='whether to use area heuristic for the DeepPoly ReLU approximation')
    parser.add_argument('--mean', nargs='+', type=float, default=config.mean, help='the mean used to normalize the data with')
    parser.add_argument('--std', nargs='+', type=float, default=config.std, help='the standard deviation used to normalize the data with')
    parser.add_argument('--data_dir', type=str, default=config.data_dir, help='data location')
    parser.add_argument('--num_params', type=int, default=config.num_params, help='Number of transformation parameters')
    parser.add_argument('--num_tests', type=int, default=config.num_tests, help='Number of images to test')
    # parser.add_argument('--from_test', type=int, default=config.from_test, help='Number of images to test')
    parser.add_argument('--test_idx', type=int, default=config.test_idx, help='Index to test')
    parser.add_argument('--debug', action='store_true', default=config.debug, help='Whether to display debug info')
    parser.add_argument('--attack', action='store_true', default=config.attack, help='Whether to attack, old from eran...')
    parser.add_argument('--use_abstract_attack', action='store_true', default=config.attack, help='Whether to attack the abstract domain for fast checking')
    parser.add_argument('--attack_method', type=str, default=config.attack_method, help='abstract attack method, either "scipy" or "pgd"')
    parser.add_argument('--use_abstract_refine', action='store_true', default=config.attack, help='Whether to refine the abstract domain to make analysis more complete')
    parser.add_argument('--x_input_dataset', type=str, default=config.x_input_dataset, help='Input x dateset location')
    parser.add_argument('--y_input_dataset', type=str, default=config.y_input_dataset, help='Input y dateset location')
    parser.add_argument('--output', type=str, default=config.output, help='Output folder')

    # Logging options
    parser.add_argument('--logdir', type=str, default=None, help='Location to save logs to. If not specified, logs are not saved and emitted to stdout')
    parser.add_argument('--logname', type=str, default=None, help='Directory of log files in `logdir`, if not specified timestamp is used')

    args = parser.parse_args()
    for k, v in vars(args).items():
        setattr(config, k, v)
    config.json = vars(args)


    mnist_relu_model = ['3_10', '3_20', '3_30', 
                        '4_10', '4_20', '4_30',
                        '5_10', '5_20',
                        '3_40', '5_30',  '4_40', '5_40', '3_50', '4_50', '5_50',
                        '6_100', '9_200', '4_1024']
    #data_folder = '../benchmark/mnist_challenge/x_y/'
    model_folder = '../benchmark/cegar/nnet/'
    output_folder = config.output
    config.start = 77
    for m in mnist_relu_model[14:15]:
        model_name = 'mnist_relu_' + m
        config.netname = '{f}{model}/original/{model}.tf'.format(f=model_folder, model=model_name)
        assert config.netname, 'a network has to be provided for analysis.'
        if config.output:
            eps = str(config.epsilon).split('.')
            info = 'refine{}_k1_{}_{}'.format(config.domain, eps[0], eps[-1]) if config.use_abstract_attack else '{}_{}_{}'.format(config.domain, eps[0], eps[-1])
            config.output = '{}/{}_{}.csv'.format(output_folder, model_name, info)
        start = time.time()
        main(config)
        end = time.time()
        tf.reset_default_graph()
        config.start = 0
        print("Total run time:", end-start, "seconds")
