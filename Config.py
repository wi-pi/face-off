import os, cv2
import tensorflow as tf
import numpy as np
from timeit import default_timer as timer


class Benchmark:
    """Stores the start and end times of execution for performance metrics."""
    def __init__(self):
        self.start = {}
        self.end = {}

    def mark(self, message=''):
        """
        Stores the start or end time depending on which call.
        Prints the execution time.
        Usage: 
            Benchmark.mark('message to print')
            Code to benchmark...
            Benchmark.mark('message to print')

        Keyword arguments:
        message -- a key for the dict and message to print
        """
        if message not in self.start:
            self.start[message] = -1
            self.end[message] = -1
        if self.start[message] is -1:
            self.start[message] = timer()
        else:
            if self.end[message] is -1:
                self.end[message] = timer()
            print('{message:{fill}{align}{width}}-{time}'.format(message=message,
                fill='-', align='<', width=50,
                time=(self.end[message] - self.start[message])))
            self.start[message] = -1
            self.end[message] = -1


S3_DIR = ''
S3_BUCKET = ''
ROOT = os.path.abspath('.')
ALIGN_96_DIR = 'data/celeb96'
ALIGN_160_DIR = 'data/celeb160'
VGG_VALIDATION_DIR = 'data/small-vgg-align-validate'
VGG_ALIGN_160_DIR = 'data/small-vgg-align-validate'
TEST_DIR = 'data/test_imgs'
FULL_DIR = 'data/celeb'
VGG_TEST_DIR = 'data/test_imgs/VGG'
OUT_DIR = 'data/new_adv_imgs'
API_DIR = 'data/new_api_results'
BM = Benchmark()

CASIA_MODEL_PATH = 'weights/facenet_casia.h5'
VGGSMALL_MODEL_PATH = 'weights/small_facenet_center.h5'
VGGADV_MODEL_PATH = 'weights/facenet_vggsmall.h5'
CENTER_MODEL_PATH = 'weights/facenet_keras_center_weights.h5'
TRIPLET_MODEL_PATH = 'weights/facenet_keras_weights.h5'
NAMES = ['barack', 'bill', 'jenn', 'leo', 'mark', 'matt', 'melania', 'meryl',
         'morgan', 'taylor']
API_PEOPLE = ['barack', 'leo', 'matt', 'melania', 'morgan', 'taylor']
PAIRS = {'barack': 'morgan', 'mark': 'bill', 'matt': 'bill', 'taylor': 'jenn',
        'melania': 'jenn', 'jenn': 'melania', 'bill': 'barack', 'morgan':
        'bill', 'leo': 'bill', 'meryl': 'jenn'}

# Adversarial training, VGGFace2
# SOURCES=['n000636', 'n001370', 'n001513', 'n002140', 'n002537', 'n004534',
#          'n005374', 'n005789', 'n006421', 'n007862', 'n001000', 'n001374',
#          'n001632', 'n002222', 'n003222', 'n005081', 'n005674', 'n005877',
#          'n007092', 'n007892', 'n001100', 'n001421', 'n001638', 'n002513',
#          'n003242', 'n005089', 'n005677', 'n006220', 'n007562', 'n008638',
#          'n001270', 'n001431', 'n001892', 'n002531', 'n003542', 'n005140',
#          'n005760', 'n006221', 'n007634', 'n009000', 'n001292', 'n001433',
#          'n002100', 'n002533', 'n003562', 'n005160', 'n005780', 'n006270',
#          'n007638', 'n009270']
# TARGETS=['n009000', 'n001292', 'n001433', 'n002100', 'n002533', 'n003562',
#          'n005160', 'n005780', 'n006270', 'n007638', 'n009270', 'n001892',
#          'n002531', 'n003542', 'n005140', 'n005760', 'n006221', 'n007634',
#          'n003242', 'n005089', 'n005677', 'n006220', 'n007562', 'n008638',
#          'n001270', 'n001431', 'n001374', 'n001632', 'n002222', 'n003222',
#          'n005081', 'n005674', 'n005877', 'n007092', 'n007892', 'n001100',
#          'n001421', 'n001638', 'n002513', 'n000636', 'n001370', 'n001513',
#          'n002140', 'n002537', 'n004534', 'n005374', 'n005789', 'n006421',
#          'n007862', 'n001000']

# Adversarial inference, VGGFace2
SOURCES=['n000082', 'n001302', 'n001976', 'n004240', 'n005137', 'n007368',
         'n008989', 'n009294', 'n001174', 'n001836', 'n003430', 'n004449',
         'n007261', 'n008932', 'n009225']
TARGETS=['n009225', 'n008932', 'n007261', 'n004449', 'n003430', 'n001836',
         'n001174', 'n009294', 'n008989', 'n007368', 'n005137', 'n004240',
         'n001976', 'n001302', 'n000082']


def string_to_bool(arg):
    """Converts a string into a returned boolean."""
    if arg.lower() == 'true':
        arg = True
    elif arg.lower() == 'false':
        arg = False
    else:
        raise ValueError('ValueError: Argument must be either "true" or "false".')
    return arg


def set_gpu(gpu):
    """Configures CUDA environment variable and returns tensorflow GPU config."""
    os.environ['CUDA_VISIBLE_DEVICES'] = gpu
    tf_config = tf.ConfigProto()
    tf_config.gpu_options.allow_growth = True
    return tf_config


def set_parameters(api_name='',
                   targeted_flag='true',
                   tv_flag='false',
                   hinge_flag='true',
                   cos_flag='false',
                   interpolation='bilinear',
                   model_type='large',
                   loss_type='triplet',
                   dataset_type='vgg',
                   target_model='large',
                   target_loss='center',
                   target_dataset='VGG',
                   attack='CW',
                   norm='2',
                   epsilon=0.1,
                   iterations=20,
                   binary_steps=5,
                   learning_rate=0.01,
                   epsilon_steps=0.01,
                   init_const=0.3,
                   mean_loss='embeddingmean',
                   batch_size=-1,
                   margin=15.0,
                   amplification=6.0,
                   granularity='normal',
                   whitebox_target=False,
                   pair_flag='false'):
    """
    Initializes params dictionary to be used in most functions.

    Keyword arguments:
    api_name -- API to evaluate against (azure, awsverify, facepp)
    targeted_flag -- true: use targeted attack
    tv_flag -- true: use tv loss
    hinge_flag -- true: use hinge loss (defined in paper)
    cos_flag -- true: use cosine similarity along with l2 norm
    interpolation -- type of interpolation to use in resizing delta
    model_type -- input size used in training model (small, large)
    loss_type -- loss type used in training model (center, triplet)
    dataset_type -- dataset used in training model (vgg, casia, vggsmall, vggadv)
    target_model -- target model size (whitebox transferability eval)
    target_loss -- target loss type (whitebox transferability eval)
    target_dataset -- target dataset type (whitebox transferability eval)
    attack -- attack type to use (CW, PGD)
    norm -- attack loss norm (2, inf)
    epsilon -- PGD epsilon upper bound
    iterations -- number of epochs for CW and PGD
    binary_steps -- number of outer binary search steps in CW
    learning_rate -- learning rate to use in attack
    epsilon_steps -- epsilon update value
    init_const -- initial CW constant
    mean_loss -- whether to use mean of embeddings or non-mean loss (embeddingmean, embedding)
    batch_size -- batch size used in attack (embedding: must be 1)
    margin -- margin or kappa value used in attack
    amplification -- amplification or alpha value used in amplifying perturbation
    granularity -- granularity of intervals for margin and amplification values
        (fine, normal, coarse, coarser, coarsest, single, fine-tuned, coarse-single, api-eval)
    whitebox_target -- true: using target model for whitebox transferability evaluation
    pair_flag -- true: use Config.PAIRS to determine source-target pairs
    """
    
    params = {}
    params['model_type'] = model_type
    params['loss_type'] = loss_type
    params['dataset_type'] = dataset_type
    params['target_model'] = target_model
    params['target_loss'] = target_loss
    params['target_dataset'] = target_dataset
    params['attack'] = attack
    params['norm'] = norm
    params['epsilon'] = epsilon
    params['iterations'] = iterations
    params['binary_steps'] = binary_steps
    params['learning_rate'] = learning_rate
    params['epsilon_steps'] = epsilon_steps
    params['init_const'] = init_const
    params['mean_loss'] = mean_loss
    params['batch_size'] = batch_size
    params['test_dir'] = TEST_DIR
    params['full_dir'] = FULL_DIR
    params['whitebox_target'] = whitebox_target
    params['targeted_flag'] = string_to_bool(targeted_flag)
    params['tv_flag'] = string_to_bool(tv_flag)
    params['hinge_flag'] = string_to_bool(hinge_flag)
    params['cos_flag'] = string_to_bool(cos_flag)
    params['pair_flag'] = string_to_bool(pair_flag)
    params['api_name'] = api_name

    if model_type == 'small' and loss_type == 'center':
        params['pixel_max'] = 1.0
        params['pixel_min'] = -1.0
    else:
        params['pixel_max'] = 1.0
        params['pixel_min'] = 0.0

    if dataset_type == 'vggsmall' and not whitebox_target:
        params['align_dir'] = VGG_ALIGN_160_DIR
        params['test_dir'] = VGG_TEST_DIR
    elif model_type == 'large' or dataset_type == 'casia':
        params['align_dir'] = ALIGN_160_DIR
    elif model_type == 'small':
        params['align_dir'] = ALIGN_96_DIR
    else:
        ValueError('ValueError: Argument must be either "small" or "large".')
    
    if interpolation == 'nearest':
        params['interpolation'] = cv2.INTER_NEAREST
    elif interpolation == 'bilinear':
        params['interpolation'] = cv2.INTER_LINEAR
    elif interpolation == 'bicubic':
        params['interpolation'] = cv2.INTER_CUBIC
    elif interpolation == 'lanczos':
        params['interpolation'] = cv2.INTER_LANCZOS4
    elif interpolation == 'super':
        ValueError('ValueError: Super interpolation not yet implemented.')
    else:
        raise ValueError('ValueError: Argument must be of the following, [nearest, bilinear, bicubic, lanczos, super].')

    if granularity == 'fine':
        params['margin_list'] = np.arange(0.0, margin, margin / 20.0)
        params['amp_list'] = np.arange(1.0, amplification, 0.2)
    elif granularity == 'normal':
        params['margin_list'] = np.arange(0.0, margin, margin / 10.0)
        params['amp_list'] = np.arange(1.0, amplification, 0.5)
    elif granularity == 'coarse':
        params['margin_list'] = np.arange(0.0, margin, margin / 5.0)
        params['amp_list'] = np.arange(1.0, amplification, 1.0)
    elif granularity == 'coarser':
        params['margin_list'] = np.arange(0.0, margin, margin / 3.0)
        params['amp_list'] = np.arange(1.0, amplification, 0.2)
    elif granularity == 'coarsest':
        params['margin_list'] = np.arange(0.0, margin, margin / 3.0)
        params['amp_list'] = np.arange(1.0, amplification, 1.0)
    elif granularity == 'single':
        params['margin_list'] = np.array([margin])
        params['amp_list'] = np.array([amplification])
    elif granularity == 'fine-tuned':
        params['margin_list'] = np.arange(10.0, margin, 1.0)
        params['amp_list'] = np.arange(1.0, amplification, 0.2)
    elif granularity == 'coarse-single':
        params['margin_list'] = np.arange(0.0, margin, margin / 3.0)
        params['amp_list'] = np.array([1.0])
    elif granularity == 'api-eval':
        params['margin_list'] = np.arange(0.0, margin, margin / 3.0)
        params['amp_list'] = np.arange(1.0, amplification, 0.8)
    else:
        raise ValueError('ValueError: Argument must be of the following, [fine, normal, coarse, coarser, single].')

    if params['hinge_flag']:
        params['attack_loss'] = 'hinge'
    else:
        params['attack_loss'] = 'target'
    if not params['targeted_flag']:
        params['attack_loss'] = 'target'
    if norm == 'inf':
        norm_name = 'i'
    else:
        norm_name = '2'
    if params['tv_flag']:
        tv_name = '_tv'
    else:
        tv_name = ''
    if params['cos_flag']:
        cos_name = '_cos'
    else:
        cos_name = ''

    params['model_name'] = '{}_{}'.format(model_type, loss_type)
    if dataset_type == 'casia' or dataset_type == 'vggsmall':
        params['model_name'] = dataset_type
    params['target_model_name'] = '{}_{}_{}'.format(target_model, target_loss, target_dataset)
    params['attack_name'] = '{}_l{}{}{}'.format(attack.lower(), norm_name, tv_name, cos_name)
    params['directory_path'] = os.path.join(ROOT,
                                            OUT_DIR,
                                            params['attack_name'],
                                            params['model_name'],
                                            '{}_loss/full'.format(params['attack_loss']))
    params['directory_path_crop'] = os.path.join(ROOT,
                                                 OUT_DIR,
                                                 params['attack_name'],
                                                 params['model_name'],
                                                 '{}_loss/crop'.format(params['attack_loss']))
    params['directory_path_npz'] = os.path.join(ROOT,
                                                OUT_DIR,
                                                params['attack_name'],
                                                params['model_name'],
                                                '{}_loss/npz'.format(params['attack_loss']))
    params['api_path'] = os.path.join(ROOT,
                                      API_DIR,
                                      params['attack_name'],
                                      params['model_name'],
                                      '{}_loss/npz'.format(params['attack_loss']))
    if params['mean_loss'] == 'embedding':
        params['directory_path'] += '_mean'
        params['directory_path_crop'] += '_mean'
        params['directory_path_npz'] += '_mean'
        params['api_path'] += '_mean'

    return params
