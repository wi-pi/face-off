import os
import re
import argparse
import numpy as np
import tensorflow as tf

# from models.inception_resnet_v1 import *
from models.inception_big import *

config = tf.ConfigProto()
config.gpu_options.allow_growth = True

ROOT = os.path.abspath('.')

npy_weights_dir = os.path.join(ROOT, 'weights/npy/')
weights_dir = os.path.join(ROOT, 'weights/')
model_dir = os.path.join(ROOT, 'models/')


# regex for renaming the tensors to their corresponding Keras counterpart
re_repeat = re.compile(r'Repeat_[0-9_]*b')
re_block8 = re.compile(r'Block8_[A-Za-z]')

def get_filename(key):
    filename = str(key)
    filename = filename.replace('/', '_')
    filename = filename.replace('InceptionResnetV1_', '')

    # remove "Repeat" scope from filename
    filename = re_repeat.sub('B', filename)

    if re_block8.match(filename):
        # the last block8 has different name with the previous 5 occurrences
        filename = filename.replace('Block8', 'Block8_6')

    # from TF to Keras naming
    filename = filename.replace('_weights', '_kernel')
    filename = filename.replace('_biases', '_bias')

    return filename + '.npy'


def extract_tensors_from_checkpoint_file(filename, output_folder):
    reader = tf.train.NewCheckpointReader(filename)

    for key in reader.get_variable_to_shape_map():
        # not saving the following tensors
        if key == 'global_step':
            continue
        if 'AuxLogit' in key:
            continue

        # convert tensor name into the corresponding Keras layer weight name and save
        path = os.path.join(output_folder, get_filename(key))
        arr = reader.get_tensor(key)
        np.save(path, arr)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--weights-path', type=str, help='path to the tf model weights')
    parser.add_argument('--ckpt-num', type=str, help='checkpoint epoch number')
    parser.add_argument('--weights-outfile', type=str, help='output path for the keras weights')
    parser.add_argument('--model-outfile', type=str, help='output path for the keras model')
    args = parser.parse_args()
    weights_path = args.weights_path
    ckpt_num = args.ckpt_num
    weights_outfile = args.weights_outfile
    model_outfile = args.model_outfile
    tf_model_dir = os.path.join(ROOT, 'weights/{}/'.format(weights_path))

    with tf.Session(config=config) as sess:

        extract_tensors_from_checkpoint_file('{}model-{}.ckpt-{}'.format(tf_model_dir, weights_path, ckpt_num), npy_weights_dir)


        model = InceptionResNetV1()
        # model.summary()

        print('Loading numpy weights from', npy_weights_dir)
        for layer in model.layers:
            if layer.weights:
                weights = []
                for w in layer.weights:
                    weight_name = os.path.basename(w.name).replace(':0', '')
                    weight_file = layer.name + '_' + weight_name + '.npy'
                    weight_arr = np.load(os.path.join(npy_weights_dir, weight_file))
                    weights.append(weight_arr)
                layer.set_weights(weights)

        print('Saving weights...')
        model.save_weights(os.path.join(weights_dir, weights_outfile))
        print('Saving model...')
        model.save(os.path.join(model_dir, model_outfile))
