import imageio
import numpy as np 
import os
import Config
import tensorflow as tf
from utils.crop import apply_delta, crop_face, read_face_from_aligned
from utils.dets.detect_face import create_mtcnn
from models.face_models import get_model


def set_bounds(params):
    """
    Description

    Keyword arguments:
    """
    
    if params['model_type'] == 'small' and params['loss_type'] == 'center':
        pixel_max = 1.0
        pixel_min = -1.0
    else:
        pixel_max = 1.0
        pixel_min = 0.0
    return pixel_max, pixel_min


def transpose_back(params,
                   adv,
                   face):
    """
    Description

    Keyword arguments:
    """
    
    if len(adv.shape) == 4:
        adv = adv[0]
    if params['model_type'] == 'small':
        if params['loss_type'] == 'center':
            adv_new_img = (adv + 1.0)/2.0 # scale to [0,1]
            adv_new_img = adv_new_img[::-1,...] # BGR to RGB
            adv_new_img = np.transpose(adv_new_img, (1,2,0)) #CHW to HWC
            face_new = (face + 1.0)/2.0
            face_new = face_new[::-1,...]
            face_new = np.transpose(face_new, (1,2,0))
        elif params['loss_type'] == 'triplet':
            adv_new_img = np.transpose(adv, (1,2,0))
            face_new = np.transpose(face, (1,2,0))
    elif params['model_type'] == 'large':
        adv_new_img = adv
        face_new = face
    return adv_new_img, face_new


def amplify(params,
            face,
            delta,
            amp,
            dets,
            imgs,
            file_names):
    """
    Description

    Keyword arguments:
    """
    
    adv_crop_stack = []
    adv_img_stack = []
    delta_clip_stack = []
    for i, f in enumerate(face):
        if delta[i] is not None:
            cur_delta = delta[i] * amp
            cur_face = face[i]
            
            adv_crop = cur_face + cur_delta
            adv_crop = np.maximum(adv_crop, params['pixel_min'])
            adv_crop = np.minimum(adv_crop, params['pixel_max'])

            adv_crop, temp_face = transpose_back(params=params,
                                                 adv=adv_crop,
                                                 face=cur_face)

            delta_clip = adv_crop - temp_face
            if len(delta_clip.shape) == 3:
                adv_img = apply_delta(delta_clip, imgs[i], dets[i], params)
            else:
                adv_img = apply_delta(delta_clip[0], imgs[i], dets[i], params)
            if len(delta_clip.shape) != 3:
                adv_crop_stack.append(adv_crop[0])
                delta_clip_stack.append(delta_clip[0])
            else:
                adv_crop_stack.append(adv_crop)
                delta_clip_stack.append(delta_clip)
            adv_img_stack.append(adv_img)
        else:
            adv_crop_stack.append(None)
            adv_img_stack.append(None)
            delta_clip_stack.append(None)
    return adv_crop_stack, delta_clip_stack, adv_img_stack


def initialize_dict(file_names):
    """
    Description

    Keyword arguments:
    """
    
    adv_crop_dict = {}
    delta_clip_dict = {}
    adv_img_dict = {}
    for i in file_names:
        adv_crop_dict[i] = []
        delta_clip_dict[i] = []
        adv_img_dict[i] = []
    return adv_crop_dict, delta_clip_dict, adv_img_dict


def populate_dict(file_names,
                  adv_crop_dict,
                  adv_crop_stack,
                  delta_clip_dict,
                  delta_clip_stack,
                  adv_img_dict,
                  adv_img_stack):
    """
    Description

    Keyword arguments:
    """
    
    for i, name in enumerate(file_names):
        adv_crop_dict[name].append(adv_crop_stack[i])
        delta_clip_dict[name].append(delta_clip_stack[i])
        adv_img_dict[name].append(adv_img_stack[i])
    return adv_crop_dict, delta_clip_dict, adv_img_dict


def save_image(file_names,
               out_img_names,
               out_img_names_crop,
               adv_img_stack,
               adv_crop_stack):
    """
    Description

    Keyword arguments:
    """
    
    print('Images written to {}'.format(out_img_names[file_names[0]]))
    for i, name in enumerate(adv_img_stack):
        if adv_img_stack[i] is not None:
            file = file_names[i]
            imageio.imwrite(out_img_names[file], (adv_img_stack[i] * 255).astype(np.uint8))
            print('SUCCESS! Images written to {}'.format(out_img_names[file]))
            imageio.imwrite(out_img_names_crop[file], (adv_crop_stack[i] * 255).astype(np.uint8))
            print('SUCCESS! Images written to {}'.format(out_img_names_crop[file]))


def save_np(out_npz_names,
            adv_crop_dict,
            delta_clip_dict,
            adv_img_dict):
    """
    Description

    Keyword arguments:
    """
    
    for key, val in adv_crop_dict.items():
        if delta_clip_dict[key] is not None:
            np.savez(out_npz_names[key], delta_clip_stack=delta_clip_dict[key])


def load_images(params, source, target):
    """
    Description

    Keyword arguments:
    """
    
    print('Loading Images...')
    faces = {'base': {}, 'source': {}, 'target': {}}
    file_names = []
    imgs = []
    dets = []
    base_faces = []
        
    base_path = os.path.join(Config.ROOT, params['test_dir'], source)
    source_path = os.path.join(Config.ROOT, params['align_dir'], source)
    target_path = os.path.join(Config.ROOT, params['align_dir'], target)

    base_files = os.listdir(base_path)
    source_files = os.listdir(source_path)
    target_files = os.listdir(target_path)

    with tf.Graph().as_default():
        gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.1)
        sess = tf.Session(config=tf.ConfigProto(gpu_options=gpu_options, log_device_placement=False))
        with sess.as_default():
            pnet, rnet, onet = create_mtcnn(sess, None)
    for file in base_files:
        img = imageio.imread(os.path.join(base_path, file))
        print(file)
        face, det = crop_face(img, params, pnet, rnet, onet)
        if face is not None:
            img = np.around(img / 255.0, decimals=12)
            file_names.append(file)
            base_faces.append(np.array([face]))
            imgs.append(img)
            dets.append(det)

    temp_files = []
    for file in source_files:
        temp_files.append(os.path.join(source_path, file))
    faces['source'] = read_face_from_aligned(temp_files, params)

    temp_files = []
    for file in target_files:
        temp_files.append(os.path.join(target_path, file))
    faces['target'] = read_face_from_aligned(temp_files, params)

    faces['base'] = np.squeeze(np.array(base_faces))
    if len(imgs) <= 1:
        faces['base'] = np.expand_dims(faces['base'], axis=0)

    return faces, file_names, imgs, dets


def return_write_path(params, file_names, target, margin, amplification):
    """
    Description

    Keyword arguments:
    """
    
    marg_str = '%0.2f' % margin
    amp_str = '%0.3f' % amplification
    print('Writing images... Margin: {}, Amplification: {}'.format(marg_str, amp_str))
    img_path = {}
    crop_path = {}
    npz_path = {}
    png_format = '{}_{}_{}_loss_{}_{}_marg_{}_amp_{}.png'
    npz_format = '{}_{}_{}_loss_{}_{}_marg_{}.npz'
    for file in file_names:
        name = file[0 : file.index('.')]
        png_name = png_format.format(params['attack_name'],
                                     params['model_name'],
                                     params['attack_loss'][0],
                                     name,
                                     target,
                                     marg_str,
                                     amp_str)
        npz_name = npz_format.format(params['attack_name'],
                                     params['model_name'],
                                     params['attack_loss'][0],
                                     name,
                                     target,
                                     marg_str)
        img_path[file] = os.path.join(params['directory_path'], png_name)
        crop_path[file] = os.path.join(params['directory_path_crop'], png_name)
        npz_path[file] = os.path.join(params['directory_path_npz'], npz_name)
    return img_path, crop_path, npz_path
