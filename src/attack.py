import numpy as np
import tensorflow as tf
import Config
from attacks.cw import CW
from attacks.pgd import PGD
from models.face_models import get_model
from utils.attack_utils import (set_bounds, transpose_back, initialize_dict, populate_dict, save_image,
    save_np, load_images, return_write_path, amplify)
import argparse
from keras import backend
import imageio
import os


def find_adv(sess,
             params,
             fr_model,
             face,
             face_stack_source,
             face_stack_target,
             margin=0):
    """
    Description

    Keyword arguments:
    sess -- tensorflow session
    params -- parameter dict (Config)
    fr_model -- loaded facenet or centerface tensorflow model
    face -- single face or batch of faces to perturb
    face_stack_source -- single face or batch of source faces (used in hinge loss)
    face_stack_target -- single face or batch of target faces (used in all loss)
    """

    num_base = face.shape[0]
    num_src = face_stack_source.shape[0]
    num_target = face_stack_target.shape[0]

    if params['attack'] == 'CW':
        cw_attack = CW(sess=sess,
                       model=fr_model,
                       params=params,
                       num_base=num_base,
                       num_src=num_src,
                       num_target=num_target,
                       confidence=margin,
                       margin=margin)
        best_lp, best_const, best_adv, best_delta = cw_attack.attack(face,
                                                                     target_imgs=face_stack_target,
                                                                     src_imgs=face_stack_source,
                                                                     params=params)
    elif params['attack'] == 'PGD':
        best_lp = []
        best_const = []
        best_adv = []
        best_delta = []
        if params['batch_size'] <= 0:
            batch_size = num_base
        else:
            batch_size = min(params['batch_size'], num_base)
        for i in range(0,len(face),batch_size):
            pgd_attack = PGD(fr_model, back='tf', sess=sess)
            pgd_params = {'eps': params['epsilon'], 
                          'eps_iter': params['epsilon_steps'], 
                          'nb_iter': params['iterations'], 
                          'ord': params['norm']}
            pgd_attack.set_parameters(params=params,
                                      target_imgs=face_stack_target,
                                      src_imgs=face_stack_source,
                                      margin=margin, 
                                      model=fr_model,
                                      base_imgs=face[i:i+batch_size],
                                      **pgd_params)

            adv, lp = pgd_attack.generate(face[i:i+batch_size], **pgd_params)
            
            delta = adv - face[i:i+batch_size]
            const = [None] * face.shape[0]

            best_lp.extend(best_lp)
            best_const.extend(const)
            best_adv.extend(adv)
            best_delta.extend(delta)

    return best_adv, best_delta, best_lp, best_const


def outer_attack(params,
                 faces,
                 file_names,
                 source,
                 target,
                 tf_config,
                 imgs,
                 dets):
    """
    Outer attack loop of margin (kappa) values. Finds adversarial example, amplifies delta,
    saves perturbed image, creates .npz file with delta values.

    Keyword arguments:
    params -- parameter dict (Config)
    faces -- dict of base, source, and target faces
    file_names -- base image file names
    source -- source person labal
    target -- target person labal
    tf_config -- tensorflow gpu configuration
    imgs -- original images containing faces
    dets -- bounding box coordinates of faces
    """

    for margin in params['margin_list']:
        backend.clear_session()
        tf.reset_default_graph()
        with tf.Session(config=tf_config) as sess:
            Config.BM.mark('Model Loaded')
            fr_model = get_model(params)
            Config.BM.mark('Model Loaded')

            Config.BM.mark('Adversarial Example Generation')
            adv, delta, lp, const = find_adv(sess, 
                                             params=params, 
                                             fr_model=fr_model,
                                             face=faces['base'], 
                                             face_stack_source=faces['source'],
                                             face_stack_target=faces['target'],
                                             margin=margin)
        Config.BM.mark('Adversarial Example Generation')

        Config.BM.mark('Dictionary Initialization')
        adv_crop_dict, delta_clip_dict, adv_img_dict = initialize_dict(file_names=file_names)
        Config.BM.mark('Dictionary Initialization')

        Config.BM.mark('Amplifying and Writing Images')
        for amplification in params['amp_list']:
            img_path, crop_path, npz_path = return_write_path(params=params,
                                                              file_names=file_names,
                                                              target=target,
                                                              margin=margin,
                                                              amplification=amplification)
            adv_crop_stack, delta_clip_stack, adv_img_stack = amplify(params=params,
                                                                      face=faces['base'],
                                                                      delta=delta,
                                                                      amp=amplification,
                                                                      dets=dets,
                                                                      imgs=imgs,
                                                                      file_names=file_names)
            save_image(file_names = file_names,
                       out_img_names = img_path,
                       out_img_names_crop = crop_path,
                       adv_img_stack = adv_img_stack,
                       adv_crop_stack = adv_crop_stack)
            adv_crop_dict, delta_clip_dict, adv_img_dict = populate_dict(file_names = file_names,
                                                                         adv_crop_dict = adv_crop_dict,
                                                                         adv_crop_stack = adv_crop_stack,
                                                                         delta_clip_dict = delta_clip_dict,
                                                                         delta_clip_stack = delta_clip_stack,
                                                                         adv_img_dict = adv_img_dict,
                                                                         adv_img_stack = adv_img_stack)
        Config.BM.mark('Amplifying and Writing Images')

        Config.BM.mark('Saving Numpy Array')
        save_np(out_npz_names = npz_path,
                adv_crop_dict = adv_crop_dict,
                delta_clip_dict = delta_clip_dict,
                adv_img_dict = adv_img_dict)
        Config.BM.mark('Saving Numpy Array')


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--gpu', type=str, default="0", help='GPU(s) to run the code on')
    parser.add_argument('--model-type', type=str, default="small", help='type of model', choices=['small','large'])
    parser.add_argument('--loss-type', type=str, default="center", help='loss function used to train the model',choices=['center','triplet'])
    parser.add_argument('--dataset-type', type=str, default='vgg', help='dataset used to train the model',choices=['vgg','casia','vggsmall'])
    parser.add_argument('--attack', type=str, default='CW', help='attack type',choices=['PGD', 'CW'])
    parser.add_argument('--norm', type=str, default='2', help='p-norm', choices=['inf', '2'])
    parser.add_argument('--targeted-flag', type=str, default='true', help='targeted (true) or untargeted (false)', choices=['true', 'false'])
    parser.add_argument('--tv-flag', type=str, default='false', help='do not use tv_loss term (false) or use it (true)', choices=['true', 'false'])
    parser.add_argument('--hinge-flag', type=str, default='true', help='hinge loss (true) or target loss (false)', choices=['true', 'false'])
    parser.add_argument('--epsilon', type=float, default=0.1, help='epsilon value needed for PGD')
    parser.add_argument('--margin', type=float, default=6.0, help='needed for determining goodness of transferability')
    parser.add_argument('--amplification', type=float, default=5.1, help='needed for amplifying adversarial examples')
    parser.add_argument('--iterations', type=int, default=20, help='number of inner step iterations for CW and number of iterations for PGD')
    parser.add_argument('--binary-steps', type=int, default=5, help='number of binary search steps for CW')
    parser.add_argument('--learning-rate', type=float, default=0.01, help='learning rate for CW')
    parser.add_argument('--epsilon-steps', type=float, default=0.01, help='epsilon per iteration for PGD')
    parser.add_argument('--init-const', type=float, default=0.3, help='initial const value for CW')
    parser.add_argument('--interpolation', type=str, default='bilinear', help='interpolation method for upscaling and downscaling', choices=['nearest', 'bilinear', 'bicubic', 'lanczos'])
    parser.add_argument('--granularity', type=str, default='normal', help='add more or less margin and amplification intervals', choices=['fine-tuned', 'fine', 'normal', 'coarse', 'coarser', 'coarse-single', 'single', 'api-eval'])
    parser.add_argument('--cos-flag', type=str, default='false', help='use cosine similarity instead of l2 for loss', choices=['true', 'false'])
    parser.add_argument('--mean-loss', type=str, default='embeddingmean', help='old:(embedding) new formulation:(embeddingmean) WIP formulation:(distancemean)', choices=['embeddingmean', 'embedding', 'distancemean'])
    parser.add_argument('--batch-size', type=int, default=-1, help='batch size')
    parser.add_argument('--pair-flag', type=str, default='false', help='optimal source target pairs')
    parser.add_argument('--source', type=str, default='none', help='', choices=['barack', 'bill', 'jenn', 'leo', 'mark', 'matt', 'melania', 'meryl', 'morgan', 'taylor', 'myface', 'none'])
    args = parser.parse_args()
    
    tf_config = Config.set_gpu(args.gpu)
    params = Config.set_parameters(targeted_flag=args.targeted_flag,
                                   tv_flag=args.tv_flag,
                                   hinge_flag=args.hinge_flag,
                                   cos_flag=args.cos_flag,
                                   interpolation=args.interpolation,
                                   model_type=args.model_type,
                                   loss_type=args.loss_type,
                                   dataset_type=args.dataset_type,
                                   attack=args.attack,
                                   norm=args.norm,
                                   epsilon=args.epsilon,
                                   iterations=args.iterations,
                                   binary_steps=args.binary_steps,
                                   learning_rate=args.learning_rate,
                                   epsilon_steps=args.epsilon_steps,
                                   init_const=args.init_const,
                                   mean_loss=args.mean_loss,
                                   batch_size=args.batch_size,
                                   margin=args.margin,
                                   amplification=args.amplification,
                                   granularity=args.granularity,
                                   pair_flag=args.pair_flag)
    
    if params['dataset_type'] == 'vggsmall':
        for i, s in enumerate(Config.SOURCES):
            faces, file_names, imgs, dets = load_images(params=params, source=Config.SOURCES[i], target=Config.TARGETS[i])
            faces['source'] = faces['source'][:64]
            faces['target'] = faces['target'][:64]
            outer_attack(params=params,
                         faces=faces,
                         file_names=file_names,
                         source=Config.SOURCES[i],
                         target=Config.TARGETS[i],
                         tf_config=tf_config,
                         imgs=imgs,
                         dets=dets)
    else:
        if args.source == 'none':
            for source in Config.NAMES:
                for target in Config.NAMES:
                    if (target != source and 
                        (target == Config.PAIRS[source] or not params['pair_flag'])):
                        faces, file_names, imgs, dets = load_images(params=params, source=source, target=target)
                        outer_attack(params=params,
                                     faces=faces,
                                     file_names=file_names,
                                     source=source,
                                     target=target,
                                     tf_config=tf_config,
                                     imgs=imgs,
                                     dets=dets)
        else:
            for target in Config.NAMES:
                if (target != args.source and 
                    (target == Config.PAIRS[args.source] or not params['pair_flag'])):
                    faces, file_names, imgs, dets = load_images(params=params, source=args.source, target=target)
                    outer_attack(params=params,
                                 faces=faces,
                                 file_names=file_names,
                                 source=args.source,
                                 target=target,
                                 tf_config=tf_config,
                                 imgs=imgs,
                                 dets=dets)
