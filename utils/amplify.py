import Config
import argparse
import numpy as np
import os
from utils.attack_utils import load_images_old, return_write_path, save_image, transpose_back, amplify
from utils.crop import apply_delta


if __name__ == '__main__':
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
    parser.add_argument('--margin', type=float, default=6.0, help='needed for determining goodness of transferability')
    parser.add_argument('--amplification', type=float, default=5.1, help='needed for amplifying adversarial examples')
    parser.add_argument('--interpolation', type=str, default='bilinear', help='interpolation method for upscaling and downscaling', choices=['nearest', 'bilinear', 'bicubic', 'lanczos'])
    parser.add_argument('--granularity', type=str, default='normal', help='add more or less margin and amplification intervals', choices=['fine-tuned', 'fine', 'normal', 'coarse', 'coarser', 'single', 'api-eval'])
    parser.add_argument('--cos-flag', type=str, default='false', help='use cosine similarity instead of l2 for loss', choices=['true', 'false'])
    parser.add_argument('--mean-loss', type=str, default='embeddingmean', help='old:(embedding) new formulation:(embeddingmean) WIP formulation:(distancemean)', choices=['embeddingmean', 'embedding', 'distancemean'])
    parser.add_argument('--pair-flag', type=str, default='false', help='optimal source target pairs')
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
                                   mean_loss=args.mean_loss,
                                   margin=args.margin,
                                   amplification=args.amplification,
                                   granularity=args.granularity,
                                   pair_flag=args.pair_flag)
    for source in Config.API_PEOPLE:
        for target in Config.API_PEOPLE:
            # if source is not target and target == Config.PAIRS[source]:
            if (target != source and
                (target is Config.PAIRS[source] or not params['pair_flag'])):
                try:
                    faces, file_names, imgs, dets = load_images_old(params=params, source=source, target=target)
                    for margin in params['margin_list']:
                        marg_str = '%0.2f' % margin
                        delta = []
                        for source_file in file_names:
                            npz_path = os.path.join(Config.ROOT, 'new_adv_imgs/{}/{}/{}_loss/npz_mean'.format(params['attack_name'],
                                                                                                              params['model_name'],
                                                                                                              params['attack_loss']))
                            inname = '{}_{}_{}_loss_{}_{}_marg_{}.npz'.format(params['attack_name'],
                                                                              params['model_name'],
                                                                              params['attack_loss'][0],
                                                                              source_file.replace('.jpg', ''),
                                                                              target,
                                                                              marg_str)
                            inname = os.path.join(npz_path, inname)
                            npzfile = np.load(inname, allow_pickle=True)
                            delta.append(npzfile['delta_clip_stack'][0])
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
                except Exception as e:
                    print(e)
