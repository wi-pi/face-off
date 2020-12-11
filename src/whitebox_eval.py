import imageio
import numpy as np 
import os
import csv
import time
import argparse
from utils.crop import *
from utils.eval_utils import *
import tensorflow as tf
from models.face_models import *
import Config


def write_csv(writer,
              params,
              source,
              target,
              image_names,
              margins,
              amplifications,
              labels,
              distances,
              labels_cos,
              cosines):
    """
    Description

    Keyword arguments:
    """
    for key, val in labels.items():
        # print('Image: {}'.format(key))
        out_dict = {}
        out_dict['model_name'] = params['model_name']
        out_dict['target_model_name'] = params['target_model_name']
        out_dict['attack_name'] = params['attack_name']
        out_dict['attack_loss'] = params['attack_loss']
        out_dict['source'] = source
        out_dict['target'] = target
        out_dict['match_source'] = source == labels[key][0]
        out_dict['match_target'] = target == labels[key][0]
        out_dict['cos_source'] = source == labels_cos[key][0]
        out_dict['cos_target'] = target == labels_cos[key][0]
        out_dict['image_name'] = image_names[key]
        out_dict['margin'] = margins[key]
        out_dict['amplification'] = amplifications[key]
        for i, v in enumerate(labels[key]):
            # print('Top {}: {} = {}'.format(i + 1, labels[key][i], distances[key][i]))
            out_dict['top{}'.format(i + 1)] = labels[key][i]
            out_dict['distance{}'.format(i + 1)] = distances[key][i]
        for i, v in enumerate(labels_cos[key]):
            out_dict['topcos{}'.format(i + 1)] = labels_cos[key][i]
            out_dict['cosine{}'.format(i + 1)] = cosines[key][i]

        writer.writerow(out_dict)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--gpu', type=str, default="0", help='GPU(s) to run the code on')
    parser.add_argument('--target-model-type', type=str, default="small", help='type of model', choices=['small','large'])
    parser.add_argument('--target-loss-type', type=str, default="center", help='loss function used to train the model',choices=['center','triplet'])
    parser.add_argument('--target-dataset-type', type=str, default='vgg', help='dataset used in training model', choices=['vgg', 'vggsmall', 'casia'])
    parser.add_argument('--model-type', type=str, default="small", help='type of model', choices=['small','large'])
    parser.add_argument('--loss-type', type=str, default="center", help='loss function used to train the model',choices=['center','triplet'])
    parser.add_argument('--dataset-type', type=str, default='vgg', help='dataset used in training model', choices=['vgg', 'vggsmall', 'casia'])
    parser.add_argument('--attack', type=str, default='CW', help='attack type',choices=['PGD', 'CW'])
    parser.add_argument('--norm', type=str, default='2', help='p-norm', choices=['inf', '2'])
    parser.add_argument('--targeted-flag', type=str, default='true', help='targeted (true) or untargeted (false)', choices=['true', 'false'])
    parser.add_argument('--tv-flag', type=str, default='false', help='do not use tv_loss term (false) or use it (true)', choices=['true', 'false'])
    parser.add_argument('--hinge-flag', type=str, default='true', help='hinge loss (true) or target loss (false)', choices=['true', 'false'])
    parser.add_argument('--cos-flag', type=str, default='false', help='use cosine similarity instead of l2 for loss', choices=['true', 'false'])
    parser.add_argument('--margin', type=float, default=6.0, help='needed for determining goodness of transferability')
    parser.add_argument('--amplification', type=float, default=8.0, help='needed for amplifying adversarial examples')
    parser.add_argument('--granularity', type=str, default='normal', help='add more or less margin and amplification intervals', choices=['fine', 'normal', 'coarse', 'coarser', 'coarsest', 'single', 'fine-tuned'])
    parser.add_argument('--mean-loss', type=str, default='embeddingmean', help='old:(embedding) new formulation:(embeddingmean) WIP formulation:(distancemean)', choices=['embeddingmean', 'embedding', 'distancemean'])
    parser.add_argument('--topn', type=int, default=5, help='do top-n evaluation of closest faces')
    parser.add_argument('--batch-size', type=int, default=9, help='batch size for evaluation')
    parser.add_argument('--pair-flag', type=str, default='false', help='optimal source target pairs')
    args = parser.parse_args()

    tf_config = Config.set_gpu(args.gpu)
    params = Config.set_parameters(targeted_flag=args.targeted_flag,
                                   tv_flag=args.tv_flag,
                                   hinge_flag=args.hinge_flag,
                                   cos_flag=args.cos_flag,
                                   model_type=args.model_type,
                                   loss_type=args.loss_type,
                                   dataset_type=args.dataset_type,
                                   target_model=args.target_model_type,
                                   target_loss=args.target_loss_type,
                                   target_dataset=args.target_dataset_type,
                                   attack=args.attack,
                                   norm=args.norm,
                                   mean_loss=args.mean_loss,
                                   margin=args.margin,
                                   amplification=args.amplification,
                                   granularity=args.granularity,
                                   batch_size=args.batch_size,
                                   whitebox_target=True,
                                   pair_flag=args.pair_flag)
    faces, people = load_images(folder=params['align_dir'],
                                params=params)

    means, _ = compute_embeddings(faces=faces,
                                            people=people,
                                            tf_config=tf_config,
                                            params=params)
    csvfile = open(os.path.join(Config.ROOT, 'fileio', 'whitebox_eval_{}-{}-{}.csv'.format(params['model_name'],
        params['target_model_name'], params['attack_name'])), 'w', newline='')
    fieldnames = ['model_name',  'target_model_name', 'attack_name', 'attack_loss', 'source', 'match_source',
                  'match_target', 'target', 'cos_source', 'cos_target', 'image_name', 'margin', 'amplification']
    for i in range(1, args.topn + 1):
        fieldnames.append('top{}'.format(i))
        fieldnames.append('distance{}'.format(i))
        fieldnames.append('topcos{}'.format(i))
        fieldnames.append('cosine{}'.format(i))
    writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
    writer.writeheader()
    faces, image_names, margins, amplifications, people_db = load_adv_images(params=params)
    _, embeddings = compute_embeddings(faces=faces,
                                       people=people_db,
                                       tf_config=tf_config,
                                       params=params)
    for person in people_db:
        labels, distances, labels_cos, cosines = whitebox_eval(embedding_means=means,
                                                               embeddings=embeddings[person],
                                                               params=params,
                                                               topn=args.topn)
        split = person.split(':')
        source = split[0]
        target = split[1]
        write_csv(writer=writer,
                  params=params,
                  source=source,
                  target=target,
                  image_names=image_names,
                  margins=margins,
                  amplifications=amplifications,
                  labels=labels,
                  distances=distances,
                  labels_cos=labels_cos,
                  cosines=cosines)
