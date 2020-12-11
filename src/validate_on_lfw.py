import os
import Config
import tensorflow as tf
import numpy as np
from models.face_models import get_model
from utils.crop import *
from utils.eval_utils import *


def matching(faces, people, embeddings):
    """
    Evaluates matching accuracy (same or different face)

    Keyword arguments:
    faces -- 
    people -- 
    embeddings -- 
    """
    
    mean_dist, mean_cos = compute_threshold(embeddings, people)
    print(mean_dist, mean_cos)
    matching_accuracy(embeddings, mean_dist, mean_cos)


def classifying(faces, people, embeddings, embedding_means, params):
    """
    Evaluates top-n accuracy (with a bucket of labels)

    Keyword arguments:
    faces -- 
    people -- 
    embeddings -- 
    embedding_means -- 
    params -- 
    """
    
    correct = 0
    total = 0
    correct_cos = 0
    total_cos = 0
    for p, person in enumerate(people):
        labels, distances, labels_cos, cosines = whitebox_eval(embedding_means=embedding_means,
            embeddings=embeddings[person],
            params=params,
            topn=3)
        for key, val in labels.items():
            total += 1
            if labels[key][0] == person:
                correct += 1
        for key, val in labels_cos.items():
            total_cos += 1
            if labels_cos[key][0] == person:
                correct_cos += 1
        print(correct/total)
        print(correct_cos/total_cos)


if __name__ == '__main__':
    # folder = Config.VGG_VALIDATION_DIR
    # folder = '../facenet/datasets/lfw-aligned'
    # folder = 'celebrities-160'
    folder = 'small-vgg-adv'
    tf_config = Config.set_gpu('0')
    correct = 0
    total = 0
    correct_cos = 0
    total_cos = 0
    params = Config.set_parameters(model_type='large',
                                   dataset_type='vggsmall',
                                   batch_size=150)
    faces, people = load_images(folder, params)
    embedding_means, embeddings = compute_embeddings(faces=faces,
                                                     people=people,
                                                     tf_config=tf_config,
                                                     params=params)
    #matching(faces, people, embeddings)
    classifying(faces, people, embeddings, embedding_means, params)
