import numpy as np
import tensorflow as tf
import os
import Config
from utils.crop import *
from keras import backend
from models.face_models import get_model
import math


def load_adv_images(params):
    """
    

    Keyword arguments:
    """
    
    model_name = params['model_name']
    attack_name = params['attack_name']
    attack_loss = params['attack_loss']
    margin_list = params['margin_list']
    amp_list = params['amp_list']
    face_db = []
    name_db = []
    margin_db = []
    amp_db = []
    people_db = []
    if params['mean_loss'] == 'embedding':
        mean_str = '_mean'
    else:
        mean_str = ''
    for s in Config.NAMES:
        for t in Config.NAMES:
            if (t != s and 
                s in Config.PAIRS and
                (t is Config.PAIRS[s] or not params['pair_flag'])):
                faces = []
                image_names = []
                margins = []
                amplifications = []
                people_db.append('{}:{}'.format(s, t))

                for file in os.listdir(os.path.join(Config.ROOT, 'test_imgs', s)):
                    source_file = file.replace('.jpg', '')
                    for ii, margin in enumerate(margin_list):
                        marg_str = '%0.2f' % margin
                        for jj, amp in enumerate(amp_list):
                            amp_str = '%0.3f' % amp
                            adv_img_file = '{}_{}_{}_loss_{}_{}_marg_{}_amp_{}.png'.format(attack_name,
                                                                                           model_name,
                                                                                           attack_loss[0],
                                                                                           source_file,
                                                                                           t,
                                                                                           marg_str,
                                                                                           amp_str)
                            adv_img_path = '{}/new_adv_imgs/{}/{}/{}_loss/crop{}/{}/'.format(Config.ROOT,
                                                                                             attack_name,
                                                                                             model_name,
                                                                                             attack_loss,
                                                                                             mean_str,
                                                                                             adv_img_file)
                            try:
                                face = imageio.imread(adv_img_path)
                                face = pre_proc(face,
                                                params=params)
                                faces.append(face)
                                image_names.append(source_file)
                                margins.append(marg_str)
                                amplifications.append(amp_str)
                            except FileNotFoundError as e:
                                print(e)
                face_db.append(np.array(faces))
                name_db.extend(image_names)
                margin_db.extend(margins)
                amp_db.extend(amplifications)
    return face_db, name_db, margin_db, amp_db, people_db


class Evaluate:
    """Class for face recognition inference. Sets up tensorflow placeholders."""
    def __init__(self,
                 fr_model,
                 params):
        height = fr_model.image_height
        width = fr_model.image_width
        channels = fr_model.num_channels
        if (params['whitebox_target'] and params['target_model'] == 'large' or
            not params['whitebox_target'] and params['model_type'] == 'large'):
            shape = (params['batch_size'], height, width, channels)
        else:
            shape = (params['batch_size'], channels, width, height)
        self.input_tensor = tf.placeholder(tf.float32, shape)
        self.embedding = fr_model.predict(self.input_tensor)
        self.batch_size = shape[0]


def load_images(folder,
                params):
    """
    Description

    Keyword arguments:
    """
    
    face_db = []
    people = []
    for person in os.listdir(folder):
        if not person.endswith('.txt'):
            person_path = os.path.join(folder, person)
            file_list = os.listdir(person_path)
            if len(file_list) > 3:
                for i in range(len(file_list)):
                    file_list[i] = os.path.join(person_path, file_list[i])
                faces = read_face_from_aligned(file_list=file_list,
                                               params=params)
                face_db.append(faces)
                people.append(person)
    return face_db, people


def compute_embeddings(faces,
                       people,
                       tf_config,
                       params):
    """
    Description

    Keyword arguments:
    """
    
    embeddings = {}
    embedding_means = {}
    backend.clear_session()
    tf.reset_default_graph()
    with tf.Session(config=tf_config) as sess:
        fr_model = get_model(params=params)
        eval = Evaluate(fr_model=fr_model, params=params)
        for p, person in enumerate(faces):
            print('Computing: {}'.format(people[p]))
            cur_embedding = []
            sub_batch = -len(person)
            for i in range(0, len(person), eval.batch_size):
                cur_batch = len(person) - i
                cur_imgs = person[i:i+eval.batch_size]
                if eval.batch_size > cur_batch:
                    sub_batch = eval.batch_size - cur_batch
                    cur_imgs = np.pad(cur_imgs, ((0,sub_batch),(0,0),(0,0),(0,0)))
                cur_embedding.extend(sess.run(eval.embedding, feed_dict={eval.input_tensor: cur_imgs}))
            embedding_mean = np.mean(cur_embedding[:-sub_batch], axis=0)
            embedding_means[people[p]] = embedding_mean
            embeddings[people[p]] = np.array(cur_embedding[:-sub_batch])
            
    return embedding_means, embeddings


def compute_threshold(embeddings, people):
    """
    Description

    Keyword arguments:
    """
    
    distances = []
    cosines = []
    for p, person in enumerate(people):
        embed = embeddings[person]
        for i in range(embed.shape[0]):
            for j in range(embed.shape[0]):
                if i != j:
                    cos_sim = np.dot(embed[i], embed[j]) / (np.linalg.norm(embed[i]) * np.linalg.norm(embed[j]))
                    distance = np.linalg.norm(embed[i] - embed[j])
                    cos_sim = np.arccos(cos_sim) / math.pi
                    distances.append(distance)
                    cosines.append(cos_sim)
    return np.mean(distances), np.mean(cosines)


def whitebox_eval(embedding_means,
                  embeddings,
                  params,
                  topn=1):
    """
    Description

    Keyword arguments:
    """
    
    final_l2 = {}
    final_cos = {}

    people = embedding_means.keys()
    
    for person in people:
        embedding_mean_person = embedding_means[person]
        length = embeddings.shape[0]
        output = [0]*length
        output_cos = [0]*length

        for i in range(length):
            # dot = np.dot(embeddings, embedding_mean_person)
            # norm = np.linalg.norm(embeddings) * np.linalg.norm(embedding_mean_person)
            # cos_sim = dot / norm
            cos_sim = np.dot(embeddings[i], embedding_mean_person) / (np.linalg.norm(embeddings[i]) * np.linalg.norm(embedding_mean_person))
            distance = np.linalg.norm(embeddings[i] - embedding_mean_person)
            cos_sim = np.arccos(cos_sim) / math.pi
            output[i] = distance
            output_cos[i] = cos_sim
        final_l2[person] = output
        final_cos[person] = output_cos
    # {matt: [3,2,1], leo: [4,5,4], bill: [6,7,7]}
    def return_top(dictionary,
                   final_cos,
                   n):
        """
        Description

        Keyword arguments:
        """
    
        distances = {}
        cosines = {}
        label_dict = {}
        final_labels = {}
        final_labels_cos = {}
        final_distances = {}
        final_cosines = {}
        keys = dictionary.keys()
        for i, key in enumerate(keys):
            label_dict[i] = key
            for j, dist in enumerate(dictionary[key]):
                if j not in distances:
                    distances[j] = []
                distances[j].append(dist)
            for j, cos in enumerate(final_cos[key]):
                if j not in cosines:
                    cosines[j] = []
                cosines[j].append(cos)

        for key, val in distances.items():
            indices = np.argsort(np.array(val))
            # print(indices)
            final_labels[key] = []
            final_distances[key] = []
            for i in range(n):
                final_labels[key].append(label_dict[indices[i]])
                final_distances[key].append(val[indices[i]])
        for key, val in cosines.items():
            indices = np.argsort(np.array(val))
            # print(indices)
            final_labels_cos[key] = []
            final_cosines[key] = []
            for i in range(n):
                final_labels_cos[key].append(label_dict[indices[i]])
                final_cosines[key].append(val[indices[i]])

        return final_labels, final_distances, final_labels_cos, final_cosines
    final_labels, final_distances, final_labels_cos, final_cosines = return_top(final_l2, final_cos, topn)
    return final_labels, final_distances, final_labels_cos, final_cosines


def matching_accuracy(embeddings, mean_distance, mean_cosine):
    """
    Description

    Keyword arguments:
    """
    
    total = 0
    dist_acc = 0
    cos_acc = 0
    for person1, embed1 in embeddings.items():
        for person2, embed2 in embeddings.items():
            for i in range(embed1.shape[0]):
                for j in range(embed2.shape[0]):
                    if i != j or person1 != person2:
                        cos_sim = np.dot(embed1[i], embed2[j]) / (np.linalg.norm(embed1[i]) * np.linalg.norm(embed2[j]))
                        distance = np.linalg.norm(embed1[i] - embed2[j])
                        cos_sim = np.arccos(cos_sim) / math.pi
                        if (distance <= mean_distance and person1 is person2 or
                            distance > mean_distance and person1 != person2):
                            dist_acc += 1
                        if (cos_sim <= mean_cosine and person1 is person2 or
                            distance > mean_distance and person1 != person2):
                            cos_acc += 1
                        total += 1
    print(dist_acc / total)
    print(cos_acc / total)
