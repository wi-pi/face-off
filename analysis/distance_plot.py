import os
import cv2
import Config
import math
import numpy as np
from models.face_models import get_model
from utils.eval_utils import load_images, load_adv_images, compute_embeddings
import matplotlib.pyplot as plt

ROOT = '/home/bjtang2/github/AdvFace/new_adv_imgs/cw_l2/large_triplet/hinge_loss/full'

SOURCE = 'barack'
TARGET = 'taylor'
not_save = False

#def return_files(path, src_input, src_embedding, target_embedding, model, d_src, d_target):
#    files = os.listdir(path)
#    for fil in files:
#        elements = fil.split('_')
#        src = elements[6]
#        margin = round(float(elements[10]), 2)
#        amp = round(float(elements[12][:-4]), 2)
#        if src_input == src:
#            file_name = os.path.join(ROOT, fil)
#            img = cv2.imread(file_name)
#            embedding = compute_embedding(img, model)
#            src_distance, target_distance = compute_distance(embedding, src_embedding, target_embedding)
#            key = str(margin)+'_'+str(amp)
#            if key not in d_src:
#                d_src[key] = [src_distance]
#            else:
#                d_src[key].append(src_distance)
            
#            if key not in d_target:
#                d_target[key] = [target_distance]
#            else:
#                d_target[key].append(target_distance)

#    return d_src, d_target

#def plot_distances(d, typ):
#    keys = d.keys()
#    margins = [0, 5, 10]
#    amps = []

#    def avg(l):
#        return sum(l)/len(l)
    
#    for marg in margins:
#        x = []
#        y = []
#        for amp in amps:
#            key = str(marg) + '_' + str(amp)
#            x.append(amp)
#            y.append(avg(d[key])
#         plot(x, y, 'o', label=r'$\kappa$='+str(marg))


def plot_distances_pair(surrogate_dist, victim_dist, f_name, typ, margins, amps):
    #plt.clf()
    '''
    margin_x = {}
    margin_y = {}
    for i, dist in enumerate(d):
        marg = margins[i]
        if marg not in margin_x:
            margin_x[marg] = []
            margin_y[marg] = []
        margin_x[marg].append(amps[i])
        margin_y[marg].append(dist)
        print(marg, amps[i], dist)
    def avg(l):
        return sum(l)/len(l)
    for marg, x in margin_x.items():
        plot(x, margin_y[i], 'o', label=r'$\kappa$='+str(marg))
    '''
    margins = margins[:2]
    for i, margin in enumerate(margins):
        print(i, margin)
        dist1_sub = surrogate_dist[i]
        dist2_sub = victim_dist[i]
        dist_sub = dist1_sub - dist2_sub
        amp_sub = amps[i]
        plt.plot(amp_sub, dist_sub, marker='o', label=r'$\kappa$='+str(margin)+', '+typ)
    plt.xlabel('Amplification ('+r'$\alpha$'+')', fontsize=18)
    plt.ylabel(r'$\ell_2$'+' Distance', fontsize=18)
    '''
    if typ == 'source':
        plt.ylabel(r'$\ell_2$'+' Distance '+r'$r(x,a,f)$' + ' from '+r'$\beta_{source}$', fontsize=18)
    if typ == 'target':
        plt.ylabel(r'$\ell_2$'+' Distance '+r'$r(x,a,f)$' + ' from '+r'$\beta_{target}$', fontsize=18)
    '''
    plt.legend(frameon=False, prop={'size': 14})
    save_path = os.path.join(os.getcwd(), 'distance_plots/pairs')
    if os.path.exists(save_path) == False:
        os.mkdir(save_path)
    plt.savefig(os.path.join(save_path, f_name+'-'+typ+'.png'))


def plot_distances(dist, f_name, typ, margins, amps):
    plt.clf()
    '''
    margin_x = {}
    margin_y = {}
    for i, dist in enumerate(d):
        marg = margins[i]
        if marg not in margin_x:
            margin_x[marg] = []
            margin_y[marg] = []
        margin_x[marg].append(amps[i])
        margin_y[marg].append(dist)
        print(marg, amps[i], dist)
    def avg(l):
        return sum(l)/len(l)
    for marg, x in margin_x.items():
        plot(x, margin_y[i], 'o', label=r'$\kappa$='+str(marg))
    '''
    margins = margins[:2]
    for i, margin in enumerate(margins):
        print(i, margin)
        dist_sub = dist[i]
        amp_sub = amps[i]
        plt.plot(amp_sub, dist_sub, marker='o', label=r'$\kappa$='+str(margin)+', '+typ)
    plt.xlabel('Amplification ('+r'$\alpha$'+')', fontsize=18)
    if 'large' in f_name:
        #plt.ylabel(r'$\ell_2$'+' Distance '+r'$r(x,a,f,s)$', fontsize=18)
        plt.ylabel(r'R$(\alpha, f)$', fontsize=18)
    else:
        #plt.ylabel(r'$\ell_2$'+' Distance '+r'$r(x,a,g,s)$', fontsize=18)
        plt.ylabel(r'$\ell_2$'+' Distance '+r'$r(x,a,g,s)$', fontsize=18)
        plt.ylabel(r'R$(\alpha, g)$', fontsize=18)

    '''
    if typ == 'source':
        plt.ylabel(r'$\ell_2$'+' Distance '+r'$r(x,a,f)$' + ' from '+r'$\beta_{source}$', fontsize=18)
    if typ == 'target':
        plt.ylabel(r'$\ell_2$'+' Distance '+r'$r(x,a,f)$' + ' from '+r'$\beta_{target}$', fontsize=18)
    '''
    plt.legend(frameon=False, prop={'size': 14})
    save_path = os.path.join(os.getcwd(), 'distance_plots/rest')
    if os.path.exists(save_path) == False:
        os.mkdir(save_path)
    final_fig_path = os.path.join(save_path, f_name+'-'+typ+'.png')
    print("saving:", final_fig_path)
    plt.savefig(final_fig_path)  
 
#def obtain_embedding(label, model):
#    if model == ''
#        return embeddings[label]

#def compute_embedding(img, model)
#    val = model.predict(img)
#    return val


def compute_distance(a, b):
    cos_sim = np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))
    distance = np.linalg.norm(a - b)
    cos_sim = np.arccos(cos_sim) / math.pi
    return distance, cos_sim

if __name__ == "__main__":
    SRC_MODEL = 'large'
    SRC_LOSS = 'triplet'
    SRC_DATA = 'vgg'
    #TAR_MODEL = 'small'
    TAR_MODEL = 'large'
    #TAR_LOSS = 'center'
    TAR_LOSS = 'triplet'
    TAR_DATA = 'vgg'
    ATTACK = 'CW'
    NORM = '2'
    CHUHAN_LOSS = 'embedding'

    params = Config.set_parameters(model_type=SRC_MODEL,
                                   loss_type=SRC_LOSS,
                                   dataset_type=SRC_DATA,
                                   target_model=TAR_MODEL,
                                   target_loss=TAR_LOSS,
                                   target_dataset=TAR_DATA,
                                   batch_size=10,
                                   pair_flag='true',
                                   granularity='api-eval',
                                   amplification=10.6,
                                   margin=15.0,
                                   mean_loss=CHUHAN_LOSS,
                                   attack=ATTACK,
                                   norm=NORM,
                                   whitebox_target=True)
    tf_config = Config.set_gpu('1')

    faces, people = load_images(folder=params['align_dir'],
                                params=params)
    centroids, _ = compute_embeddings(faces=faces,
                                      people=people,
                                      tf_config=tf_config,
                                      params=params)

    #faces, names, margins, amps, people, _ = load_adv_images(params=params)
    faces, names, margins, amps, people = load_adv_images(params=params)
    _, embeddings = compute_embeddings(faces=faces,
                                       people=people,
                                       tf_config=tf_config,
                                       params=params)

    def compress(distances, margins, amps):
        N = len(distances)
        d = {}
        print(len(distances), len(margins), len(amps))
        assert N == len(margins)
        assert N == len(amps)
        for i in range(N):
            key = str(margins[i])+'_'+str(amps[i])
            if key not in d:
                d[key] = [distances[i]]
            else:
                d[key].append(distances[i])

        def avg(l):
            return sum(l)/len(l)

        amplifications = [[], []]
        margins_set = {'Z'}
        values = [[], []]
       
        keys = list(d.keys())
        for key in keys:
            val = avg(d[key])
            d[key] = val

        keys.sort()
        print(keys) 
        for key in keys:
            splits = key.split('_')
            margin = int(float(splits[0])) 
            margins_set.add(margin)
            amplification = float(splits[1])
            val = d[key]
            if margin == 0:
                amplifications[0].append(amplification)
                values[0].append(val)
            elif margin == 10:
                amplifications[1].append(amplification)
                values[1].append(val)
            '''            
            elif margin == 10:
                amplifications[2].append(amplification)
                values[2].append(val)
            '''
        margins_set.remove('Z')
        print("inside:", margins_set)
        margins_set = list(margins_set)
        return margins_set, values, amplifications
        

    index = 0
    def save_as_npz(f_name, typ, dist, y, x):
        file_name = f_name + '-' + typ + '.npz'
        file_name = './distance_plots/'+file_name
        x = np.asarray(x)
        y = np.asarray(y)
        dist = np.asarray(dist)
        np.savez(file_name, x=x, y=y, dist=dist)        

    def read_from_npz(f_name, typ):
        file_name = f_name + '-' + typ + '.npz'
        file_name = './distance_plots/'+file_name
        data = np.load(file_name)
        x = list(data['x'])
        y = list(data['y'])
        dist = list(data['dist'])
        return dist, x, y         

    #'''
    for person in people:
        if not_save:
            split = person.split(':')
            source = split[0]
            target = split[1]
            adv_embed = embeddings[person]
            src_dist = []
            targ_dist = []
            src_cos = []
            targ_cos = []
            for i in adv_embed:
                dist, cos = compute_distance(i, centroids[source])
                src_dist.append(dist)
                src_cos.append(cos)
                dist, cos = compute_distance(i, centroids[target])
                targ_dist.append(dist)
                targ_cos.append(cos)
            end = index + len(adv_embed)
            margins_out, src_dist_out, amps_out = compress(src_dist, margins[index:end], amps[index:end])
            plt.clf()
            #print(margins, len(src_dist), len(amps))
            #plot_distances(src_dist_out, person.replace(':','-')+'-source-'+TAR_MODEL, margins_out, amps_out)
            save_as_npz(person.replace(':','-')+'-'+TAR_MODEL, 'source', src_dist_out, margins_out, amps_out)
            plot_distances(src_dist_out, person.replace(':','-')+'-'+TAR_MODEL, 'source', margins_out, amps_out)
    
            margins_out, targ_dist_out, amps_out = compress(targ_dist, margins[index:end], amps[index:end])
            save_as_npz(person.replace(':','-')+'-'+TAR_MODEL, 'target', targ_dist_out, margins_out, amps_out)
            #plot_distances(targ_dist_out, person.replace(':', '-')+'-target-'+TAR_MODEL, margins_out, amps_out)
            plot_distances(targ_dist_out, person.replace(':', '-')+'-'+TAR_MODEL, 'target', margins_out, amps_out)
            index += len(adv_embed)
        else:
            src_dist_out, amps_out, margins_out = read_from_npz(person.replace(':', '-')+'-'+TAR_MODEL, 'source')
            plot_distances(src_dist_out, person.replace(':', '-')+'-'+TAR_MODEL, 'source', margins_out, amps_out)
            targ_dist_out, amps_out, margins_out = read_from_npz(person.replace(':', '-')+'-'+TAR_MODEL, 'target')
            plot_distances(targ_dist_out, person.replace(':', '-')+'-'+TAR_MODEL, 'target', margins_out, amps_out)
    #'''
   
    ''' 
    for person in people:
            plt.clf()
            src_dist_out, amps_out, margins_out = read_from_npz(person.replace(':', '-')+'-'+'small', 'source')
            targ_dist_out, amps_out, margins_out = read_from_npz(person.replace(':', '-')+'-'+'large', 'source')
            plot_distances_pair(targ_dist_out, src_dist_out, person.replace(':', '-'), 'source', margins_out, amps_out)
            src_dist_out, amps_out, margins_out = read_from_npz(person.replace(':', '-')+'-'+'small', 'target')
            targ_dist_out, amps_out, margins_out = read_from_npz(person.replace(':', '-')+'-'+'large', 'target')
            plot_distances_pair(targ_dist_out, src_dist_out, person.replace(':', '-'), 'target', margins_out, amps_out)
    '''


#    src_embedding = obtain_embedding(SOURCE, model)
#    target_embedding = obtain_embedding(TARGET, model)
#    d_src, d_target = return_files(ROOT, src_embedding, target_embedding, model, d_src, d_target)
#    plot_distances(d_src, "source")
#    plot_distances(d_target, "target")
    
    # black box
#    model = black_box_model
#    src_embedding = obtain_embedding(SOURCE, model)
#    target_embedding = obtain_embedding(TARGET, model)
#    d_src, d_target = return_files(ROOT, src_embedding, target_embedding, model, d_src, d_target)
#    plot_distances(d_src, "source")
#    plot_distances(d_target, "target")




