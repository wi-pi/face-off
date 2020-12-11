import os
import sys
import numpy as np
import matplotlib
matplotlib.use('Agg')
from matplotlib import pyplot
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D 

FONTSIZE = 16
LEGENDSIZE = 14
ROOT = os.path.join(Config.ROOT, 'new_api_results')

OUT = os.path.join(Config.ROOT, 'topn_api_plots')

pairings = {0:'Barack', 1:'Barack', 2:'Barack',  3:'Barack',  4:'Barack',  5:'Leo', 6:'Leo', 7:'Leo', 8:'Leo', 9:'Leo',
            10:'Matt', 11:'Matt', 12:'Matt', 13:'Matt', 14:'Matt', 15:'Melania', 16:'Melania', 17:'Melania', 18:'Melania', 19:'Melania',
            20:'Morgan', 21:'Morgan', 22:'Morgan', 23:'Morgan', 24:'Morgan', 25:'Taylor', 26:'Taylor', 27:'Taylor', 28:'Taylor', 29:'Taylor'}


def plot_2d(topn, amp_out, margins, filename, folder_name, N=1):
    plt.clf()
    elements = folder_name.split('/')
    path = os.path.join(os.path.join(elements[-5],elements[-4]), os.path.join(elements[-3], elements[-2]))
    file_path = os.path.join(OUT, path)
    if os.path.isdir(file_path) == False:
        os.makedirs(file_path)
    file_path = os.path.join(file_path, filename)

    for i, margin in enumerate(margins):
        amp = amp_out[i]
        top = topn[i]
        plt.plot(amp, top, marker='o', label=r'$\kappa=$'+str(margin))
    plt.legend(loc='best', frameon=False, fontsize=LEGENDSIZE)
    plt.xlabel('Amplification (' + r'$\alpha$)', fontsize=FONTSIZE)
    plt.ylabel('Top-'+str(N)+' Success', fontsize=FONTSIZE)
    plt.savefig(file_path)

def plot(topn, amp_out, margin_out, filename, folder_name, N=1):
    plt.clf()
    elements = folder_name.split('/')
    path = os.path.join(os.path.join(elements[-5],elements[-4]), os.path.join(elements[-3], elements[-2]))
    file_path = os.path.join(OUT, path)
    if os.path.isdir(file_path) == False:
        os.makedirs(file_path)
    file_path = os.path.join(file_path, filename)
    fig = pyplot.figure()
    ax = Axes3D(fig)
    ax.scatter(amp_out, margin_out, topn)
    ax.set_xlabel('Amplification (' + r'$\alpha$)', fontsize=FONTSIZE)     
    ax.set_ylabel('Margin ('+r'$\kappa$)', fontsize=FONTSIZE)     
    #ax.set_zlabel('Match Target (%)')     
    ax.set_zlabel('Top'+str(N)+' Success', fontsize=FONTSIZE)     
    plt.savefig(file_path)  

def read(api, attack, model, n):
    folder_name = os.path.join(ROOT, os.path.join(api, os.path.join(attack, model)))
    folder_name = os.path.join(folder_name, os.path.join('hinge_loss', 'npz_mean'))
    files = os.listdir(folder_name)
   
    if len(files) == 0:
        exit()
 
    def find(l, search_type='min'):
        n = len(l)
        indices = np.argsort(l)
        indices = list(indices)
        final = []
        count = 0
        for index in indices:
            count += 1
            final.append(index)
        if count == 0:
            final = indices
        return final
        
        
    def match(inp1, indices, n=1):
        matchings = []
        for index in indices:
            matchings.append(pairings[index].lower())
        len_matchings = len(matchings)
        if len_matchings < n:
            n = len_matchings

        final = matchings[-n:]
        #print(final, inp1)
        if inp1 in final:
            return 1
        else:
            return 0

    files1 = [f for f in files if 'barack' in f and 'morgan' in f]
    for fil in files1:
        file_name = os.path.join(folder_name, fil)
        elements = fil.split('.')[0].split('_')
        src = elements[-3]
        tmp = np.load(file_name)
        #['th_self_crop', 'margin_list', 'score_target_crop', 'score_all', 'th_self', 'score_self', 'th_all_crop', 'th_target', 'th_all', 'score_all_crop', 'th_target_crop', 'score_target', 'l2_mtx', 'amp_list', 'score_self_crop']
        
        #th_self_crop = tmp['th_self_crop']
        margin_list = tmp['margin_list']
        #score_target_crop = tmp['score_target_crop']
        score_all = tmp['score_all']
        #th_self = tmp['th_self']
        #score_self = tmp['score_self']
        #th_all_crop = tmp['th_all_crop']
        #th_target = tmp['th_target']
        if 'th_all' in tmp:
            th_all = tmp['th_all']
        else:
            exit()
        #score_all_crop = tmp['score_all_crop']
        #th_target_crop = tmp['th_target_crop']
        #score_target = tmp['score_target']
        #l2_mtx = tmp['l2_mtx']
        amp_list = tmp['amp_list']
        #score_self_crop = tmp['score_self_crop']
        num_margins, num_amps, num_comparisons = score_all.shape
        num_inputs = 6
        N = 5
       
        #th_all = np.average(th_all, axis=3) 
        
        topn = []
        amp_out = []
        margin_out = []
        topn_2d_final = []
        amp_2d_final = []
        for i in range(num_margins):
            amp_2d = []
            topn_2d = []
            for j in range(num_amps):
                out_val = 0
                tmp_list = []
                for k in range(num_inputs):
                    for l in range(N):
                        actual_l = (k*N) + l
                        val = (margin_list[i], round(amp_list[j],2), round(score_all[i][j][actual_l],2))
                        val = val[2] - 50 #round(th_all[i][j][actual_l][2], 2)
                        tmp_list.append(val)
                        #print(val)
                indices = find(tmp_list, 'min')
                    #dest = pairings[index].lower()
                out_val += match(src, indices, n)
                out_val = float(out_val)
                print(out_val, margin_list[i], round(amp_list[j]), tmp_list[indices[len(indices)-1]], src, pairings[indices[len(indices)-1]])
                topn.append(out_val)
                topn_2d.append(out_val)
                amp_out.append(amp_list[j])
                amp_2d.append(amp_list[j])
                margin_out.append(margin_list[i])
            amp_2d_final.append(amp_2d)
            topn_2d_final.append(topn_2d)
        plot_2d(topn_2d_final, amp_2d_final, margin_list, fil.replace('.npz', '_2d_top'+str(n)+'.png'), folder_name, n)
        plot(topn, amp_out, margin_out, fil.replace('.npz', '_top'+str(n)+'.png'), folder_name, n)

def read_average(api, attack, model, n):
    folder_name = os.path.join(ROOT, os.path.join(api, os.path.join(attack, model)))
    folder_name = os.path.join(folder_name, os.path.join('hinge_loss', 'npz'))
    files = os.listdir(folder_name)
   
    if len(files) == 0:
        exit()
 
    def find(l, search_type='min'):
        n = len(l)
        indices = np.argsort(l)
        indices = list(indices)
        final = []
        count = 0
        for index in indices:
            if l[index] > 0:
                count += 1
                final.append(index)
        if count == 0:
            final = indices
        return final
        
        
    def match(inp1, indices, n=1):
        matchings = []
        for index in indices:
            matchings.append(pairings[index].lower())
        len_matchings = len(matchings)
        if len_matchings < n:
            n = len_matchings

        final = matchings[:n]

        if inp1 in final:
            return 1
        else:
            return 0

    for ind, fil in enumerate(files):
        #print("fil:", fil)
        file_name = os.path.join(folder_name, fil)
        elements = fil.split('.')[0].split('_')
        src = elements[-3]
        tmp = np.load(file_name)
        #['th_self_crop', 'margin_list', 'score_target_crop', 'score_all', 'th_self', 'score_self', 'th_all_crop', 'th_target', 'th_all', 'score_all_crop', 'th_target_crop', 'score_target', 'l2_mtx', 'amp_list', 'score_self_crop']
       
        tmp_margin = tmp['margin_list']
        tmp_score = tmp['score_all']
        tmp_th = tmp['th_all']
        tmp_amp = tmp['amp_list']

        margin_shape = tuple([1] + list(tmp_margin.shape))
        score_shape = tuple([1] + list(tmp_score.shape))
        th_shape = tuple([1] + list(tmp_th.shape))
        amp_shape = tuple([1] + list(tmp_amp.shape))

        tmp_margin = tmp_margin.reshape(margin_shape)
        tmp_score = tmp_score.reshape(score_shape)
        tmp_th = tmp_th.reshape(th_shape)
        tmp_amp = tmp_amp.reshape(amp_shape)

        #th_self_crop = tmp['th_self_crop']
        if ind == 0:
            margin_list_all = tmp_margin
        else:
            margin_list_all = np.vstack((margin_list_all, tmp_margin))
        #score_target_crop = tmp['score_target_crop']
        if ind == 0:
            score_all_all = tmp_score
        else:
            score_all_all = np.vstack((score_all_all, tmp_score))
        #th_self = tmp['th_self']
        #score_self = tmp['score_self']
        #th_all_crop = tmp['th_all_crop']
        #th_target = tmp['th_target']
        if ind == 0:
            if 'th_all' in tmp:
                th_all_all = tmp_th
            else:
                exit()
        else:
            if 'th_all' in tmp:
                th_all_all = np.vstack((th_all_all, tmp_th))
            else:
                exit()
        #score_all_crop = tmp['score_all_crop']
        #th_target_crop = tmp['th_target_crop']
        #score_target = tmp['score_target']
        #l2_mtx = tmp['l2_mtx']
        if ind == 0:
            amp_list_all = tmp_amp
        else:
            amp_list_all = np.vstack((amp_list_all, tmp_amp))
        #score_self_crop = tmp['score_self_crop']

    def rename(f):
        items = f.split('.')[0].split('_')
        n = len(items) - 2
        tmp = ''
        for i in range(n):
            tmp+= items[i]
            if i != n-1:
                tmp+='_'
        return tmp+'.npz'
        
    f_n = rename(fil)
    
    margin_list = np.average(margin_list_all, axis=0)
    amp_list = np.average(amp_list_all, axis=0)
    th_all = np.average(th_all_all, axis=0)
    score_all = np.average(score_all_all, axis=0)
    
    num_margins, num_amps, num_comparisons = score_all.shape
    num_inputs = 6
    N = 5
       
    #th_all = np.average(th_all, axis=3) 
        
    topn = []
    amp_out = []
    margin_out = []
    topn_2d_final = []
    amp_2d_final = []
    for i in range(num_margins):
        amp_2d = []
        topn_2d = []
        for j in range(num_amps):
            out_val = 0
            for k in range(num_inputs):
                tmp_list = []
                for l in range(N):
                    actual_l = (k*N) + l
                    val = (margin_list[i], round(amp_list[j],2), round(score_all[i][j][actual_l],2))
                    val = val[2] - round(th_all[i][j][actual_l][2], 2)
                    tmp_list.append(val)
                indices = find(tmp_list, 'min')
                #dest = pairings[index].lower()
                out_val += match(src, indices, n)
            out_val = float(out_val)/N
            topn.append(out_val)
            topn_2d.append(out_val)
            amp_out.append(amp_list[j])
            amp_2d.append(amp_list[j])
            margin_out.append(margin_list[i])
        amp_2d_final.append(amp_2d)
        topn_2d_final.append(topn_2d)
    plot_2d(topn_2d_final, amp_2d_final, margin_list, f_n.replace('.npz', '_2d_top'+str(n)+'_avg.png'), folder_name, n)
    plot(topn, amp_out, margin_out, f_n.replace('.npz', '_top'+str(n)+'_avg.png'), folder_name, n)

if __name__ == "__main__":
    api = sys.argv[1]
    attack = sys.argv[2]
    model = sys.argv[3]
    n = 3
    for i in range(1,n+1):
        read(api, attack, model, i)
        read_average(api, attack, model, i)
    
