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

OUT = os.getcwd() 

pairings = {0:'Barack', 1:'Leo', 2:'Matt', 3:'Melania', 4:'Morgan', 5:'Taylor'}

big_amps = []
big_margins = []
big_vals = []

def print_table(margins, amps, vals):
    #print(margins, amps, vals)
    amp_string = ' \t'
    for i, amp in enumerate(amps[0]):
       if i!=0:
           amp_string += ' & {\\bf ' + str(round(amp,2))+'}'
           amp_string += '\t'
    print(amp_string)

    for i, margin in enumerate(margins):
        temp_val = vals[i]
        string = '{\\bf '+str(round(margin,2)) + '} \t'
        for j, val in enumerate(temp_val):
            if j != 0:
                string += ' & ' + str(int(round(val,2))) + '\%\t'
        print(string)

def print_stats(topn, amp_out, margins, filename, folder_name, N=1, counter=-1):
    #print(filename)
    #print(amp_out)
    def sort(amp, val):
        for i in range(len(amp)):
            for j in range(i+1, len(amp)):
                if amp[i] >= amp[j]:
                    temp = amp[i]
                    amp[i] = amp[j]
                    amp[j] = temp

                    temp = val[i]
                    val[i] = val[j]
                    val[j] = temp

        return amp, val

    temp_amp = []
    temp_val = [] 
    for i, margin in enumerate(margins):
        amp = amp_out[i]
        val = topn[i]
        amp, val = sort(amp, val)
        temp_amp.append(amp)
        temp_val.append(val)
        '''
        N = len(amp)
        for j in range(N):
            if amp[j] != 0 and val[j] != 0:
                print(margin, round(amp[j],2), round(val[j],2))
        '''
    #print("counter:", counter)
    big_amps.append(temp_amp)
    big_margins.append(margins)
    big_vals.append(temp_val)
    if counter != 0 and counter%6 == 0 and counter != -1:
        big_amp = big_amps[n-1: (n-1)*6]
        big_margin = big_margins[n-1: (n-1)*6]
        big_val = big_vals[n-1: (n-1)*6]
 
        avg_amp = list(map(lambda x: sum(x)/len(x), zip(*big_amp)))
        avg_margins = list(map(lambda x: sum(x)/len(x), zip(*big_margin)))
        avg_val = list(map(lambda x: sum(x)/len(x), zip(*big_val)))
        print("top-",N)
        print_table(avg_margins, avg_amp, avg_val)
        print("\n")
    elif counter == -1:
        print_table(margins, temp_amp, temp_val)
        print("\n") 

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
        #print(filename, len(amp), len(top))
        plt.plot(amp, top, marker='o', label=r'$\kappa=$'+str(margin))
    plt.legend(loc='best', frameon=False, fontsize=LEGENDSIZE)
    plt.xlabel('Amplification (' + r'$\alpha$)', fontsize=FONTSIZE)
    plt.ylabel('Matching Success (%)', fontsize=FONTSIZE)
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
    #ax.set_zlabel('Top'+str(N)+' Failure', fontsize=FONTSIZE)     
    ax.set_zlabel('Matching Success (%)', fontsize=FONTSIZE)     
    plt.savefig(file_path)  

def read_average2(api, attack, model, n, flag, counter_start):
    #print("counter_start", counter_start)
    folder_name = os.path.join(ROOT, os.path.join(api, os.path.join(attack, model)))
    folder_name = os.path.join(folder_name, os.path.join('hinge_loss', 'npz_mean'))
    #folder_name = os.path.join(folder_name, os.path.join('hinge_loss', 'npz'))
    files = os.listdir(folder_name)
    print(folder_name, files)
    
    if len(files) == 0:
        exit()
 
    def find(l, search_type='min'):
    	#returns indices sorted in increasing order of values in l
    	#larger the value in l, more confident the classifier is of its prediction
        n = len(l)
        indices = np.argsort(l)
        indices = list(indices)
        final = []
        count = 0
        for index in indices:
            if l[index] >= 0:
            # or l[index] <= 0:
                count += 1
                final.append(index)
        if count == 0:
            final = []
            #indices
        #print(l, final)
        return final
        
        
    def match(inp1, indices, n=1):
        matchings = []
        if indices == []:
            return 0
        for index in indices:
            matchings.append(pairings[index].lower())
        len_matchings = len(matchings)
        if len_matchings < n:
            final = matchings
        else:
            final = matchings[len_matchings - n:]
        #print(inp1, final)
        if inp1 in final:
            return 1
        else:
            #return 0
            #print("matchings:", matchings)
            #print("src:", inp1, ";top-",n," candidates for matching:",final)
            return 0
    
    def load_content(fil, flag):
        elements = fil.split('.')[0].split('_')
        src = elements[-3]
        tmp = np.load(file_name)
        #['th_self_crop', 'margin_list', 'score_target_crop', 'score_all', 'th_self', 'score_self', 'th_all_crop', 'th_target', 'th_all', 'score_all_crop', 'th_target_crop', 'score_target', 'l2_mtx', 'amp_list', 'score_self_crop']
        
        margin_list = tmp['margin_list']
        score_all = tmp['score_all']
        if flag == 0:
            if 'th_all' in tmp:
                th_all = tmp['th_all']
            else:
                exit()
        else:
            #print("Returning None")
            th_all = None
        amp_list = tmp['amp_list']

        #print("amp_list:", amp_list)
        return src, margin_list, score_all, th_all, amp_list

    amp_2d_final2 = []
    topn_2d_final2 =[]


    #amp_np = np.zeros((len(files), 3, 11))
    #topn_np = np.zeros((len(files), 3, 11))
    
    amp_final = []
 
    #print(files)
    keys = pairings.keys()
    for key in keys:
        final_files = {0}
        src_label = pairings[key].lower()
        f_n = ''
        for fil in files:
            tmp_elements = fil.split('_')
            len_tmp_elements = len(tmp_elements) - 2
            src_from_fil = tmp_elements[-3]
            if src_from_fil == src_label:
                #print(src_label, src_from_fil)
                if f_n == '':
                    for i in range(len_tmp_elements):
                        f_n += tmp_elements[i]
                        if i != len_tmp_elements-1:
                            f_n += '_'
                final_files.add(fil)
        f_n += '.npz'
        final_files.remove(0)
        final_files = list(final_files)
        #print(src_label, final_files)
        if len(final_files) != 0:
            counter_start += 1
            topn_np = np.zeros((len(final_files), 3, 13))
            amp_np = np.zeros((len(final_files), 3, 13))
            for fil_num, fil in enumerate(final_files):
                file_name = os.path.join(folder_name, fil)
                src, margin_list, score_all, th_all, amp_list = load_content(fil, flag)
                 
                
                num_margins, num_amps, num = score_all.shape
                #print(len(amp_list), num_amps)
                if num_amps != len(amp_list):
                    #print("breaking")
                    break
                #print("not breaking")
                N = 5
                num_comparisons = num/N
                num_inputs = int(num_comparisons)
        
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
                        tmp_list_2 = []
                        for k in range(num_inputs): #for each image
                            tmp_list = []
                            for l in range(N): #for each of the 6 images it is compared to
                                actual_l = (k*N) + l
                                if flag == 0:
                                    val = round(score_all[i][j][actual_l],2) - round(th_all[i][j][actual_l][2], 2)
                                else:
                                    val = round(score_all[i][j][actual_l],2) - (0.5*flag)
                                tmp_list.append(val)
                            tmp_val = sum(tmp_list)/len(tmp_list) #averaged value per label
                            tmp_list_2.append(tmp_val)
                        
                        indices = find(tmp_list_2, 'min')
                    
                        #dest = pairings[index].lower()
                        out_val = match(src, indices, n)
                        #out_val = float(out_val)/N

                        topn_np[fil_num][i][j] = out_val * 100
                        #print("adding:", amp_list[j])
                        amp_np[fil_num][i][j] = amp_list[j]
                    amp_final.append(amp_list)

            average_val = np.mean(topn_np, axis=0)
            average_amp = np.mean(amp_np, axis=0)
            '''
            average_val_list = []
            #print("num_amps:", num_amps, "amp_list:", len(amp_list))
            for i in range(3):
                t1 = list(average_val[i])
                t2 = t1[:len(amp_list)]
                print("T2:", len(t2), "amp_list:", len(amp_list))
                average_val_list.append(t2)
                print("x:", len(average_val_list[i]), "y:", len(amp_final[i]))
            print("\n")
            '''
            print_stats(average_val, average_amp, margin_list, f_n.replace('.npz', '_2d_top'+str(n)+'_avg_mean.png'), folder_name, n, counter_start)

def read(api, attack, model, n, flag = 1):
    folder_name = os.path.join(ROOT, os.path.join(api, os.path.join(attack, model)))
    folder_name = os.path.join(folder_name, os.path.join('hinge_loss', 'npz_mean'))
    files = os.listdir(folder_name)
   
    if len(files) == 0:
        exit()
 
    def find(l, search_type='min'):
    	#returns indices sorted in increasing order of values in l
    	#larger the value in l, more confident the classifier is of its prediction
        n = len(l)
        indices = np.argsort(l)
        indices = list(indices)
        final = []
        count = 0
        for index in indices:
            if l[index] >= 0 or l[index] <= 0:
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
            final = matchings
        else:
            final = matchings[len_matchings - n:]
        if inp1 in final:
            return 1
        else:
            #return 0
            #print("matchings:", matchings)
            #print("src:", inp1, ";top-",n," candidates for matching:",final)
            return 0
    for fil in files:
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
        if flag == 0:
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
        num_margins, num_amps, num = score_all.shape
        N = 5
        num_comparisons = num/N
        num_inputs = int(num_comparisons)        

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
                tmp_list_2 = []
                for k in range(num_inputs): #for each image
                    tmp_list = []
                    for l in range(N): #for each of the 6 images it is compared to
                        actual_l = (k*N) + l
                        val = (margin_list[i], round(amp_list[j],2), round(score_all[i][j][actual_l],2))
                        if flag == 0:
                            val = val[2] - round(th_all[i][j][actual_l][2], 2)
                        else:
                            val = val[2] - (0.5 * flag)
                        tmp_list.append(val)
                    tmp_val = sum(tmp_list)/len(tmp_list) #averaged value per label
                    tmp_list_2.append(tmp_val)
                indices = find(tmp_list_2, 'min')
                #dest = pairings[index].lower()
                out_val = match(src, indices, n) * 100
                #out_val = float(out_val)/N
                topn.append(out_val)
                topn_2d.append(out_val)
                amp_out.append(amp_list[j])
                amp_2d.append(amp_list[j])
                margin_out.append(margin_list[i])
            amp_2d_final.append(amp_2d)
            topn_2d = topn_2d
            topn_2d_final.append(topn_2d)
        #print(topn_2d_final)
        plot_2d(topn_2d_final, amp_2d_final, margin_list, fil.replace('.npz', '_2d_top'+str(n)+'_mean.png'), folder_name, n)
        plot(topn, amp_out, margin_out, fil.replace('.npz', '_top'+str(n)+'_mean.png'), folder_name, n)

'''
def read_average(api, attack, model, n):
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
            final = matchings
        else:
            final = matchings[len_matchings - n:]
        print("matchings:", matchings)
        print("src:", inp1, ";top-",n," candidates for matching:",final)
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
'''

if __name__ == "__main__":
    api = sys.argv[1]
    attack = sys.argv[2]
    model = sys.argv[3]
    flag = 0
    if api == 'azure':
        flag = 1
    elif api == 'awsverify':
        flag = 100
    #print(api, flag)
    n = 3
    for i in range(3,n+1):
        counter_start = (i-1) * 6
        #read(api, attack, model, i, flag)
        read_average2(api, attack, model, i, flag, counter_start)
    
