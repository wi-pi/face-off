import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import sys
import os
import numpy as np

ROOT = os.getcwd()

LEGENDSIZE = 16
FONTSIZE = 20

def normalize_list(input_list):
    max_element = max(input_list)
    if max_element == 0:
        max_element = 1
    norm = []
    for i in input_list:
        norm.append(float(i)/max_element)
    return norm

def plot_figure_2d(plt, x_list, y_list, amp_val, marg_val):

    out = normalize_list(y_list)
    plt.plot(x_list, out, marker='o', label=r'$\kappa=$'+str(marg_val)+','+r'$\alpha$='+str(amp_val))
    
    '''
    # c=match_target, cmap='Greens')
    plt.legend(loc='best', frameon=False, fontsize=LEGENDSIZE)
    plt.xlabel('Amplification (' + r'$\alpha$)', fontsize=FONTSIZE)
    #plt.ylabel('Match Target (%)')
    plt.ylabel('Normalized Success', fontsize=FONTSIZE)
    plt.savefig(file_name)
    '''




def aggregate(filename, attack_type):
    #file_name = 'large_triplet-large_center.csv'
    file_name = os.path.join(ROOT, os.path.join('csv', os.path.join(attack_type, filename)))
    '''
    #margin_list = [0,5,10,11,12]
    margin_list = [10,11,12]
    #amp_list = [1,2,3,4,5]
    amp_list = list(np.arange(1,5,0.2))
    '''
    f = open(file_name, 'r')
    lines = f.readlines()
    lines = lines[1:-1]

    write_path = os.path.join(ROOT, os.path.join('txt', attack_type))
    if os.path.exists(write_path) == False:
        os.mkdir(write_path)
    write_path = os.path.join(write_path, filename.replace('.csv', '-aggregate.txt'))
    w = open(write_path, 'w')
    amp_list = set()
    margin_list = set()
    num_iterations = set()
    for line in lines:
        elements = line.strip().split(',')
        margin_list.add(elements[11])
        amp_list.add(elements[12])
        num_iterations.add(elements[13])
    
    margin_list = list(margin_list)
    amp_list = list(amp_list)
    num_iterations = list(num_iterations)

    def convert_to_list(inp):
        for i, element in enumerate(inp):
            if '.' in element:
                inp[i] = float(element)
            else:
                inp[i] = int(element)

        return inp

    margin_list = convert_to_list(margin_list)
    amp_list = convert_to_list(amp_list)
    num_iterations = convert_to_list(num_iterations)

    amp_list.sort(reverse=True)
    margin_list.sort(reverse=True)
    num_iterations.sort(reverse=True)

    keys = {'barack':0, 'bill':1, 'jenn':2, 'leo':3, 'mark':4, 'matt':5, 'melania':6, 'meryl':7, 'morgan':8, 'taylor':9}
    d = {}
    n = len(lines)
    for line in lines:
        split = line.strip().split(',')
        margin = round(float(split[11]),2)
        amp = round(float(split[12]),2)
        num_iter = int(split[13])
        key = str(margin) + '_' + str(amp) + '_' + str(num_iter)
        src = split[4]
        target = split[7]
        match_src = split[5]
        match_target = split[6]
        
        if match_src.lower() == 'true':
            match_src = 1
        else:
            match_src = 0
        
        if match_target.lower() == 'true':
            match_target = 1
        else:
            match_target = 0

        if key not in d:
            d[key] = [match_src, match_target, 1]
        else:
            tmp = d[key]
            tmp[0] += match_src
            tmp[1] += match_target
            tmp[2] += 1
            d[key] = tmp

    final_output = []
    for num_iter in num_iterations:
        for amp in amp_list:
            for margin in margin_list:
                margin = round(margin, 2)
                amp = round(amp, 2)
                key = str(margin) + '_' + str(amp) + '_' + str(num_iter)
                n = d[key][2]
                match_src = round(float(d[key][0])/n * 100, 2)
                match_target = round(float(d[key][1])/n * 100, 2)
                string = "iter,"+str(num_iter)+",amp,"+str(amp)+",margin,"+str(margin)+",match_src,"+str(match_src)+",match_target,"+str(match_target)
                #print(string)
                w.write(string+"\n")
                final_output.append(string)

    w.close()

def process_txt(plt, file_name, attack_type, amp_val, marg_val, plot_type):

    read_path = os.path.join(ROOT, os.path.join('txt', os.path.join(attack_type, file_name.replace('.csv', '-aggregate.txt'))))
    f = open(read_path, 'r')
    lines = f.readlines()
    
    d = {}
    for line in lines:
        elements = line.split(',')
        n_iter = int(elements[1])
        amp = float(elements[3])
        margin = float(elements[5])
        match_src = float(elements[7])
        match_target = float(elements[9])
        if amp == amp_val and margin == marg_val:
            if n_iter not in d:
                d[n_iter] = [match_src, match_target, 1]
            else:
                tmp = d[n_iter]
                tmp[0] += match_src
                tmp[1] += match_target
                tmp[2] += 1
                d[n_iter] = temp

    keys = d.keys()
    keys.sort()
    match_src_list = []
    match_target_list = []
    for key in keys:
        match_src_list.append(float(d[key][0])/d[key][2])
        match_target_list.append(float(d[key][1])/d[key][2])

    print(keys, match_src_list)

    if plot_type == 'src':
        plot_figure_2d(plt, keys, match_src_list, amp_val, marg_val)
    else:
        plot_figure_2d(plt, keys, match_target_list, amp_val, marg_val)


    
   
if __name__ == "__main__": 
    aggregate(sys.argv[1], sys.argv[2])
    plt.clf()
    fig = plt.figure()
    amp_list = [1,2,3]
    marg_list = [0] 
    for amp_val in amp_list:
        for marg_val in marg_list:
            process_txt(plt, sys.argv[1], sys.argv[2], amp_val, marg_val, sys.argv[3])
    
    fig_name_super = os.path.join(ROOT, os.path.join('plots', sys.argv[2]))
    if os.path.isdir(fig_name_super) == False:
        os.mkdir(fig_name_super)
    fig_name = os.path.join(fig_name_super, sys.argv[1].replace('.csv', '-' + sys.argv[3] + '.png'))
    plt.legend(loc='best', frameon=False, fontsize=LEGENDSIZE)
    plt.xlabel('Number of Iterations', fontsize=FONTSIZE)
    plt.ylabel('Normalized Success', fontsize=FONTSIZE)
    plt.savefig(fig_name)
