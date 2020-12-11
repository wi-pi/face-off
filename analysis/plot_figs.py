import sys
import os
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

SMALL = 1.0
FONTSIZE = 14
LEGENDSIZE = 14

def list_div(l1, l2):
    res = [i / j for i, j in zip(l1, l2)] 
    return res

def normalize_list(raw):
    norm = [float(i)/max(raw) for i in raw]
    return norm

def process(lines):
    amp_list = []
    margin_list = []
    match_src_list = []
    match_target_list = []
    for line in lines:
        elements = line.strip().split(',')
        amp = float(elements[1])
        margin = float(elements[3])
        match_src = float(elements[5])
        match_target = float(elements[7])
        if match_src < 1:
            match_src = SMALL
        amp_list.append(amp)
        margin_list.append(margin)
        match_src_list.append(match_src)
        match_target_list.append(match_target)
    
    return amp_list, margin_list, match_src_list, match_target_list


def process2(lines):
    amp_list = []
    margins = set()
    match_target_list = []
    match_src_list = []
    for line in lines:
        elements = line.strip().split(',')
        margin = float(elements[3])
        margins.add(margin)

    margins = list(margins)
    for m in margins:
        t1 = []
        t2 = []
        t3 = []
        for line in lines:
            elements = line.strip().split(',')
            amp = float(elements[1])
            margin = float(elements[3])
            match_src = float(elements[5])
            match_target = float(elements[7])
            if match_src < 1:
                match_src = SMALL
            if margin == m:
                t1.append(amp)
                t2.append(match_src)
                t3.append(match_target)
        amp_list.append(t1)
        match_src_list.append(t2)
        match_target_list.append(t3)
  
    return margins, amp_list, match_src_list, match_target_list

def plot_figure_3d(amp, margin, match_list, file_name, inverse=False):
    plt.clf()
    fig = plt.figure()
    ax = fig.gca(projection='3d')
    #out = list_div(match_target, match_src)
    out = normalize_list(match_list)
    #if inverse:
    #    out = [ 1 - i for i in out ]
    ax.scatter3D(amp, margin, out)
    # c=match_target, cmap='Greens')
    ax.set_xlabel('Amplification (' + r'$\alpha$)', fontsize=FONTSIZE)
    ax.set_ylabel('Margin ('+r'$\kappa$)', fontsize=FONTSIZE)
    #ax.set_zlabel('Match Target (%)')
    ax.set_zlabel('Normalized Success', fontsize=FONTSIZE)
    plt.savefig(file_name)    

def plot_figure_2d(margins, amp_list, match_lists, file_name, inverse=False):
    plt.clf()
    fig = plt.figure()
 
    margins.sort()   
    for i, margin in enumerate(margins):
        amp = amp_list[i]
        match_list = match_lists[i]
        #match_src = match_src_list[i]
        #out = list_div(match_target, match_src)
        out = normalize_list(match_list)
        #if inverse:
        #    out = [1 - i for i in out]
        plt.plot(amp, out, marker='o', label=r'$\kappa=$'+str(margin))
    # c=match_target, cmap='Greens')
    plt.legend(loc='best', frameon=False, fontsize=LEGENDSIZE)
    plt.xlabel('Amplification (' + r'$\alpha$)', fontsize=FONTSIZE)
    #plt.ylabel('Match Target (%)')
    plt.ylabel('Normalized Success', fontsize=FONTSIZE)
    plt.savefig(file_name)

def main_func(txt_files, data_path, plot_path):
    for txt_file in txt_files:
        #amp = [], margin = [], match_src = [], match_target = []
        if 'whitebox' in txt_file:
            f = open(os.path.join(data_path, txt_file))
            lines = f.readlines()
            lines = lines[:-3]
            amp, margin, match_src, match_target = process(lines)
            margins, amp_list, match_src_list, match_target_list = process2(lines)
            fig_file = txt_file.replace('.txt', '-src.png')
            fig_path_3d = os.path.join(plot_path, fig_file)
            fig_path_2d = os.path.join(plot_path, fig_file.replace('.png', '-2d.png'))
            #print("amp:", amp, "margin:", margin, "match_target:", match_target)
            
            #plot_figure_3d(amp, margin, match_src, fig_path_3d)
            plot_figure_2d(margins, amp_list, match_src_list, fig_path_2d, True)
            
            fig_path_3d = fig_path_3d.replace('src', 'target')
            fig_path_2d = fig_path_2d.replace('src', 'target')
            
            #plot_figure_3d(amp, margin, match_target, fig_path_3d, True)
            plot_figure_2d(margins, amp_list, match_target_list, fig_path_2d)

def constrain(input_list):
    output = []
    for item in input_list:
        if 'whitebox_eval_large_triplet-large_center_casia' in item:
            output.append(item)
    return output


if __name__ == "__main__":
	ROOT = os.getcwd()
	attack_type = sys.argv[1]
        plot_path = os.path.join(ROOT, os.path.join('plots', attack_type))
	data_path = os.path.join(ROOT, os.path.join('txt', attack_type))
        if os.path.exists(plot_path) == False:
            os.mkdir(plot_path)
	txt_files = os.listdir(data_path)
        txt_files = constrain(txt_files)
        main_func(txt_files, data_path, plot_path)
