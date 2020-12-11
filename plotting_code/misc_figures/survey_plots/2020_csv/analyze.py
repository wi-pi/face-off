import os
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

types = ['PC', 'NPC']
gender = ['M', 'F']

path = os.getcwd()

def get_val(q1, q2, q3):
    val = [0,0,0,0,0]
    val[q1] += 1
    val[q2] += 1
    val[q3] += 1
    return val

q_l = ['SD', 'D', 'N', 'A', 'SA']

def plot(np_mat, amps, q_l, typ):
    fig, ax = plt.subplots()
    im = ax.imshow(np_mat, cmap='Greens')

    # We want to show all ticks...
    ax.set_xticks(np.arange(len(q_l)))
    ax.set_yticks(np.arange(len(amps)))
    
    # ... and label them with the respective list entries
    ax.set_xticklabels(q_l)
    ax.set_yticklabels(amps)

    # Rotate the tick labels and set their alignment.
    plt.setp(ax.get_xticklabels(), rotation=0, ha="right",rotation_mode="anchor")

    # Loop over data dimensions and create text annotations.
    for i in range(len(amps)):
        for j in range(len(q_l)):
            text = ax.text(j, i, np_mat[i, j], ha="center", va="center", color="black", fontsize=15)

    #ax.set_title("Harvest of local q_l (in tons/year)")
    fig.tight_layout()
    plt.savefig('2020_'+typ+'.png', bbox_inches='tight')

def print_normalized(d, cnt):
    keys = d.keys()
    rows = len(keys)
    amp = []
    cols = 5
    np_mat = np.zeros((rows, cols))
    for i, k in enumerate(keys):
        l = d[k]
        val = cnt[k]
        new_list = [round(float(e)/(val*3),2) for e in l]
        for j, element in enumerate(new_list):
            np_mat[i][j] = element
        print(k, new_list)
        amp.append(str(k))
    return amp, np_mat

for typ in types:
    print(typ)
    lines = []
    d = {}
    cnt = {}
    for g in gender:
        f_name = typ + '_' + g +'.csv'
        file_path = os.path.join(path, f_name)
        f = open(file_path, 'r')
        temp_lines = f.readlines()
        temp_lines = temp_lines[:-1]
        if lines == []:
            lines = temp_lines
        else:
            lines.extend(temp_lines)    
    for line in lines:
        elements = line.split(',')
        '''
        for i, e in enumerate(elements):
            print(i, e)
        '''
        amp = float(elements[19])
        q1 = int(elements[22])
        q2 = int(elements[23])
        q3 = int(elements[24])
        val = get_val(q1, q2, q3)
        if amp not in d:
            d[amp] = val
            cnt[amp] = 1
        else:
            t = d[amp]
            sum_list = [a + b for a, b in zip(t, val)]
            d[amp] = sum_list
            cnt[amp] += 1
    amps, np_mat = print_normalized(d, cnt)
    plot(np_mat, amps, q_l, typ)
