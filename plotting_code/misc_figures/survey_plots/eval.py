import sys
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
#import seaborn as sns
import scipy.stats as stats

sphinx_gallery_thumbnail_number = 2


def print_range(val, name):
    if 'bike' not in name:
        if 1.2 <= val < 1.4:  # in range(1.2,1.4):
            return 6
        elif 1.4 <= val < 1.6:  # in range(1.4,1.6):
            return 0
        elif 1.6 <= val < 1.8:  # in range(1.6,1.8):
            return 1
        elif 1.8 <= val < 2.0:  # in range(1.8,2):
            return 2
        elif 2.0 <= val < 2.2:  # in range(2,2.2):
            return 3
        elif 2.2 <= val <= 2.4:  # in range(2.2,2.4):
            return 4
    else:
        if 1 <= val < 2:
            return 0
        elif 2 <= val < 3:
            return 1
        elif 3 <= val < 4:
            return 2
        elif 4 <= val <= 5:
            return 3


def bucketize(input1, input2):
    with open(input1, 'r') as f:
        f1 = f.read().strip('\t').splitlines()

    with open(input2, 'r') as f:
        f2 = f.read().strip('\t').splitlines()

    name = input1.split('_')[0]
    #print name
    l1 = []
    for e in f1:
        for element in e.split('\t'):
            l1.append(float(element))

    l2 = []
    for e in f2:
        for element in e.split('\t'):
            l2.append(float(element))

    #print len(l1)
    #print len(l2)
    rows = 5
    cols = 5
    if 'bike' in name:
        rows = 4
        cols = 5
    output = []
    for row in range(rows):
        output += [[0]*cols]

    for i in range(0,len(l2)):
        #print l2[i]
        i1 = int(print_range(l2[i],name))
        i2 = int(l1[i] - 1)
        #print l2[i], i1
        #print l1[i], i2
        output[i1][i2] = output[i1][i2] + 1

    return name, output
    #merge = [e1 + e2 for e1,e2 in zip(f1,f2)]


def plot_matrices_user_study(input1, input2):
    name, op_list = bucketize(input1, input2)
    array = np.asarray(op_list, dtype=np.float32)
    print(np.sum(array))
    # array = np.round(array/np.sum(array),2)
    # cols = ["Strongly Agree", "Agree", "Neither", "Disagree", "Strongly Disagree"]
    cols = ["SA", "A", "N", "D", "SD"]
    rows = ["[1.4,1.6)","[1.6,1.8)","[1.8,2)","[2,2.2)","[2.2,2.4]"]
    if 'bike' in name :
        rows = ["[1,2)","[2,3)","[3,4)","[4,5]"]

    fig, ax = plt.subplots()
    im = ax.imshow(array, cmap='Greens')

    ax.set_xticks(np.arange(len(cols)))
    ax.set_yticks(np.arange(len(rows)))
    ax.set_xticklabels(cols, fontsize=15)
    ax.set_yticklabels(rows, fontsize=15)

    # Rotate the tick labels and set their alignment.
    #plt.setp(ax.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor", fontsize=15)

    # Loop over data dimensions and create text annotations.
    for i in range(len(rows)):
        for j in range(len(cols)):
            text = ax.text(j, i, array[i, j], ha="center", va="center", color="black", fontsize=15)
    if "bike" in name:
        img = 'Background'
    else:
        img = 'Portrait'
    # ax.set_title("Normalized responses to "+img+"-type images")
    fig.tight_layout()
    plt.savefig(name+".pdf", bbox_inches='tight')
    plt.savefig(name+".png", bbox_inches='tight')
    return array


if __name__ == "__main__":
    # plot_matrices_user_study('bike_scores.csv', 'bike_amp.csv')
    arr_pc_p = plot_matrices_user_study('pc_scores.csv', 'pc_amp.csv')
    arr_npc_p = plot_matrices_user_study('npc_scores.csv', 'npc_amp.csv')
    arr_npc_b = plot_matrices_user_study('npcbike_scores.csv', 'npcbike_amp.csv')
    arr_pc_b = plot_matrices_user_study('pcbike_scores.csv', 'pcbike_amp.csv')

    print(arr_npc_b)
    print(arr_pc_b)
    for i in range(len(arr_npc_p)):
        print(stats.chi2_contingency(observed=[arr_pc_p[i, :], arr_npc_p[i, :]]))

    for i in range(len(arr_npc_b)):
        print(stats.chi2_contingency(observed=[arr_pc_b[i, :], arr_npc_b[i, :]]))

    print(stats.chi2_contingency(observed=[arr_pc_p[0, :] + arr_pc_p[1, :]+ arr_pc_p[2, :] +
                                           arr_npc_p[0, :] + arr_npc_p[1, :] + arr_npc_p[2, :], arr_npc_b[0, :]+arr_pc_b[0, :]]))

    print(stats.chi2_contingency(observed=[arr_pc_p[0, :] + arr_pc_p[1, :]+ arr_pc_p[2, :], arr_pc_b[0, :]]))

    print(stats.chi2_contingency(observed=[arr_npc_p[0, :] + arr_npc_p[1, :] + arr_npc_p[2, :],
                                           arr_npc_b[0, :]]))


