import numpy as np
import os

import matplotlib
matplotlib.use('Agg')

import matplotlib.pyplot as plt
import sys
from mpl_toolkits.axes_grid1.inset_locator import zoomed_inset_axes
from mpl_toolkits.axes_grid1.inset_locator import mark_inset
from matplotlib.ticker import FormatStrFormatter

FONT_SIZE = 22
LABEL_SIZE = 15
LEGEND_SIZE = 18

#not used in paper (??)
def plot_xfer_JPG_PNG(npz_filename, npz_dir_path, img_dir_path, attack_alg, test_num, api_name):
    plt.close('all')
    fig, ax = plt.subplots()

    npzfile = np.load(npz_dir_path + npz_filename + '.npz')

    l2mtx = npzfile['l2_mtx']
    scores_self = npzfile['score_self']
    scores_self_jpg = npzfile['score_self_jpg']
    #scores_self_cropped = npzfile['score_self_crop']
    scores = scores_self
    # if test_num == '1':
    #     scores = scores_self
    # else:
    #     scores = scores_self_cropped
    amp_list = npzfile['amp_list']
    print(scores)
    # if api_name == 'awsverify':
    #     scores[scores == 999] = 0
    #     scores[scores == -99] = 0
    #     scores = scores.astype(float)
    #     scores = scores / 100
    # if api_name == 'facepp':
    #     if test_num == '1':
    #      threshold = npzfile['th_self']
    #     else:
    #      threshold = npzfile['th_self_crop']
    #     print(scores)
    #     print(threshold)
    #     threshold[threshold < 0] = -1 * threshold[threshold < 0]
    #     scores[scores < 0] = -1 * scores[scores < 0]
    #     scores = scores.astype(float)
    #     threshold = threshold.astype(float)
    #     threshold = np.average(threshold,axis=3)
    #     scores = np.divide(scores,threshold) * 0.5
    #     #scores = scores / 100
    margin_list = npzfile['margin_list']

    plt.ylim([-0.2, 0.2])
    for i, val in enumerate(margin_list):
        print(val)
        if not (5.5 < val < 5.8):
            continue
        # if round(val * 10) % 10 != 0 or val == 0:  # not in [1,2,3,4,5]:
        #    continue

        if attack_alg == 'pgd':
            lab_str = r'$\epsilon$=' + "%0.1f" % (val)
            # print(val)

        elif attack_alg == 'cw':
            lab_str = r'$\kappa$=' + "%0.1f" % (val)
        scores[scores == -1] = 0
        print_scores = np.average(scores[i] - scores_self_jpg[i], axis=1)
        print(print_scores)
        # print(amp_list)
        if 'ds4' in npz_filename:
            factor = 0.387
        else:
            factor = 2.228

        if test_num == '1':
            l2norm = l2mtx[i]*factor
        else:
            l2norm = l2mtx[i]

        ax.plot(amp_list, print_scores, label=lab_str, linewidth=1.5)
        #ax2 = ax.twiny()
        #ax2.plot(amp_list, print_scores, label=lab_str + 'AMP', linewidth=2.5)

        #ax.set_xlabel('L2 norm of Perturbation', fontsize=FONT_SIZE)
        ax.set_xlabel('Amplification Factor', fontsize=FONT_SIZE)
        ax.set_ylabel('Diff. Matching Confidence', fontsize=FONT_SIZE)
        ax.text(2.5, 0.15, 'JPG Region', fontsize=FONT_SIZE)
        ax.text(2.5, -0.15, 'PNG Region', fontsize=FONT_SIZE)
    ax.tick_params(axis='both', which='major', labelsize=LABEL_SIZE)
        # plt.grid()

    # plt.plot(l2mtx[i], 0.5*np.ones((len(l2mtx[i]))), color='black', linewidth=2.5, linestyle='--')
    plt.axhline(y=0, color='black', linewidth=2.5, linestyle='--')

    # plt.legend(loc='best', prop={'size': 12})

    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)
    plt.savefig(img_dir_path + npz_filename + '_user_study.png', bbox_inches='tight')
    # print(min_l2)
    # print(l2mtx.shape)


def plot_xfer_black_user_study(npz_filename, npz_dir_path, img_dir_path, attack_alg, test_num, api_name):
    plt.close('all')
    fig, ax = plt.subplots()

    npzfile = np.load(npz_dir_path + npz_filename + '.npz')

    l2mtx = npzfile['l2_mtx']
    amp_list = npzfile['amp_list']
    margin_list = npzfile['margin_list']
    scores_self = npzfile['score_self']
    scores_self_cropped = npzfile['score_self_crop']
    
    if test_num == '1':
        scores = scores_self
    else:
        scores = scores_self_cropped
    
    
    if api_name == 'awsverify':
        scores[scores == 999] = 0
        scores[scores == -99] = 0
        scores = scores.astype(float)
        scores = scores / 100
    
    if api_name == 'facepp':
        if test_num == '1':
         threshold = npzfile['th_self']
        else:
         threshold = npzfile['th_self_crop']
        print(scores)
        print(threshold)
        threshold[threshold < 0] = -1 * threshold[threshold < 0]
        scores[scores < 0] = -1 * scores[scores < 0]
        scores = scores.astype(float)
        threshold = threshold.astype(float)
        threshold = np.average(threshold,axis=3)
        scores = np.divide(scores,threshold) * 0.5
        #scores = scores / 100
    

    plt.ylim([0, 1])
    for i, val in enumerate(margin_list):
        #print(val)
        if val < 5.8:
            continue

        if attack_alg == 'pgd':
            lab_str = r'$\epsilon$=' + "%0.1f" % (val)

        elif attack_alg == 'cw':
            lab_str = r'$\kappa$=' + "%0.1f" % (val)
        
        scores[scores == -1] = 0
        print_scores = np.average(scores[i], axis=1)
        #print(print_scores)
        
        if 'ds4' in npz_filename:
            factor = 0.387
        else:
            factor = 2.228

        if test_num == '1':
            l2norm = l2mtx[i]*factor
        else:
            l2norm = l2mtx[i]

        ax.plot(l2norm, print_scores, label=lab_str, linewidth=2.5, color='red')
        ax2 = ax.twiny()
        ax2.plot(amp_list, print_scores, label=lab_str + 'AMP', linewidth=2.5)

        ax.set_xlabel('L2 norm of Perturbation', fontsize=FONT_SIZE)
        ax2.set_xlabel('Amplification Factor', fontsize=FONT_SIZE)
        ax.set_ylabel('Matching Confidence', fontsize=FONT_SIZE)

        ax.tick_params(axis='both', which='major', labelsize=LABEL_SIZE)
        ax2.tick_params(axis='both', which='major', labelsize=LABEL_SIZE)
        # plt.grid()

    # plt.plot(l2mtx[i], 0.5*np.ones((len(l2mtx[i]))), color='black', linewidth=2.5, linestyle='--')
    plt.axhline(y=0.5, color='black', linewidth=2.5, linestyle='--')

    # plt.legend(loc='best', prop={'size': 12})

    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)
    ax.tick_params(axis='both', which='major', labelsize=LABEL_SIZE)
    #ax2.tick_params(axis='both', which='major', labelsize=LABEL_SIZE)
    plt.savefig(img_dir_path + npz_filename + '_user_study.png', bbox_inches='tight')
    # print(min_l2)
    # print(l2mtx.shape)

#don't think this function is useful
def plot_xfer_black(npz_filename, npz_dir_path, img_dir_path, attack_alg, test_num, api_name):

    plt.close('all')
    fig, ax = plt.subplots()
    axins = zoomed_inset_axes(ax, 3.5, loc='upper right')  # zoom-factor: 2.5, location: upper-left
    npzfile = np.load(npz_dir_path + npz_filename + '.npz')

    l2mtx = npzfile['l2_mtx']
    scores_self = npzfile['score_self']
    scores_self_cropped = npzfile['score_self_crop']
    if test_num == '1':
     scores = scores_self
    else:
     scores = scores_self_cropped
    amp_list = npzfile['amp_list']
    print(scores)
    if api_name =='awsverify':
     scores[scores == 999]=0
     scores[scores == -99] = 0
     scores = scores.astype(float)
     scores = scores/100
    if api_name == 'facepp':
        if test_num == '1':
         threshold = npzfile['th_self']
        else:
         threshold = npzfile['th_self_crop']
        print(scores)
        print(threshold)
        threshold[threshold < 0] = -1 * threshold[threshold < 0]
        scores[scores < 0] = -1 * scores[scores < 0]
        scores = scores.astype(float)
        threshold = threshold.astype(float)
        threshold = np.average(threshold, axis=3)
        print(threshold.shape)
        print(scores.shape)
        scores = np.divide(scores,threshold) * 0.5
        #scores = scores / 100
    margin_list = npzfile['margin_list']
    
    plt.ylim([0, 1])
    for i, val in enumerate(margin_list):

        if round(val * 10) % 10 != 0 or val == 0:  # not in [1,2,3,4,5]:
            continue

        if attack_alg == 'pgd':
            lab_str = r'$\epsilon$=' + "%0.1f" % (val)
            #print(val)

        elif attack_alg == 'cw':
            lab_str = r'$\kappa$=' + "%0.1f" % (val)
        scores[scores==-1]=0
        print_scores = np.average(scores[i], axis=1)
        print(print_scores)
        #print(amp_list)
        axins.plot(l2mtx[i], print_scores)

        ax.plot(l2mtx[i], print_scores, label=lab_str, linewidth=2.5)
        #ax2 = ax.twiny()
        # ax2.plot(amp_list, print_scores, label=lab_str+'AMP', linewidth=2.5)

        ax.set_xlabel('L2 norm of Perturbation', fontsize=FONT_SIZE)
        # ax2.set_xlabel('Amplification Factor', fontsize=FONT_SIZE)
        ax.set_ylabel('Matching Confidence', fontsize=FONT_SIZE)
        # plt.grid()

    axins.set_xlim(7, 12)  # apply the x-limits
    axins.set_ylim(0.28, 0.34)  # apply the y-limits


    # plt.plot(l2mtx[i], 0.5*np.ones((len(l2mtx[i]))), color='black', linewidth=2.5, linestyle='--')


    ax.legend(loc='upper left', prop={'size': 12})

    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)
    mark_inset(ax, axins, loc1=2, loc2=4, fc="none", ec="0.5")
    ax.axhline(y=0.5, color='black', linewidth=2.5, linestyle='--')
    ax.set_ylim(0, 1)
    plt.savefig(img_dir_path + npz_filename + '.png', bbox_inches='tight')
    # print(min_l2)
    # print(l2mtx.shape)


def plot_xfer_black_withzoom(npz_filename, npz_dir_path, img_dir_path, attack_alg, test_num, api_name, zoom=True, two_axes=False):
    plt.close('all')
    
    if zoom:
        f, axs = plt.subplots(2, 2, figsize=(5.5, 6))
        ax = plt.subplot(2,1,2)
        ax2 = plt.subplot(2, 1, 1)
    else:
        fig, ax = plt.subplots()
        ax = plt.subplot(111)
        if two_axes:
            ax2 = ax.twiny()

    npzfile = np.load(npz_dir_path + npz_filename + '.npz')

    l2mtx = npzfile['l2_mtx']
    amp_list = npzfile['amp_list']
    margin_list = npzfile['margin_list']
    scores_self = npzfile['score_self']
    scores_self_cropped = npzfile['score_self_crop']
   
    if test_num == '1':
        scores = scores_self
    else:
        scores = scores_self_cropped
    
    if api_name == 'awsverify':
        scores[scores == 999] = 0
        scores[scores == -99] = 0
        scores = scores.astype(float)
        scores = scores / 100
    
    if api_name == 'facepp':
        if test_num == '1':
            threshold = npzfile['th_self']
        else:
            threshold = npzfile['th_self_crop']
        #print(scores)
        #print(threshold)
        threshold[threshold < 0] = -1 * threshold[threshold < 0]
        scores[scores < 0] = -1 * scores[scores < 0]
        scores = scores.astype(float)
        threshold = threshold.astype(float)
        threshold = np.average(threshold, axis=3)
        #print(threshold.shape)
        #print(scores.shape)
        scores = np.divide(scores, threshold) * 0.5
        # scores = scores / 100

    plt.ylim([0, 1])
    min_val = 1
    max_val = 0

    for i, val in enumerate(margin_list):

        if round(val * 10) % 10 != 0 or val == 0:  # not in [1,2,3,4,5]:
            continue

        if attack_alg == 'pgd':
            lab_str = r'$\epsilon$=' + "%0.1f" % (val)
            # print(val)

        elif attack_alg == 'cw':
            lab_str = r'$\kappa$=' + "%0.1f" % (val)
        scores[scores == -1] = 0
        print_scores = np.average(scores[i], axis=1)
        print_scores[print_scores<0.01] = None

        if test_num == '1':
            l2norm = l2mtx[i]*2.228
        else:
            l2norm = l2mtx[i]

        ax.plot(l2norm, print_scores, label=lab_str, linewidth=1)
        if zoom:
            ax2.plot(l2norm, print_scores, label=lab_str, linewidth=1)
        min_val = min(min_val,np.min(print_scores))
        max_val = max(max_val, np.max(print_scores))
        

        ax.set_xlabel('Perturbation Norm (' + r'$||\alpha.\delta||_2$)', fontsize=FONT_SIZE)
        ax.set_ylabel('Matching Confidence', fontsize=FONT_SIZE)
   
    if two_axes:
        start = ax.get_xlim()[0]
        end = ax.get_xlim()[1]
        n = amp_list.shape[0]
        amp_list = list(amp_list)
        n = len(amp_list)
        space = n/5
        final = []
        for i in range(5):
            final.append("%.1f" % amp_list[i*space])
        locations = np.linspace(start, end, num=5, endpoint=False)
        ax2.set_xlim(ax.get_xlim())
        #print(final)
        ax2.set_xticks(locations)
        ax2.set_xticklabels(final)
        ax2.set_xlabel('Amplification (' + r'$\alpha$)', fontsize=FONT_SIZE)
        plt.setp(ax2.get_xticklabels(), fontsize=17)
    
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)
    ax.tick_params(axis='both', which='major', labelsize=LABEL_SIZE)
    ax.axhline(y=0.5, color='black', linewidth=2.5, linestyle='--')
    ax.set_ylim(0, 1)
    if zoom:
        ax.get_yaxis().set_label_coords(-0.1, 1.02)
    else:
        ax.legend(loc='best', frameon=False, prop={'size': LEGEND_SIZE}, ncol=1)
    if zoom:
        if 'matt_03_leo_aws_verify_center_hing_ds3_2' in npz_filename:
            ax2.legend(loc='upper center', prop={'size': LEGEND_SIZE}, ncol=1, frameon=False, bbox_to_anchor=(0.2, 0.65))
        elif 'matt_03_leo_azure_center_hing_ds3_2' in npz_filename:
            ax2.legend(loc='upper center', prop={'size': LEGEND_SIZE}, ncol=1, frameon=False, bbox_to_anchor=(0.2, 0.45))
        else:
            ax2.legend(loc='upper center', prop={'size': LEGEND_SIZE}, ncol=1, frameon=False, bbox_to_anchor=(0.2, 0.35))

        ax2.spines['right'].set_visible(False)
        ax2.spines['top'].set_visible(False)
        ax2.spines['bottom'].set_visible(False)

        ax2.tick_params(axis='both', which='major', labelsize=LABEL_SIZE)

        if min_val<0:
            min_val =0

        ax2.set_ylim(min_val, max_val)
        ax2.yaxis.set_major_formatter(FormatStrFormatter('%.1f'))
        ax2.get_xaxis().set_ticks([])
    
    img_save_path = img_dir_path + npz_filename + '.png'
    if zoom:
        img_save_path = img_dir_path + npz_filename + '_zoom.png'
    if two_axes:
        img_save_path = img_save_path.replace('.png', '_2axes.png')
    
    plt.savefig(img_save_path, bbox_inches='tight')


if __name__ == "__main__":
    # plot_xfer_black('cw', 'cifar')
    model = sys.argv[1]
    loss_func = sys.argv[2]
    dataset_num = sys.argv[3]
    test_num = sys.argv[4]
    api_name = sys.argv[5]
    attack_alg = sys.argv[6]
    if loss_func == 'target':
        loss_func_short = 'targ'
    else:
        loss_func_short = 'hing'
    if api_name == 'awsverify':
     api_name_1 = 'awsverify'
    else:
     api_name_1 = api_name
    npz_dir_path = './api_results/'+api_name+'/dataset'+dataset_num+'/'
    img_dir_path = './plots/'+api_name+'/dataset'+dataset_num+'/'

    if os.path.isdir(img_dir_path) == False:
        os.makedirs(img_dir_path) 
    npz_filename =  'matt_03_leo_'+api_name_1+'_'+model+'_'+loss_func_short+'_ds'+dataset_num+'_'+test_num

    if not os.path.exists(npz_dir_path + npz_filename + '.npz'):
        print(npz_dir_path + npz_filename + '.npz does not exist')
        exit()

    plot_xfer_black_withzoom(npz_filename, npz_dir_path, img_dir_path, attack_alg, test_num, api_name, True, False)           
    plot_xfer_black_withzoom(npz_filename, npz_dir_path, img_dir_path, attack_alg, test_num, api_name, False, False)           
    plot_xfer_black_withzoom(npz_filename, npz_dir_path, img_dir_path, attack_alg, test_num, api_name, False, True)           
    


    # plot_xfer_black_user_study(npz_filename, npz_dir_path, img_dir_path, attack_alg, test_num, api_name)

    # plot_xfer_JPG_PNG(npz_filename, npz_dir_path, img_dir_path, attack_alg, test_num, api_name)
    # plot_xfer_black('pgd', 'cifar')
    # plot_xfer_black('pgd', 'mnist')

    # plot_amplification_black('cw', 'cifar')
    # plot_amplification_black('cw', 'mnist')
    # plot_amplification_black('pgd', 'cifar')
    # plot_amplification_black('pgd', 'mnist')
