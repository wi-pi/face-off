import math
import numpy as np
import os

import matplotlib
matplotlib.use('Agg')

import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D


FONT_SIZE = 12
LEGEND_SIZE = 10
LABEL_SIZE = 10

ROOT = os.getcwd()

def plot_xfer_black(attack_alg, dataset, plot_type):

    plt.close('all')
    ax = plt.subplot(111)

    npz_file = np.load(ROOT+ '/npz/result_4_from_' + attack_alg + '_' + dataset + '.npz')  # or result_4_from_cw.npz

    param_list = npz_file['arr_0'] #2d array of shape epsi_list x iteration_list
    amp_list = npz_file['arr_1'] #1d array of shape 98
    amp_sqr_list = [math.sqrt(i) for i in amp_list]
    mean_l2_amp = npz_file['arr_2'] #2d array of shape (epsi_list * iteration_list) x amp_list 
    mean_succ_nontarg_amp = npz_file['arr_3'] #2d array of shape (epsi_list * iteration_list) x amp_list

    epsi_list = sorted(list(set(param_list[:, 0])))
    iteration_list = sorted(list(set(param_list[:, 1])))
    min_l2 = np.zeros((len(epsi_list), len(iteration_list)))
    min_amp = np.zeros((len(epsi_list), len(iteration_list)))
  
    if plot_type == '3':
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
    
    for i, epsi in enumerate(epsi_list):

        if attack_alg == 'cw' and dataset == 'cifar':
            if int(epsi) < 30:
                continue
        elif attack_alg == 'cw' and dataset == 'mnist':
            plt.ylim([2, 7])
            if epsi > 51:
                continue
        elif attack_alg == 'pgd':
            # plt.xlim([2, 80])
            if (int(epsi*10)-5) % 10 == 0 or int(epsi) == 1 or int(epsi) == 8 or int(epsi) == 6:  # skip all the 0.5s
                continue

        for j, iteration in enumerate(iteration_list):
            param_ind = (i) * len(iteration_list) + j
            l2_vector = mean_l2_amp[param_ind]
            just_delta = l2_vector/amp_sqr_list
            succ_idxs = [index for index, succ in enumerate(mean_succ_nontarg_amp[param_ind]) if succ >= 0.85]
            if len(succ_idxs) == 0:
                min_l2[i][j] = None
                continue
            succ_idx = succ_idxs[0]

            
            min_l2[i][j] = l2_vector[succ_idx]
            #min_l2[i][j] = just_delta[succ_idx]
            min_amp[i][j] = amp_list[succ_idx]

        if attack_alg is 'pgd':
            lab_str = r'$\epsilon$=' + str(epsi)
        elif attack_alg is 'cw':
            lab_str = r'$\kappa$=' + str(epsi)

        if plot_type == '3':
            ax.scatter(iteration_list, min_l2[i], min_amp[i], label=lab_str, marker='o')
            ax.set_xlabel('n', fontsize=FONT_SIZE)
            ax.set_ylabel(r'$||\alpha.\delta||_2$', fontsize=FONT_SIZE)
            ax.set_zlabel(r'$\alpha$', fontsize=FONT_SIZE)
        else:
            plt.plot(iteration_list, min_l2[i], label=lab_str, linewidth=2.5, marker='o')
            plt.xlabel('n', fontsize=FONT_SIZE)
            plt.ylabel(r'$||\alpha.\delta||_2$', fontsize=FONT_SIZE)

    plt.legend(loc='upper right', prop={'size': LEGEND_SIZE})
    ax.tick_params(axis='both', which='major', labelsize=LABEL_SIZE)

    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)
    plt.savefig(ROOT + '/png/'+dataset + '_' + attack_alg + '_' + plot_type + 'd.png', bbox_inches='tight')


def plot_amplification_black(attack_alg, dataset):

    plt.close('all')
    ax = plt.subplot(111)

    npz_file = np.load(ROOT+'/npz/result_4_from_' + attack_alg + '_' + dataset + '.npz')  # or result_4_from_cw.npz

    param_list = npz_file['arr_0']
    amp_list = npz_file['arr_1']
    mean_l2_amp = npz_file['arr_2']
    mean_succ_nontarg_amp = npz_file['arr_3']

    epsi_list = sorted(list(set(param_list[:, 0])))
    iteration_list = sorted(list(set(param_list[:, 1])))
    min_l2 = np.zeros((len(epsi_list), len(iteration_list)))

    #print(amp_list)
    for i, epsi in enumerate(epsi_list):

        if attack_alg == 'cw' and dataset == 'cifar':
            plt.xlim([0, 50])
            if int(epsi) < 30:
                continue
        elif attack_alg == 'cw' and dataset == 'mnist':
            plt.xlim([0, 50])
            if epsi > 51:
                continue
        elif attack_alg == 'pgd':
            plt.xlim([0, 50])
            if (int(epsi*10)-5) % 10 == 0 or int(epsi) == 1 or int(epsi) == 8 or int(epsi) == 6:  # skip all the 0.5s
                continue

        for j, iteration in enumerate(iteration_list):
            if attack_alg == 'pgd' and iteration != 80:
                continue
            if attack_alg == 'cw' and iteration != 5000:
                continue
            param_ind = (i) * len(iteration_list) + j
            l2_vector = mean_l2_amp[param_ind]
            #print(param_list[param_ind], epsi, iteration, attack_alg, dataset)

        if attack_alg is 'pgd':
            lab_str = r'$\epsilon$=' + str(epsi)
        elif attack_alg is 'cw':
            lab_str = r'$\kappa$=' + str(epsi)

        print(attack_alg, dataset, len(amp_list), len(mean_succ_nontarg_amp[param_ind]))
        #plt.plot(l2_vector, mean_succ_nontarg_amp[param_ind] * 100, label=lab_str, linewidth=2.5)
        plt.plot(amp_list, mean_succ_nontarg_amp[param_ind] * 100, label=lab_str, linewidth=2.5)
        #plt.xlabel(r'$||\alpha.\delta||_2$', fontsize=FONT_SIZE)
        plt.xlabel(r'$\alpha$', fontsize=FONT_SIZE)
        plt.ylabel('Transferability (%)', fontsize=FONT_SIZE)
        # plt.grid()
    print("\n")
    plt.ylim([0,100])
    plt.legend(loc='lower right', prop={'size': LEGEND_SIZE})
    ax.tick_params(axis='both', which='major', labelsize=LABEL_SIZE)
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)
    plt.savefig(ROOT+'/png/'+dataset + '_' + attack_alg + '_ampl.png', bbox_inches='tight')


if __name__ == "__main__":
    plot_xfer_black('cw', 'cifar', '3')
    plot_xfer_black('cw', 'mnist', '3')
    plot_xfer_black('pgd', 'cifar', '3')
    plot_xfer_black('pgd', 'mnist', '3')

    plot_xfer_black('cw', 'cifar', '2')
    plot_xfer_black('cw', 'mnist', '2')
    plot_xfer_black('pgd', 'cifar', '2')
    plot_xfer_black('pgd', 'mnist', '2')
    
    plot_amplification_black('cw', 'cifar')
    plot_amplification_black('cw', 'mnist')
    plot_amplification_black('pgd', 'cifar')
    plot_amplification_black('pgd', 'mnist')
