import numpy as np
import matplotlib.pyplot as plt


def plot_lipschitz_constant(targ_model, attack_in):
    # targ_model = 'center'
    # attack_in = 'cw'
    plt.close('all')
    ax = plt.subplot(111)

    adv_output = np.loadtxt('../figures/' + targ_model + '+' + attack_in + '.txt')
    amp_list = np.loadtxt('../figures/amp_list_' + attack_in + '.txt')
    margin_list = np.loadtxt('../figures/margin_list.txt')
    no_adv = np.loadtxt('../figures/no_adv_' + targ_model + '.txt')

    for margin_index, margin in enumerate(margin_list):
        if int(margin) in [2, 4] or margin>5.1:
            continue
        print(adv_output[margin_index].shape)
        print(amp_list.shape)
        lbl_str = ''
        if attack_in == 'cw':
            lbl_str = r'$\kappa$=' + "%0.1f" % margin
        elif attack_in == 'pgd':
            lbl_str = r'$\epsilon$=' + "%0.1f" % margin
        plt.plot(amp_list, adv_output[margin_index], label=lbl_str, linewidth=3.0)

    #plt.plot(amp_list, adv_output[margin_index], label=lbl_str)
    plt.fill_between(amp_list, np.min(no_adv)*np.ones((len(amp_list))), np.max(no_adv)*np.ones((len(amp_list))),
                     facecolor='gray', alpha=0.35, label='Not Adv.')
    plt.xlabel('Amplification factor', fontsize=20)
    plt.ylabel('Libschitz Constant', fontsize=20)
    plt.legend(loc='lower left', prop={'size': 16})
    ax.tick_params(axis='both', which='major', labelsize=15)
    #plt.grid()

    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)
    plt.savefig('../figures/' + targ_model + '+' + attack_in + '.png', bbox_inches='tight')


if __name__ == "__main__":
    plot_lipschitz_constant('center', 'cw')
    plot_lipschitz_constant('triplet', 'pgd')
    plot_lipschitz_constant('center', 'pgd')
    plot_lipschitz_constant('triplet', 'cw')
