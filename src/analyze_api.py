# The Azure results are located in /home/chuhan/ML/face_eval/api_results/azure/, saved as npz files. In there, /dataset1/matt_03_leo_api_center_hinge.npz means:
# The adv examples are found on dataset1
# The original image was matt_03.jpg
# Target is Leo
# The model is center face model
# Hinge loss is used
# Additional information about the score given by Azure. The threshold is 0.5. If score > 0.5, Azure thinks the two faces belong to the same person. If score < 0.5, Azure thinks otherwise. Score = -1 meaning Azure didn't find the face from the adversarial image.
#
# In the npz file, you will find the following arrays:
## margin_list: the margins that were used for the attack
## amp_list: the amplification factor that was used
# score_self: with shape [len(margin_list), len(amp_list), # of true matt images tested]. score_self[x,y,z] is the score Azure gives, for the adversarial image found with margin=margin_list[x], amp=amp_list[y], compared with the z-th matt photo.
# score_target: with shape [len(margin_list), len(amp_list), # of true matt images tested]. score_target[x,y,z] is the score Azure gives, for the adversarial image found with margin=margin_list[x], amp=amp_list[y], compared with the z-th leo photo.
# score_self_crop: same as score_self. except that I was using the cropped face from the adversarial image. True matt images that were compared against were not cropped.
# score_target_crop: same as score_target. except that I was using the cropped face from the adversarial image. True leo images that were compared against were not cropped.
# l2_mtx: with shape [len(margin_list), len(amp_list)]. l2_mtx[x,y] is the perturbation L-2 amplitude of adv image with margin=margin_list[x], amp=amp_list[y]
# To find the saved png/jpg files of the adversarial images, and the true matt and leo images being compared with, check out /home/chuhan/ML/face_eval/api_eval.py, and go to "test_on_API()" at line 413. You will find the paths there.
#
# Call me anytime if you have questions!

import numpy as np
import sys


def process_result(path):
    #npzfile = np.load('./api_results/awsverify/dataset3/matt_03_leo_aws_verify_center_hing_ds3_1.npz')
    npzfile = np.load(path)
    score_target = npzfile['score_target']
    score_self = npzfile['score_self']
    score_target_crop = npzfile['score_target_crop']
    score_self_crop = npzfile['score_self_crop']
    l2_mtx = npzfile['l2_mtx']
    margin_list = npzfile['margin_list']
    amp_list = npzfile['amp_list']

    print(score_self_crop.shape)
    print(margin_list.shape)
    print(amp_list.shape)
    for ind_mar, margin in enumerate(margin_list):
        for ind_amp, amp in enumerate(amp_list):
            sucessful_attacks = len(score_self[ind_mar][ind_amp][score_self[ind_mar][ind_amp]<0.5]) / len(score_self[ind_mar][ind_amp]) * 100
            sucessful_attacks_crop = len(score_self_crop[ind_mar][ind_amp][score_self_crop[ind_mar][ind_amp]<0.5]) / len(score_self_crop[ind_mar][ind_amp]) * 100
            print('Margin: {:.2f}, Amp: {:.2f}, Succ_Attack: {:.2f}%, Succ_Attack_Crop: {:.2f}%'.format(margin, amp, sucessful_attacks, sucessful_attacks_crop))
            #print("%0.1f" % margin, "%0.1f" % amp,score_self_crop[ind_mar][ind_amp])
            # print("%0.1f" % margin, "%0.1f" % amp,len(score_self[ind_mar][ind_amp][score_self[ind_mar][ind_amp]<0.5]),len(score_self_crop[ind_mar][ind_amp][score_self_crop[ind_mar][ind_amp]<0.5]))
            #print(score_self_crop[ind_mar][ind_amp])


            #if len(score_self[ind_mar][ind_amp][score_self[ind_mar][ind_amp]<0.5]) > 0:
             #       if (amp > 3.2 and amp < 4.1):
              #          print("%0.1f" % margin, "%0.1f" % amp, len(score_self[ind_mar][ind_amp][score_self[ind_mar][ind_amp]<0.5]))


if __name__ == "__main__":
    path = sys.argv[1]
    process_result(path)
