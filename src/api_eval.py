import numpy as np
import os
import time
import Config
import argparse
from FaceAPI.microsoft_face import verify_API
from FaceAPI.aws_rekognition import aws_compare
from FaceAPI.facepp import facepp
from FaceAPI import credentials

BASE_DIR = credentials.s3_bucket_url


class API_Tester:
    """Class for storing parameters, API results, and querying the APIs."""

    def __init__(self,
                 params,
                 source,
                 target,
                 topn_flag,
                 credentials):
        self.model_type = params['model_type']
        self.model_name = params['model_name']
        self.attack_name = params['attack_name']
        self.attack_loss = params['attack_loss']
        self.api_name = params['api_name']
        self.margin_list = params['margin_list']
        self.amp_list = params['amp_list']
        self.align_dir = params['align_dir']
        self.test_dir = params['test_dir']
        self.full_dir = params['full_dir']
        self.source = source
        self.target = target
        self.len_marg = len(self.margin_list)
        self.len_amp = len(self.amp_list)
        self.topn_flag = topn_flag
        self.total = 0
        self.success = 0
        self.credentials = credentials
        self.mean_loss = params['mean_loss']

    def setup_file_lists(self):
        def populate_file_list(dir1, dir2, base_flag):
            files = []
            path = os.path.join(Config.ROOT, dir1, dir2)
            for file in os.listdir(path):
                if base_flag:
                    files.append(file.replace('.jpg', ''))
                    return files
                else:
                    files.append(os.path.join(path, file))
            return files

        self.source_files_full = populate_file_list(self.full_dir, self.source, base_flag=False)
        self.target_files_full = populate_file_list(self.full_dir, self.target, base_flag=False)
        self.source_files_crop = populate_file_list(self.align_dir, self.source, base_flag=False)
        self.target_files_crop = populate_file_list(self.align_dir, self.target, base_flag=False)
        self.base_files = populate_file_list(self.test_dir, self.source, base_flag=True)
        self.all_files_full = []
        self.all_files_crop = []
        for i in Config.API_PEOPLE:
            self.all_files_full.extend(populate_file_list(self.full_dir, i, base_flag=False))
            self.all_files_crop.extend(populate_file_list(self.align_dir, i, base_flag=False))

        self.len_src_full = len(self.source_files_full)
        self.len_targ_full = len(self.target_files_full)
        self.len_all_full = len(self.all_files_full)
        self.len_src_crop = len(self.source_files_crop)
        self.len_targ_crop = len(self.target_files_crop)
        self.len_all_crop = len(self.all_files_crop)

        self.f1 = open('./fileio/log_{}_{}_{}_{}_{}.txt'.format(self.api_name,
            self.model_name, self.attack_name, self.source, self.target), 'w')

    def intialize_scores(self):
        """Initialized numpy arrays to store API results."""

        source_shape = (self.len_marg, self.len_amp, self.len_src_full)
        target_shape = (self.len_marg, self.len_amp, self.len_targ_full)
        all_shape = (self.len_marg, self.len_amp, self.len_all_full)
        source_shape_crop = (self.len_marg, self.len_amp, self.len_src_crop)
        target_shape_crop = (self.len_marg, self.len_amp, self.len_targ_crop)
        all_shape_crop = (self.len_marg, self.len_amp, self.len_all_crop)
        th_source_shape = (self.len_marg, self.len_amp, self.len_src_full, 3)
        th_target_shape = (self.len_marg, self.len_amp, self.len_targ_full, 3)
        th_all_shape = (self.len_marg, self.len_amp, self.len_all_full, 3)
        th_source_shape_crop = (self.len_marg, self.len_amp, self.len_src_crop, 3)
        th_target_shape_crop = (self.len_marg, self.len_amp, self.len_targ_crop, 3)
        th_all_shape_crop = (self.len_marg, self.len_amp, self.len_all_crop, 3)

        l2_shape = (self.len_marg, self.len_amp)

        self.score_self = np.zeros(source_shape)
        self.score_target = np.zeros(target_shape)
        self.score_all = np.zeros(all_shape)
        self.score_self_crop = np.zeros(source_shape_crop)
        self.score_target_crop = np.zeros(target_shape_crop)
        self.score_all_crop = np.zeros(all_shape_crop)
        self.th_self = np.zeros(th_source_shape)
        self.th_target = np.zeros(th_target_shape)
        self.th_all = np.zeros(th_all_shape)
        self.th_self_crop = np.zeros(th_source_shape_crop)
        self.th_target_crop = np.zeros(th_target_shape_crop)
        self.th_all_crop = np.zeros(th_all_shape_crop)
        self.l2_mtx = np.zeros(l2_shape)

    def get_scores(self,
                   ii,
                   jj,
                   files,
                   adv_img,
                   scores,
                   ths):
        """
        Query APIs with the URLs to images and retrieves scores.

        Keyword arguments:
        ii -- margin index
        jj -- amplification index
        files -- source/target files to compare to (non-adversarial)
        adv_img -- adversarial image file
        scores -- numpy array which stores results
        ths -- numpy array which stores face++ dynamic threshold
        """

        Config.BM.mark('api calls')
        self.total = 0
        self.success = 0
        count = 0
        for kk, img in enumerate(files):
            print(img, kk)
            img_success = False
            start_time = time.time()
            s3_img = img.replace(Config.ROOT, Config.S3_DIR)
            if self.source in img:
                target_print = 'Untargeted'
                target_flag = False
            else:
                target_print = 'Targeted'
                target_flag = True
            if self.api_name == 'awsverify':
                try:
                    score = aws_compare(adv_img, s3_img, 0, self.credentials)
                    count += 1
                except Exception as e:
                    self.f1.write(str(e))
                    score = None
                    self.f1.write('Error {} {}\n'.format(adv_img, s3_img))
                if score is not None:
                    if target_flag and score and score >= 50:
                        img_success = True
                    elif not target_flag and score and score <= 50:
                        img_success = True
            elif self.api_name == 'azure':
                score, isSame, has_face = verify_API(adv_img, s3_img, 1, self.credentials)
                count += 1
                if score is not None:
                    if not has_face:
                        self.f1.write('No face detected\n')
                        self.f1.write('{}\n'.format(adv_img))
                    if target_flag and isSame:
                        img_success = True
                    elif not target_flag and not isSame:
                        img_success = True
            elif self.api_name == 'facepp':
                score, th = facepp(adv_img, s3_img, self.credentials)
                count += 1
                ths[ii, jj, kk, :] = th
                if score is not None:
                    if target_flag and score >= th[1]:
                        img_success = True
                    elif not target_flag and score <= th[1]:
                        img_success = True
            if img_success:
                # print('{} attack succeed'.format(target_print))
                # print(adv_img)
                # print(s3_img)
                self.f1.write('{} attack succeed {} {} {}\n'.format(target_print,score,adv_img,s3_img))
                self.success += 1
            else:
                self.f1.write('{} attack failed {} {} {}\n'.format(target_print, score, adv_img, s3_img))
            self.total += 1

            if self.api_name == 'azure' and has_face:
                scores[ii, jj, kk] = score
            elif self.api_name != 'azure':
                scores[ii, jj, kk] = score
            elif not has_face:
                scores[ii, jj, kk] = None
            elapsed_time = time.time() - start_time

            # if elapsed_time < 1.1:
                # time.sleep(1.1 - elapsed_time)
        print(count)
        Config.BM.mark('api calls')
        return scores, ths

    def test(self):
        """Main execution block. Sets up files and paths, gets scores, writes scores to .npz files."""

        self.setup_file_lists()
        for source_file in self.base_files:
            if self.mean_loss == 'embedding':
                full = 'full_mean'
                crop = 'crop_mean'
                npz = 'npz_mean'
            else:
                full = 'full'
                crop = 'crop'
                npz = 'npz'

            api_folder = 'new_api_results/{}/{}/{}/{}_loss/{}'.format(self.api_name,
                                                                      self.attack_name,
                                                                      self.model_name,
                                                                      self.attack_loss,
                                                                      npz)
            npz_format = '{}_{}_{}_loss_{}_{}.npz'
            out_file_temp = npz_format.format(self.attack_name,
                                              self.model_name,
                                              self.attack_loss[0],
                                              source_file,
                                              self.target)
            self.out_file_name = os.path.join(api_folder, out_file_temp)
            self.intialize_scores()
            for ii, margin in enumerate(self.margin_list):
                marg_str = '%0.2f' % margin
                npz_path = os.path.join(Config.ROOT, 'new_adv_imgs/{}/{}/{}_loss/{}'.format(self.attack_name,
                                                                                            self.model_name,
                                                                                            self.attack_loss,
                                                                                            npz))
                inname = '{}_{}_{}_loss_{}_{}_marg_{}.npz'.format(self.attack_name,
                                                                  self.model_name,
                                                                  self.attack_loss[0],
                                                                  source_file,
                                                                  self.target,
                                                                  marg_str)
                inname = os.path.join(npz_path, inname)
                npzfile = np.load(inname, allow_pickle=True)
                delta_clip_stack = npzfile['delta_clip_stack']
                for jj, amp in enumerate(self.amp_list):
                    amp_str = '%0.3f' % amp
                    print('Testing with margin {}, amplification {}'.format(marg_str, amp_str))
                    adv_img_name = '{}_{}_{}_loss_{}_{}_marg_{}_amp_{}.png'.format(self.attack_name,
                                                                                   self.model_name,
                                                                                   self.attack_loss[0],
                                                                                   source_file,
                                                                                   self.target,
                                                                                   marg_str,
                                                                                   amp_str)
                    full_adv_url = '{}/new_adv_imgs/{}/{}/{}_loss/{}/{}'.format(BASE_DIR,
                                                                                self.attack_name,
                                                                                self.model_name,
                                                                                self.attack_loss,
                                                                                full,
                                                                                adv_img_name)
                    crop_adv_url = '{}/new_adv_imgs/{}/{}/{}_loss/{}/{}'.format(BASE_DIR,
                                                                                self.attack_name,
                                                                                self.model_name,
                                                                                self.attack_loss,
                                                                                crop,
                                                                                adv_img_name)
                    if jj >= len(delta_clip_stack):
                        mismatch = True
                    else:
                        mismatch = False
                    if mismatch or delta_clip_stack[jj] is not None:
                        if not mismatch:
                            self.l2_mtx[ii, jj] = np.linalg.norm(delta_clip_stack[jj])

                        # full untargeted
                        if self.topn_flag:
                            self.score_all, self.th_all = self.get_scores(ii=ii,
                                                                          jj=jj,
                                                                          files=self.all_files_full,
                                                                          adv_img=full_adv_url,
                                                                          scores=self.score_all,
                                                                          ths=self.th_all)
                        else:
                            self.score_self, self.th_source = self.get_scores(ii=ii,
                                                                                jj=jj,
                                                                                files=self.source_files_full,
                                                                                adv_img=full_adv_url,
                                                                                scores=self.score_self,
                                                                                ths=self.th_self)
                            print('full untargeted: {}'.format(self.success / self.total))
                            # full targeted
                            self.score_target, self.th_target = self.get_scores(ii=ii,
                                                                                jj=jj,
                                                                                files=self.target_files_full,
                                                                                adv_img=full_adv_url,
                                                                                scores=self.score_target,
                                                                                ths=self.th_target)
                            print('full targeted: {}'.format(self.success / self.total))
                        if self.topn_flag:
                            self.score_all_crop, self.th_all_crop = self.get_scores(ii=ii,
                                                                                    jj=jj,
                                                                                    files=self.all_files_crop,
                                                                                    adv_img=crop_adv_url,
                                                                                    scores=self.score_all_crop,
                                                                                    ths=self.th_all_crop)
                        else:
                        # crop untargeted
                            self.score_self_crop, self.th_self_crop = self.get_scores(ii=ii,
                                                                                      jj=jj,
                                                                                      files=self.source_files_full,
                                                                                      adv_img=crop_adv_url,
                                                                                      scores=self.score_self_crop,
                                                                                      ths=self.th_self_crop)
                            print('crop untargeted: {}'.format(self.success / self.total))
                            # crop targeted
                            self.score_target_crop, self.th_target_crop = self.get_scores(ii=ii,
                                                                                          jj=jj,
                                                                                          files=self.target_files_full,
                                                                                          adv_img=crop_adv_url,
                                                                                          scores=self.score_target_crop,
                                                                                          ths=self.th_target_crop)
                            print('crop targeted: {}'.format(self.success / self.total))

            if self.api_name == 'awsverify' or self.api_name == 'azure':
                np.savez(self.out_file_name,
                         score_target=self.score_target, score_self=self.score_self,
                         score_target_crop=self.score_target_crop, score_self_crop=self.score_self_crop,
                         l2_mtx=self.l2_mtx,
                         margin_list=self.margin_list, amp_list=self.amp_list,
                         score_all=self.score_all, score_all_crop=self.score_all_crop)
            elif self.api_name == 'facepp':
                np.savez(self.out_file_name,
                         score_target=self.score_target, score_self=self.score_self,
                         score_target_crop=self.score_target_crop, score_self_crop=self.score_self_crop,
                         th_target=self.th_target, th_self=self.th_self,
                         th_target_crop=self.th_target_crop, th_self_crop=self.th_self_crop,
                         l2_mtx=self.l2_mtx,
                         margin_list=self.margin_list, amp_list=self.amp_list,
                         score_all=self.score_all, score_all_crop=self.score_all_crop,
                         th_all=self.th_all, th_all_crop=self.th_all_crop)
        self.f1.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--api-name', type=str, default='azure', help='API to evaluate against', choices=['azure', 'awsverify', 'facepp'])
    parser.add_argument('--model-type', type=str, default="small", help='type of model', choices=['small','large'])
    parser.add_argument('--loss-type', type=str, default="center", help='loss function used to train the model',choices=['center','triplet'])
    parser.add_argument('--dataset-type', type=str, default='vgg', help='dataset used to train the model',choices=['vgg','casia','vggsmall'])
    parser.add_argument('--attack', type=str, default='CW', help='attack type',choices=['PGD', 'CW'])
    parser.add_argument('--norm', type=str, default='2', help='p-norm', choices=['inf', '2'])
    parser.add_argument('--targeted-flag', type=str, default='true', help='targeted (true) or untargeted (false)', choices=['true', 'false'])
    parser.add_argument('--tv-flag', type=str, default='false', help='do not use tv_loss term (false) or use it (true)', choices=['true', 'false'])
    parser.add_argument('--hinge-flag', type=str, default='true', help='hinge loss (true) or target loss (false)', choices=['true', 'false'])
    parser.add_argument('--cos-flag', type=str, default='false', help='use cosine similarity instead of l2 for loss', choices=['true', 'false'])
    parser.add_argument('--margin', type=float, default=6.0, help='needed for determining goodness of transferability')
    parser.add_argument('--amplification', type=float, default=5.1, help='needed for amplifying adversarial examples')
    parser.add_argument('--interpolation', type=str, default='bilinear', help='interpolation method for upscaling and downscaling', choices=['nearest', 'bilinear', 'bicubic', 'lanczos'])
    parser.add_argument('--granularity', type=str, default='normal', help='add more or less margin and amplification intervals', choices=['fine', 'normal', 'coarse', 'single', 'coarser', 'fine-tuned', 'api-eval'])
    parser.add_argument('--topn-flag', type=str, default='true', help='topn evaluation or not', choices=['true', 'false'])
    parser.add_argument('--pair-flag', type=str, default='false', help='optimal source target pairs')
    parser.add_argument('--credentials', type=str, default='0', help='api keys')
    parser.add_argument('--mean-loss', type=str, default='embeddingmean', help='loss to use')
    args = parser.parse_args()
    params = Config.set_parameters(api_name=args.api_name,
                                   model_type=args.model_type,
                                   loss_type=args.loss_type,
                                   dataset_type=args.dataset_type,
                                   attack=args.attack,
                                   norm=args.norm,
                                   targeted_flag=args.targeted_flag,
                                   tv_flag=args.tv_flag,
                                   hinge_flag=args.hinge_flag,
                                   cos_flag=args.cos_flag,
                                   margin=args.margin,
                                   amplification=args.amplification,
                                   granularity=args.granularity,
                                   interpolation=args.interpolation,
                                   pair_flag=args.pair_flag,
                                   mean_loss=args.mean_loss)
    for source in Config.API_PEOPLE:
        for target in Config.API_PEOPLE:
            if (target != source and 
                (not params['pair_flag'] or target is Config.PAIRS[source])):
                print('{} - {}'.format(source, target))
                tester = API_Tester(params=params,
                                    source=source,
                                    target=target,
                                    topn_flag=args.topn_flag=='true',
                                    credentials=args.credentials)
                tester.test()
