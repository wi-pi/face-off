import boto3
import os
import Config
import argparse
from FaceAPI import credentials


s3 = boto3.resource('s3', aws_access_key_id=credentials.aws_access_key_id, aws_secret_access_key= credentials.aws_secret_access_key)


def upload(source_folder_name, target_folder_name):
    count = 0

    files = [f for f in os.listdir(source_folder_name) if os.path.isfile(os.path.join(source_folder_name, f))]
    print(len(files))
    for file_name in files:
        if ".png" in file_name:
            print(file_name)
            s3.meta.client.upload_file(os.path.join(source_folder_name, file_name), Config.S3_BUCKET, target_folder_name + "/" + file_name)
            count = count + 1

    print("Total upload: " + str(count))


def upload_api(source_folder_name, target_folder_name, params):
    def populate_file_list(dir1, dir2, base_flag):
        files = []
        path = os.path.join(Config.ROOT, dir1, dir2)
        for file in os.listdir(path):
            if base_flag:
                files.append(file.replace('.jpg', ''))
            else:
                files.append(os.path.join(path, file))
        return files
    count = 0
    for source in Config.API_PEOPLE:
        base_files = populate_file_list(Config.TEST_DIR, source, base_flag=True)
        for target in Config.API_PEOPLE:
            if (source is not target and
                (target == Config.PAIRS[source] or not params['pair_flag'])):
                for file in base_files:
                    for margin in params['margin_list']:
                        for amplification in params['amp_list']:
                            marg_str = '%0.2f' % margin
                            amp_str = '%0.3f' % amplification
                            png_format = '{}_{}_{}_loss_{}_{}_marg_{}_amp_{}.png'
                            file_name = png_format.format(params['attack_name'],
                                                          params['model_name'],
                                                          params['attack_loss'][0],
                                                          file,
                                                          target,
                                                          marg_str,
                                                          amp_str)
                            # print(file_name)
                            try:
                                s3.meta.client.upload_file(os.path.join(source_folder_name, file_name), Config.S3_BUCKET, target_folder_name + "/" + file_name)
                            except Exception as e:
                                print(e)
                            count = count + 1
            print(source, target, count)

    print("Total upload: " + str(count))


def upload_override(source_folder_name):
    count = 0
    path = os.path.join(Config.ROOT, source_folder_name)
    for person in os.listdir(path):
        for file in os.listdir(os.path.join(source_folder_name, person)):
            if file.endswith('.png') or file.endswith('jpg'):
                print(file)
                s3.meta.client.upload_file(os.path.join(path, person, file), Config.S3_BUCKET, source_folder_name + "/" + person + '/' + file)
                count = count + 1

    print("Total upload: " + str(count))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model-type', type=str, default="small", help='type of model', choices=['small','large'])
    parser.add_argument('--loss-type', type=str, default="center", help='loss function used to train the model', choices=['center','triplet'])
    parser.add_argument('--dataset-type', type=str, default='vgg', help='type of dataset in training')
    parser.add_argument('--attack', type=str, default='CW', help='attack type',choices=['PGD', 'CW'])
    parser.add_argument('--norm', type=str, default='2', help='p-norm', choices=['inf', '2'])
    parser.add_argument('--targeted-flag', type=str, default='true', help='targeted (true) or untargeted (false)', choices=['true', 'false'])
    parser.add_argument('--tv-flag', type=str, default='false', help='do not use tv_loss term (false) or use it (true)', choices=['true', 'false'])
    parser.add_argument('--hinge-flag', type=str, default='true', help='hinge loss (true) or target loss (false)', choices=['true', 'false'])
    parser.add_argument('--cos-flag', type=str, default='false', help='use cosine similarity instead of l2 for loss', choices=['true', 'false'])
    parser.add_argument('--margin', type=float, default=6.0, help='needed for determining goodness of transferability')
    parser.add_argument('--amplification', type=float, default=5.1, help='needed for amplifying adversarial examples')
    parser.add_argument('--granularity', type=str, default='normal', help='add more or less margin and amplification intervals', choices=['fine-tuned', 'fine', 'normal', 'coarse', 'coarser', 'coarse-single', 'single', 'api-eval'])
    parser.add_argument('--mean-loss', type=str, default='embeddingmean', help='old:(embedding) new formulation:(embeddingmean) WIP formulation:(distancemean)', choices=['embeddingmean', 'embedding', 'distancemean'])
    parser.add_argument('--override-dir', type=str, default='none', help='', choices=['none', 'test_imgs', 'celeb160', 'celeb96', 'celeb'])
    parser.add_argument('--pair-flag', type=str, default='false', help='', choices=['true', 'false'])
    args = parser.parse_args()

    params = Config.set_parameters(model_type=args.model_type,
                                   loss_type=args.loss_type,
                                   dataset_type=args.dataset_type,
                                   attack=args.attack,
                                   norm=args.norm,
                                   targeted_flag=args.targeted_flag,
                                   tv_flag=args.tv_flag,
                                   hinge_flag=args.hinge_flag,
                                   cos_flag=args.cos_flag,
                                   granularity=args.granularity,
                                   mean_loss=args.mean_loss,
                                   amplification=args.amplification,
                                   margin=args.margin,
                                   pair_flag=args.pair_flag)
    if args.override_dir is 'none':
        source_folder_name = params['directory_path']
        source_folder_crop = params['directory_path_crop']
        target_folder_name = source_folder_name.replace(Config.ROOT + '/', '')
        target_folder_crop = source_folder_crop.replace(Config.ROOT + '/', '')
        # upload(source_folder_name, target_folder_name)
        # upload(source_folder_crop, target_folder_crop)
        upload_api(source_folder_name, target_folder_name, params)
        upload_api(source_folder_name, target_folder_crop, params)
    else:
        source_folder_name = args.override_dir
        upload_override(source_folder_name)
