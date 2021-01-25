from os.path import abspath, join, exists
from os import mkdir


def create(path):
    """
    Description

    Keyword arguments:
    """
    if not exists(path):
        mkdir(path)
        print('Creating directory: {}'.format(path))


def check_depth(depth):
    """
    Description

    Keyword arguments:
    """
    if depth is -1:
        directory_list = API_NAMES
    if depth is 0:
        directory_list = ATTACKS
    if depth is 1:
        directory_list = MODEL_SIZE
    if depth is 2:
        directory_list = ATTACK_LOSS
    if depth is 3:
        directory_list = CROP
    return directory_list


def doit(depth, limit, cur, prev_path):
    """
    Description

    Keyword arguments:
    """
    new_path = join(prev_path, cur)
    create(new_path)
    if depth < limit:
        recurse_directories(depth + 1, limit, new_path)


def recurse_directories(depth, limit, prev_path):
    """
    Description

    Keyword arguments:
    """
    directory_list = check_depth(depth)
    for i in directory_list:
        doit(depth, limit, i, prev_path)


ROOT = abspath('./data')
ADV_IMGS = join(ROOT, 'new_adv_imgs')
API_RESULTS = join(ROOT, 'new_api_results')

API_NAMES =   ['azure', 'awsverify', 'facepp']
ATTACKS =     ['cw_l2', 'cw_li', 'pgd_l2', 'pgd_li', 'cw_l2_tv', 'cw_li_tv', 'pgd_l2_tv', 'pgd_li_tv',
               'cw_l2_cos', 'cw_li_cos', 'pgd_l2_cos', 'pgd_li_cos', 'cw_l2_tv_cos', 'cw_li_tv_cos', 
               'pgd_l2_tv_cos', 'pgd_li_tv_cos']
MODEL_SIZE =  ['large_center', 'large_triplet', 'small_center', 'small_triplet', 'casia', 'vggsmall']
ATTACK_LOSS = ['hinge_loss', 'target_loss']
CROP =        ['crop', 'full', 'npz', 'crop_mean', 'full_mean', 'npz_mean']

create(ADV_IMGS)
recurse_directories(0, 3, ADV_IMGS)

create(API_RESULTS)
recurse_directories(-1, 3, API_RESULTS)

print('SUCCESS!')