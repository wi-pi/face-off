import os, random, shutil
import Config

def copy():
    ROOT = Config.ROOT
    ROOT_FACENET = os.path.abspath('../../facenet')
    path = os.path.join(ROOT, 'small-vgg-align-train-160')
    for identity in os.listdir(path):
        vgg_test = os.path.join(ROOT, 'test_imgs', 'VGG', identity)
        facenet_vgg = os.path.join(ROOT_FACENET, 'datasets', 'vggface2', 'vggface2_train_182', identity)
        if not os.path.exists(vgg_test):
            os.mkdir(vgg_test)
        images = os.listdir(os.path.join(path, identity))
        total = len(images)
        random.shuffle(images)
        adversarial = images[:int(total / 2)]
        print(identity)
        for img in adversarial:
            try:
                shutil.copy2(os.path.join(facenet_vgg, img), os.path.join(vgg_test, img))
            except FileNotFoundError:
                shutil.copy2(os.path.join(facenet_vgg, img.replace('.png', '.jpg')), os.path.join(vgg_test, img.replace('.png', '.jpg')))


def sample():
    ROOT = os.path.abspath('.')
    path = os.path.join(ROOT, 'small-vgg-align-train-160')
    for identity in os.listdir(path):
        vgg_test = os.path.join(ROOT, 'test_imgs', 'VGG', identity)
        if not os.path.exists(vgg_test):
            os.mkdir(vgg_test)
        images = os.listdir(os.path.join(path, identity))
        total = len(images)
        random.shuffle(images)
        adversarial = images[:int(total / 2)]
        print(identity)
        for img in adversarial:
            shutil.copy2(os.path.join(path, identity, img), os.path.join(vgg_test, img))



def move():
    sources = Config.SOURCES
    targets = Config.TARGETS
    ROOT = Config.ROOT
    path = os.path.join(ROOT, 'small-vgg-adv')
    for file in os.listdir(path):
        if file.endswith('.png'):
            end = file.index('_marg_')
            temp = file[:end]
            label = temp[-7:]
            index = targets.index(label)
            shutil.move(os.path.join(path, file), os.path.join(path, sources[index], file))


def remove():
    path = os.path.join(Config.ROOT, 'small-vgg-align-out')
    for s in Config.SOURCES:
        for file in os.listdir(os.path.join(path, s)):
            if file.endswith('_marg_5.00_amp_2.000.png'):
                shutil.move(os.path.join(path, s, file), os.path.join(path, 'dump', s))


def move_duplicates():
    path = os.path.join(Config.ROOT, 'small-vgg-adv')
    for s in Config.SOURCES:
        for file in os.listdir(os.path.join(path, s)):
            if file.startswith('cw_l2'):
                begin = file.index('_loss_')
                end = file.index('_marg_')
                temp = file[begin + 6: end - 8] + '.png'
                shutil.move(os.path.join(path, s, temp), os.path.join(path, 'dump', temp))



def main():
    move()
    move_duplicates()


if __name__ == '__main__':
    main()
