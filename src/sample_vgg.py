import os, random, shutil


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
