import Config
from utils.fr_utils import *
from models.inception_big import *
from keras.models import load_model, model_from_json


def get_model(params):
    """
    """
    if params['whitebox_target']:
        model_type = params['target_model']
        loss_type = params['target_loss']
        dataset_type = params['target_dataset']
    else:
        model_type = params['model_type']
        loss_type = params['loss_type']
        dataset_type = params['dataset_type']

    if dataset_type == 'vgg':
        if model_type == 'small':
            if loss_type == 'center':
                fr_model = CenterModel()
            elif loss_type == 'triplet':
                fr_model = TripletModel()
        elif model_type == 'large':
            if loss_type == 'center':
                fr_model = FacenetLarge(Config.CENTER_MODEL_PATH, classes=512)
            elif loss_type == 'triplet':
                fr_model = FacenetLarge(Config.TRIPLET_MODEL_PATH, classes=128)
    elif dataset_type == 'casia':
        fr_model = FacenetLarge(Config.CASIA_MODEL_PATH, classes=512)
    elif dataset_type == 'vggsmall':
        fr_model = FacenetLarge(Config.VGGSMALL_MODEL_PATH, classes=512)
    elif dataset_type == 'vggadv':
        fr_model = FacenetLarge(Config.VGGADV_MODEL_PATH, classes=512)
    return fr_model


class CenterModel:
    """
    """
    def __init__(self, session=None):
        self.num_channels = 3
        self.image_height = 112
        self.image_width = 96
        self.model = load_model(os.path.join(Config.ROOT, 'weights/face_model_caffe_converted.h5'))

    def predict(self, im):
        return self.model(im)


class TripletModel:
    """
    """
    def __init__(self, session=None):
        self.num_channels = 3
        self.image_height = 96
        self.image_width = 96
        self.image_size = 96

        json_file = open(os.path.join(Config.ROOT, 'models/FRmodel.json'), 'r')
        loaded_model_json = json_file.read()
        json_file.close()
        FRmodel = model_from_json(loaded_model_json)
        FRmodel.load_weights(os.path.join(Config.ROOT, "weights/FRmodel.h5"))
        self.model = FRmodel
    
    def predict(self, im):
        return self.model(im)


class FacenetLarge:
    """
    """
    def __init__(self, path, classes=512, session=None):
        self.num_channels = 3
        self.image_height = 160
        self.image_width = 160
        self.image_size = 160

        FRmodel = faceRecoModel((self.image_height, self.image_width, self.num_channels), classes=classes)
        FRmodel.load_weights(os.path.join(Config.ROOT, path))
        self.model = FRmodel
    
    def predict(self, im):
        return self.model(im)
