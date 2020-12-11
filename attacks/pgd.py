import numpy as np
from cleverhans.model import Model, CallableModelWrapper
from cleverhans.attacks import Attack
import tensorflow as tf
from abc import ABCMeta
import warnings
import collections

import cleverhans.utils as utils
from cleverhans.model import Model, CallableModelWrapper
from attacks.tv_loss import get_tv_loss
from cleverhans.utils_tf import clip_eta

_logger = utils.create_logger("cleverhans.attacks")


class PGD(Attack):

    """
    The Projected Gradient Descent Attack (Madry et al. 2017).
    Paper link: https://arxiv.org/pdf/1706.06083.pdf
    """

    #from reference implementation
    def __init__(self, 
                model, 
                back='tf', 
                sess=None, 
                dtypestr='float32',
                default_rand_init=True,
                **kwargs):
        """
        Create a PGD instance.
        """
        if not isinstance(model, Model):
            model = CallableModelWrapper(model, 'logits')
        
        #print(model, back,sess, dtypestr)
        super(PGD, self).__init__(model, 
                                back = back,
                                sess = sess, 
                                dtypestr = dtypestr,
                                **kwargs) 
        
        self.feedable_kwargs = {'eps': self.np_dtype,
                                'eps_iter': self.np_dtype,
                                'y': self.np_dtype,
                                'y_target': self.np_dtype,
                                'clip_min': self.np_dtype,
                                'clip_max': self.np_dtype}
        
        self.structural_kwargs = ['norm', 'nb_iter', 'rand_init']
    
        self.default_rand_init = default_rand_init


    #custom function
    #provide default values for parameters in function definition
    def set_parameters(self, 
                       params,
                       target_imgs, 
                       src_imgs, 
                       margin, 
                       model,
                       base_imgs,
                       **kwargs):
        """
        Set the parameters specific to our attack
        :param model_type:
        :param loss_type:
        :param targeted:
        :param hinge_loss:
        :param adv_x:
        :param target_imgs:
        :param src_imgs:
        :param margin:
        :param model:
        :return:
        """
        self.model_type = params['model_type']
        self.loss_type = params['loss_type']
        self.TARGET_FLAG = params['targeted_flag']
        self.TV_FLAG = params['tv_flag']
        self.HINGE_FLAG = params['hinge_flag']
        self.target_imgs = target_imgs
        self.src_imgs = src_imgs
        self.margin = margin
        self.model = model
        self.LOSS_IMPL = params['mean_loss']

        if self.model_type == 'small':
            if self.loss_type == 'center':
                boxmin = -1
                boxmax = 1
            elif self.loss_type == 'triplet':
                boxmin = 0
                boxmax = 1
        elif self.model_type == 'large':
            boxmin = 0
            boxmax = 1
        
        boxmul = (boxmax - boxmin) / 2. #what is the rationale for this variable name?
        boxplus = (boxmin + boxmax) / 2.

        target_imgs_tanh = tf.tanh(self.target_imgs)*boxmul + boxplus
        src_imgs_tanh = tf.tanh(self.src_imgs) * boxmul + boxplus

        self.outputTarg = self.model.predict(target_imgs_tanh)
        self.outputSelf = self.model.predict(src_imgs_tanh)

        # self.outputSelf = tf.ones([batch_size, src_imgs.shape[0], self.outputSelf.get_shape()[1]]) * self.outputSelf
        # if not self.model_type == 'large':
        #     self.outputSelf = tf.transpose(self.outputSelf, [1, 0, 2])

        # self.outputTarg = tf.ones([batch_size, target_imgs.shape[0], self.outputTarg.get_shape()[1]]) * self.outputTarg
        # if not self.model_type == 'large':
        #     self.outputTarg = tf.transpose(self.outputTarg, [1, 0, 2])
        self.parse_params(**kwargs)
        self.old_x = base_imgs
        self.assign_adv_x = tf.placeholder(self.tf_dtype, self.old_x.shape)
        if self.rand_init:
            self.init_eta = tf.random_uniform(tf.shape(self.old_x), -self.eps, self.eps, dtype=self.tf_dtype)
            self.init_eta = clip_eta(self.init_eta, self.norm, self.eps)
        else:
            self.init_eta = tf.zeros_like(self.old_x)
        self.adv_x = tf.Variable(self.old_x + self.init_eta, dtype=self.tf_dtype)
        # Making unbounded adv_x bounded
        # def compute_loss(adv_x, x):
        self.newimg = tf.tanh(self.adv_x) * boxmul + boxplus
        self.oldimg = tf.tanh(self.old_x) * boxmul + boxplus

        # model.predict(x) obtains the fixed length embedding of x
        self.outputNew = self.model.predict(self.newimg)  # this returns the logits, can be pre-computed actually!
        self.outputOld = self.model.predict(self.oldimg)
        self.outputTargMean = tf.reduce_mean(self.outputTarg, axis=0)
        self.outputSelfMean = tf.reduce_mean(self.outputSelf, axis=0)

        if self.LOSS_IMPL == 'embeddingmean':
            self.target_loss = tf.sqrt(tf.reduce_sum(tf.square(self.outputNew - self.outputTargMean), [1]))
            self.src_loss = tf.sqrt(tf.reduce_sum(tf.square(self.outputNew - self.outputSelfMean), [1]))
            self.orig_loss = tf.sqrt(tf.reduce_sum(tf.square(self.outputOld - self.outputSelfMean), [1]))
        else:
            self.target_loss = tf.reduce_mean(tf.reduce_sum(tf.square(self.outputNew - self.outputTarg),1), axis=0)
            self.src_loss = tf.reduce_mean(tf.reduce_sum(tf.square(self.outputNew - self.outputSelf),1), axis=0)
            self.orig_loss = tf.reduce_mean(tf.reduce_sum(tf.square(self.outputOld - self.outputSelf),1), axis=0)

        def ZERO():
            return np.asarray(0., dtype=np.dtype('float32'))

        if self.TARGET_FLAG:

            if self.HINGE_FLAG:
                self.hinge_loss = self.target_loss - self.src_loss + self.margin
                self.hinge_loss = tf.maximum(self.hinge_loss, ZERO())
                self.loss = self.hinge_loss
            else:
                self.loss = self.target_loss

        else:
            self.loss = self.orig_loss - self.src_loss + self.margin
            self.loss = tf.maximum(self.loss, ZERO())
        
        if not self.TV_FLAG:
            self.loss = -self.loss
        else:
            if self.model_type == 'large':
                transpose_newimg = tf.transpose(self.newimg, (0, 3, 1, 2))
            else:
                transpose_newimg = self.newimg
            self.loss = self.loss + get_tv_loss(transpose_newimg)
            self.loss = -self.loss
        self.grad, = tf.gradients(self.loss, self.adv_x)
        self.scaled_signed_grad = self.eps_iter * tf.sign(self.grad)
        self.adv_x_out = self.adv_x + self.scaled_signed_grad
        if self.clip_min is not None and self.clip_max is not None:
            self.adv_x_out = tf.clip_by_value(self.adv_x_out, self.clip_min, self.clip_max)
        self.eta = self.adv_x_out - self.old_x
        self.eta = clip_eta(self.eta, self.norm, self.eps)
        self.setup = []
        self.setup.append(self.adv_x.assign(self.assign_adv_x))
            # return self.loss
        # self.loss = compute_loss(self.adv_x, self.old_x)
        # self.grad, = tf.gradients(self.loss, self.adv_x)
        # self.scaled_signed_grad = self.eps_iter * tf.sign(self.grad)
        # self.adv_x = self.adv_x + self.scaled_signed_grad
        # if self.clip_min is not None and self.clip_max is not None:
        #     self.adv_x = tf.clip_by_value(self.adv_x, self.clip_min, self.clip_max)
        # self.eta = self.adv_x - self.old_x
        # self.eta = clip_eta(self.eta, self.norm, self.eps)

    #same as cleverhans implementation
    def parse_params(self, 
                    eps=0.3, 
                    eps_iter=0.01, 
                    nb_iter=40, 
                    y=None,
                    norm=np.inf, 
                    clip_min=None, 
                    clip_max=None,
                    y_target=None, 
                    #rand_init=True, 
                    rand_init=None,
                    rand_init_eps=None,
                    clip_grad=False,
                    sanity_checks=True,
                    **kwargs):
        """
        Take in a dictionary of parameters and applies attack-specific checks
        before saving them as attributes.

        Attack-specific parameters:

        :param eps: (required float) maximum distortion of adversarial example
                    compared to original input
        :param eps_iter: (required float) step size for each attack iteration
        :param nb_iter: (required int) Number of attack iterations.
        :param y: (optional) A tensor with the model labels.
        :param y_target: (optional) A tensor with the labels to target. Leave
                         y_target=None if y is also set. Labels should be
                         one-hot-encoded.
        :param norm: (optional) Order of the norm (mimics Numpy). Possible values: np.inf, 1 or 2.
        :param clip_min: (optional float) Minimum input component value
        :param clip_max: (optional float) Maximum input component value
        :param rand_init: (optional bool) If True, an initial random perturbation is added.
        """

        # Save attack-specific parameters
        self.eps = eps
        self.eps_iter = eps_iter
        self.nb_iter = nb_iter
        self.y = y #current label
        self.y_target = y_target #target label
        self.norm = norm
        self.clip_min = clip_min
        self.clip_max = clip_max

        
        if rand_init is None:
            rand_init = self.default_rand_init
        self.rand_init = rand_init
    
        if rand_init_eps is None:
            rand_init_eps = self.eps
        self.rand_init_eps = rand_init_eps

        if isinstance(eps, float) and isinstance(eps_iter, float):
        # If these are both known at compile time, we can check before anything
        # is run. If they are tf, we can't check them yet.
            assert eps_iter <= eps, (eps_iter, eps)

        if self.y is not None and self.y_target is not None:
            raise ValueError("Must not set both y and y_target")
        # Check if normer of the norm is acceptable given current implementation
        if self.norm not in [np.inf, 1, 2]:
            raise ValueError("Norm normer must be either np.inf, 1, or 2.")

        if clip_grad and (self.clip_min is None or self.clip_max is None):
            raise ValueError("Must set clip_min and clip_max if clip_grad is set")

        self.sanity_checks = sanity_checks

        if len(kwargs.keys()) > 0:
            warnings.warn("kwargs is unused and will be removed on or after 2019-04-26.")
         
        #why return a true statement?
        return True 

    #from reference implementation
    def generate(self, 
                x, 
                **kwargs):
        """ 
        Generate symbolic graph for adversarial examples and return.

        :param x: The model's symbolic inputs.
        :param eps: (required float) maximum distortion of adversarial example
                    compared to original input
        :param eps_iter: (required float) step size for each attack iteration
        :param nb_iter: (required int) Number of attack iterations.
        :param y: (optional) A tensor with the model labels.
        :param y_target: (optional) A tensor with the labels to target. Leave
                         y_target=None if y is also set. Labels should be
                         one-hot-encoded.
        :param norm: (optional) Order of the norm (mimics Numpy).
                    Possible values: np.inf, 1 or 2.
        :param clip_min: (optional float) Minimum input component value
        :param clip_max: (optional float) Maximum input component value
        :param rand_init: (optional bool) If True, an initial random
                    perturbation is added.
        """

        # Parse and save attack-specific parameters
        assert self.parse_params(**kwargs)

        # labels, nb_classes = self.get_or_guess_labels(x, kwargs)
        self.targeted = self.y_target is not None

        # Initialize loop variables

        #print(x.shape)
        adv_x = self.attack(x)

        return adv_x, tf.norm(adv_x - x, ord=self.norm)

    #same as reference implementation
    def attack_single_step(self, 
                           x, 
                           eta,
                           first=False):
        """
        Given the original image and the perturbation computed so far, computes
        a new perturbation.

        :param x: A tensor with the original input.
        :param eta: A tensor the same shape as x that holds the perturbation.
        """
        # self.adv_x = x + eta
        # loss, eta, _ = self.sess.run([self.loss, self.eta, self.grad])
        

        # loss, eta = self.sess.run([self.loss, self.eta])
        if first:
            eta = np.squeeze(self.sess.run(eta))

        if len(eta.shape) < 4:
            eta = np.expand_dims(eta, axis=0)
        adv_x = np.squeeze(x + eta)
        if len(adv_x.shape) < 4:
            adv_x = np.expand_dims(adv_x, axis=0)
        self.sess.run(self.setup, {self.assign_adv_x: adv_x})
        loss, _, eta = self.sess.run([self.loss, self.adv_x, self.eta])
        # loss, _ = self.sess.run([self.loss, self.adv_x])
        # def get_loss(adv_x):
        #     loss, _ = self.sess.run([self.loss, self.adv_x])
        #     return self.loss
        # loss = get_loss(self.adv_x)

        # self.grad, = tf.gradients(self.loss, self.adv_x)
        # print(self.grad)
        # print(loss)
        # print(self.adv_x.shape)
        # print(eta.shape)
        # print(x.shape)
        # self.scaled_signed_grad = self.eps_iter * tf.sign(self.grad)
        # adv_x = adv_x + self.scaled_signed_grad
        # if self.clip_min is not None and self.clip_max is not None:
        #     adv_x = tf.clip_by_value(adv_x, self.clip_min, self.clip_max)
        # eta = adv_x - x
        # eta = clip_eta(eta, self.norm, self.eps)


        # adv_x = x + eta
        # loss = self.pgd_loss(adv_x, x)

        # grad, = tf.gradients(loss, adv_x)
        # print(grad)
        # print(self.sess.run(loss))
        # print(self.adv_x.shape)
        # scaled_signed_grad = self.eps_iter * tf.sign(grad)
        # adv_x = adv_x + scaled_signed_grad
        # if self.clip_min is not None and self.clip_max is not None:
        #     adv_x = tf.clip_by_value(adv_x, self.clip_min, self.clip_max)
        # eta = adv_x - x
        # eta = clip_eta(eta, self.norm, self.eps)
        return loss, eta

    #same as reference implementation
    def attack(self, 
                x):
        """
        :param x: A tensor with the input image.
        """

        if self.rand_init:
            eta = tf.random_uniform(tf.shape(x), -self.eps, self.eps, dtype=self.tf_dtype)
            eta = clip_eta(eta, self.norm, self.eps)
        else:
            eta = tf.zeros_like(x)

        first = True
        for i in range(self.nb_iter):
            loss, eta = self.attack_single_step(x, eta, first)
            print('iter: ', i, loss)
            first = False

        adv_x = x + eta
        if self.clip_min is not None and self.clip_max is not None:
            adv_x = tf.clip_by_value(adv_x, self.clip_min, self.clip_max)

        return adv_x

    def pgd_loss(self, 
                 adv_x,
                 x):
        """
        :param model_type: the model we are attacking
        :param targeted: binary, indicating targted or untargeted attack
        :param hinge_loss: binary, indicating whether using hinge loss
        :param adv_x: adv_x = x + delta (single image?)
        :param target_imgs: stack of target person faces, with shape [?,3,image_height,image_width]
        :param src_imgs: stack of faces belonging to 'x', with shape [?,3,image_height,image_width]
        :param margin: margin
        """

        '''
        Tensor shape and format should depend on the model
        Center loss model: image_height = 112, image_width = 96
                            images should be BGR
                            target_imgs, src_imgs should have pixel value range in [-1, 1]
        Triplet model:     image_height = 96, image_width = 96
                            images should be RGB
                            target_imgs, src_imgs should have pixel value range in [0, 1]
        '''

        if self.model_type == 'small':
            if self.loss_type == 'center':
                boxmin = -1
                boxmax = 1
            if self.loss_type == 'triplet':
                boxmin = 0
                boxmax = 1
        elif self.model_type == 'large':
            boxmin = 0
            boxmax = 1
        boxmul = (boxmax - boxmin) / 2.
        boxplus = (boxmin + boxmax) / 2.

        # Making unbounded adv_x bounded
        self.newimg = tf.tanh(adv_x) * boxmul + boxplus
        self.oldimg = tf.tanh(x) * boxmul + boxplus

        # model.predict(x) obtains the fixed length embedding of x
        self.outputNew = self.model.predict(self.newimg)  # this returns the logits, can be pre-computed actually!
        self.outputOld = self.model.predict(self.oldimg)
        self.outputTargMean = tf.reduce_mean(self.outputTarg, axis=0)
        self.outputSelfMean = tf.reduce_mean(self.outputSelf, axis=0)

        self.target_loss = tf.sqrt(tf.reduce_sum(tf.square(self.outputNew - self.outputTargMean), [1]))
        self.src_loss = tf.sqrt(tf.reduce_sum(tf.square(self.outputNew - self.outputSelfMean), [1]))
        self.orig_loss = tf.sqrt(tf.reduce_sum(tf.square(self.outputOld - self.outputSelfMean), [1]))

        def ZERO():
            return np.asarray(0., dtype=np.dtype('float32'))

        if self.TARGET_FLAG:

            if self.HINGE_FLAG:
                self.hinge_loss = self.target_loss - self.src_loss + self.margin
                self.hinge_loss = tf.maximum(self.hinge_loss, ZERO())
                self.loss = self.hinge_loss
            else:
                self.loss = self.target_loss

        else:
            self.loss = self.orig_loss - self.src_loss + self.margin
            self.loss = tf.maximum(self.loss, ZERO())
        
        if not self.TV_FLAG:
            return -self.loss
        else:
            if self.model_type == 'large':
                transpose_newimg = tf.transpose(self.newimg, (0, 3, 1, 2))
            else:
                transpose_newimg = self.newimg
            self.loss = self.loss + get_tv_loss(transpose_newimg)
            return -self.loss
        '''
        if self.TARGET_FLAG:
            # targeted case
            
            targ_loss = outputNew - self.outputTargMean
            if not self.model_type == 'large':
                targ_loss = tf.transpose(targ_loss, [1, 0, 2])
            # target_loss = tf.reduce_sum(tf.square(targ_loss), (1))
            target_loss = tf.norm(targ_loss, ord=2)
            loss = target_loss
            
            if self.hinge_loss:
                src_loss = outputNew - self.outputSelfMean
                if not self.model_type == 'large':
                    src_loss = tf.transpose(src_loss, [1, 0, 2])
                # loss = loss - tf.reduce_sum(tf.square(src_loss), (1)) + self.margin
                loss = -tf.norm(src_loss, ord=2)
                loss = tf.maximum(loss, 0)
        else:
            # untargeted case
            
            src_loss = outputNew - self.outputSelfMean
            if not self.model_type == 'large':
                src_loss = tf.transpose(src_loss, [1, 0, 2])
            # loss = -tf.reduce_sum(tf.square(src_loss), 1)  # we rely on broadcasting,
            loss = -tf.norm(src_loss, ord=2)
            # only one image can be fed a at time.
            # need to batch here; this will improve the situation greatly!
        loss /= batch_size
        
        return -loss
        '''
