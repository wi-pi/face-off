## lp_attack.py -- attack a network optimizing for l_2 distance
##
## Copyright (C) 2016, Nicholas Carlini <nicholas@carlini.com>.
##
## This program is licenced under the BSD 2-Clause licence,
## contained in the LICENCE file in this directory.

import sys, math
import tensorflow as tf
import numpy as np
from attacks.tv_loss import get_tv_loss
from cleverhans.compat import reduce_sum, reduce_max
from utils.recog import *
import Config

BINARY_SEARCH_STEPS = 9  # number of times to adjust the constant with binary search
MAX_ITERATIONS = 10000   # number of iterations to perform gradient descent
ABORT_EARLY = False       # if we stop improving, abort gradient descent early
LEARNING_RATE = 1e-2     # larger values converge faster to less accurate results
TARGET_FLAG = False          # should we target one specific class? or just be wrong?
CONFIDENCE = 0           # how strong the adversarial example should be
INITIAL_CONST = 1e-2     # the initial constant c to pick as a first guess
MARGIN = 0
TV_FLAG = False
LARGE = 1e10

#needed for l_inf attack
LARGEST_CONST = 2e+1    # the largest value of c to go up to before giving up
REDUCE_CONST = True    # try to lower c each iteration; faster to set to false
CONST_FACTOR = 4.0      # f>1, rate at which we increase constant, smaller better
DECREASE_FACTOR = 0.5   # 0<f<1, rate at which we shrink tau; larger is more accurate

class CW:
    def __init__(self,
                 sess,
                 model,
                 params,
                 num_base = 1,
                 num_src = 1,
                 num_target = 1,
                 confidence = CONFIDENCE,
                 margin = MARGIN,
                 abort_early = ABORT_EARLY,
                 hinge_loss = True,
                 largest_const = LARGEST_CONST,
                 reduce_const = REDUCE_CONST,
                 decrease_factor = DECREASE_FACTOR,
                 const_factor = CONST_FACTOR):
        
        """

        This attack is the most efficient and should be used as the primary 
        attack to evaluate potential defenses.

        Returns adversarial examples for the supplied model.

        confidence: Confidence of adversarial examples: higher produces examples
          that are farther away, but more strongly classified as adversarial.
        
        batch_size: Number of attacks to run simultaneously.
        
        targeted: True if we should perform a targetted attack, False otherwise.
        
        learning_rate: The learning rate for the attack algorithm. Smaller values
          produce better results but are slower to converge.
        
        binary_search_steps: The number of times we perform binary search to
          find the optimal tradeoff-constant between distance and confidence. 
        
        max_iterations: The maximum number of iterations. Larger values are more
          accurate; setting too small will require a large learning rate and will
          produce poor results.
        
        abort_early: If true, allows early aborts if gradient descent gets stuck.
        
        initial_const: The initial tradeoff-constant to use to tune the relative
        importance of distance and confidence. If binary_search_steps is large,
          the initial constant is not important.
        
        boxmin: Minimum pixel value (default -0.5).
        
        boxmax: Maximum pixel value (default 0.5).
        
        
        """

        #above: missing several parameter descriptions 
        Config.BM.mark('cw params')
        image_height, image_width, num_channels = model.image_height, model.image_width, model.num_channels
        self.sess = sess
        self.model = model
        self.model_type = params['model_type']
        self.loss_type = params['loss_type']
        self.TARGET_FLAG = params['targeted_flag']
        self.LEARNING_RATE = params['learning_rate']
        self.MAX_ITERATIONS = params['iterations']
        #params['iterations']
        self.BINARY_SEARCH_STEPS = params['binary_steps']
        self.ABORT_EARLY = abort_early
        self.CONFIDENCE = confidence
        self.MARGIN = margin
        if params['batch_size'] <= 0:
            self.batch_size = num_base
        else:
            self.batch_size = min(params['batch_size'], num_base)
        self.num_target = num_target
        self.num_src = num_src
        self.is_hinge_loss = params['hinge_flag']
        self.p_norm = params['norm']
        if self.p_norm != '2':
            self.batch_size = 1
        self.INITIAL_CONST = [params['init_const']] * self.batch_size
        self.TV_FLAG = params['tv_flag']
        self.COS_FLAG = params['cos_flag']
        self.LOSS_IMPL = params['mean_loss']
        self.boxmin = params['pixel_min']
        self.boxmax = params['pixel_max']

        #needed for l_inf attack
        self.LARGEST_CONST = largest_const
        self.DECREASE_FACTOR = decrease_factor
        self.REDUCE_CONST = reduce_const
        self.const_factor = const_factor

        self.repeat = self.BINARY_SEARCH_STEPS >= 10

        print('Batch size: {}'.format(self.batch_size))
        print('Margin: {}'.format(self.MARGIN))

        if self.model_type == 'large':
            shape = (self.batch_size, image_height, image_width, num_channels)
            target_db_shape = (num_target, image_height, image_width, num_channels)
            self_db_shape = (num_src, image_height, image_width, num_channels)
        else:
            shape = (self.batch_size, num_channels, image_width, image_height)
            target_db_shape = (num_target, num_channels, image_width, image_height)
            self_db_shape = (num_src, num_channels, image_width, image_height)
       
        print("shape:", shape) 
        modifier = tf.Variable(tf.random_uniform(shape, minval=-0.1, maxval=0.1, dtype=tf.float32))

        self.timg = tf.Variable(np.zeros(shape), dtype=tf.float32)
        self.const = tf.Variable(np.zeros(self.batch_size), dtype=tf.float32)
        self.tau = tf.Variable(np.zeros(1), dtype=tf.float32)

        self.assign_timg = tf.placeholder(tf.float32, shape)
        self.assign_const = tf.placeholder(tf.float32, [self.batch_size])
        self.assign_tau = tf.placeholder(tf.float32, [1])

        self.targetdb = tf.Variable(np.zeros(target_db_shape), dtype=tf.float32)
        self.selfdb = tf.Variable(np.zeros(self_db_shape), dtype=tf.float32)

        self.assign_targetdb = tf.placeholder(tf.float32, target_db_shape)
        self.assign_selfdb = tf.placeholder(tf.float32, self_db_shape)
        
        # what are the 2 variables below? 
        self.boxmul = (self.boxmax - self.boxmin) / 2.
        self.boxplus = (self.boxmin + self.boxmax) / 2.
        
        # this condition is different from carlini's original implementation
        self.newimg = tf.tanh(modifier + self.timg) 
        self.newimg = self.newimg * self.boxmul + self.boxplus
        
        self.targetdb_bounded = tf.tanh(self.targetdb) 
        self.targetdb_bounded = self.targetdb_bounded * self.boxmul + self.boxplus
        
        self.selfdb_bounded = tf.tanh(self.selfdb) 
        self.selfdb_bounded = self.selfdb_bounded * self.boxmul + self.boxplus
        
        self.outputNew = model.predict(self.newimg)
        # self.outputOld = model.predict(self.timg)
        
        self.outputTarg = model.predict(self.targetdb_bounded)
        self.outputSelf = model.predict(self.selfdb_bounded)


        if self.LOSS_IMPL == 'embeddingmean':
            if self.p_norm == '2':
                self.lpdist = tf.sqrt(tf.reduce_sum(tf.square(self.newimg - (tf.tanh(self.timg) * self.boxmul + self.boxplus)), [1,2,3]))
            else: #check this line below
                self.lpdist = tf.reduce_sum(tf.maximum(0.0, tf.abs(self.newimg - (tf.tanh(self.timg) * self.boxmul + self.boxplus)) - self.tau))
        else:
            if self.p_norm == '2':
                self.lpdist = tf.reduce_sum(tf.square(self.newimg-(tf.tanh(self.timg) * self.boxmul + self.boxplus)),[1,2,3])
            else:
                self.lpdist = tf.reduce_sum(tf.maximum(0.0, tf.abs(self.newimg - (tf.tanh(self.timg) * self.boxmul + self.boxplus)) - self.tau))

        self.modifier_bounded = self.newimg-(tf.tanh(self.timg) * self.boxmul + self.boxplus)

        self.outputTargMean = tf.reduce_mean(self.outputTarg, axis=0)
        self.outputSelfMean = tf.reduce_mean(self.outputSelf, axis=0)

        def ZERO():
            return np.asarray(0., dtype=np.dtype('float32'))

        self.target_loss = tf.sqrt(tf.reduce_sum(tf.square(self.outputNew - self.outputTargMean), [1]))
        self.src_loss = tf.sqrt(tf.reduce_sum(tf.square(self.outputNew - self.outputSelfMean), [1]))
        # self.orig_loss = tf.sqrt(tf.reduce_sum(tf.square(self.outputOld - self.outputSelfMean), [1]))
        
        if self.COS_FLAG:
            self.cosTargMean = tf.multiply(self.outputTargMean, np.ones(self.outputNew.shape))
            self.cosSelfMean = tf.multiply(self.outputSelfMean, np.ones(self.outputNew.shape))
            dot_targ = tf.reduce_sum(tf.multiply(self.outputNew, self.cosTargMean), [1])
            dot_src = tf.reduce_sum(tf.multiply(self.outputNew, self.cosSelfMean), [1])
            norm_targ = tf.sqrt(tf.reduce_sum(tf.square(self.outputNew), [1])) * tf.sqrt(tf.reduce_sum(tf.square(self.cosTargMean), [1]))
            norm_src = tf.sqrt(tf.reduce_sum(tf.square(self.outputNew), [1])) * tf.sqrt(tf.reduce_sum(tf.square(self.cosSelfMean), [1]))
            self.target_loss_cos = tf.acos(dot_targ / norm_targ) / math.pi * 180
            self.src_loss_cos = tf.acos(dot_src / norm_src) / math.pi * 180
            if self.TARGET_FLAG:
                if self.is_hinge_loss:
                    self.hinge_loss_cos = self.target_loss_cos - self.src_loss_cos + (self.CONFIDENCE * 6)
                    self.hinge_loss_cos = tf.maximum(self.hinge_loss_cos, ZERO())
                    self.loss4 = self.hinge_loss_cos
                else:
                    self.loss4 = self.target_loss_cos
        else:
            self.loss4 = 0
        
        if self.LOSS_IMPL == 'embeddingmean':
            if self.TARGET_FLAG:
                if self.is_hinge_loss:
                    self.hinge_loss = self.target_loss - self.src_loss + self.CONFIDENCE
                    self.hinge_loss = tf.maximum(self.hinge_loss, ZERO())
                    self.loss1 = self.hinge_loss
                else:
                    self.loss1 = self.target_loss
        
            else:
                # self.loss1 = self.orig_loss - self.src_loss + self.CONFIDENCE
                self.loss1 = tf.maximum(self.loss1, ZERO())
        
        else:
            if self.TARGET_FLAG:
                self.target_loss = tf.reduce_mean(tf.reduce_sum(tf.square(self.outputNew - self.outputTarg),1), axis=0)
                self.src_loss = tf.reduce_mean(tf.reduce_sum(tf.square(self.outputNew - self.outputSelf),1), axis=0)
                if self.is_hinge_loss:
                    self.hinge_loss = self.target_loss - self.src_loss + self.CONFIDENCE
                    self.hinge_loss = tf.maximum(self.hinge_loss, ZERO())
                    self.loss1 = self.hinge_loss
            else:
                self.loss1 = -tf.reduce_sum(tf.square(self.outputNew - self.outputTarg),1)

        #add condition to check if smoothing term is needed/not
        if not self.TV_FLAG:
            self.loss3 = 0
        else:
            if self.model_type == 'large':
                transpose_newimg = tf.transpose(self.newimg, (0, 3, 1, 2))
            else:
                transpose_newimg = self.newimg
            self.loss3 =  get_tv_loss(transpose_newimg)

        self.loss1 = tf.reduce_sum(self.const * self.loss1)
        self.loss2 = tf.reduce_sum(self.lpdist)
        self.loss4 = tf.reduce_sum(self.const * self.loss4)

        self.loss = self.loss1 + self.loss2 + self.loss3 + self.loss4

        start_vars = set(x.name for x in tf.global_variables())
        optimizer = tf.train.AdamOptimizer(self.LEARNING_RATE)
        self.train = optimizer.minimize(self.loss, var_list=[modifier])
        end_vars = tf.global_variables()
        new_vars = [x for x in end_vars if x.name not in start_vars]

        self.setup = []
        self.setup.append(self.timg.assign(self.assign_timg))
        self.setup.append(self.const.assign(self.assign_const))
        self.setup.append(self.targetdb.assign(self.assign_targetdb))
        self.setup.append(self.selfdb.assign(self.assign_selfdb))
        self.setup.append(self.tau.assign(self.assign_tau))
        
        self.init = tf.variables_initializer(var_list=[modifier]+new_vars)
        Config.BM.mark('cw params')

    def attack(self, 
               imgs, 
               target_imgs, 
               src_imgs,
               params):
        """
        Perform the L_2 attack on the given images for the given targets.

        If self.targeted is true, then the targets represents the target labels.
        If self.targeted is false, then targets are the original class labels.
        """
        lp_list = []
        const_list = []
        adv_list = []
        delta_list = []
        if self.p_norm != '2':
            batch_size = 1
        else:
            batch_size = self.batch_size
        for i in range(0,len(imgs),batch_size):
            cur_batch = len(imgs) - i
            if self.batch_size > cur_batch:
                self.__init__(sess=self.sess,
                              model=self.model,
                              params=params,
                              num_base=cur_batch,
                              num_src=self.num_src,
                              num_target=self.num_target,
                              confidence=self.MARGIN,
                              margin=self.MARGIN)
            if self.p_norm == '2':
                lp, const, adv, delta = self.attack_batch_l2(imgs[i:i+self.batch_size], target_imgs, src_imgs)
            else:
                #lp, const, adv, delta = self.attack_batch_linf(imgs[i:i+self.batch_size], target_imgs, src_imgs)
                lp, const, adv, delta = self.attack_batch_linf(imgs[i:i+1], target_imgs, src_imgs)

            lp_list.extend(lp)
            const_list.extend(const)
            adv_list.extend(adv)
            delta_list.extend(delta)
        r = np.squeeze(np.array([(lp_list, const_list, adv_list, delta_list)]))
        return r


    def attack_batch_l2(self, 
                     imgs, 
                     target_imgs, 
                     src_imgs):
        """
        Run the attack on a batch of images and labels.
        """
        Config.BM.mark('imgsetup')
        batch_size = self.batch_size

        imgs = np.arctanh((imgs - self.boxplus) / self.boxmul * 0.999999)
        face_stack_target = np.arctanh((target_imgs - self.boxplus) / self.boxmul * 0.999999)
        face_stack_self = np.arctanh((src_imgs - self.boxplus) / self.boxmul * 0.999999)

        CONST = np.ones(batch_size)*self.INITIAL_CONST

        const_high = [1e3] * batch_size
        const_low = [0.0] * batch_size
        
        best_lp = [9999.0] * batch_size
        best_adv = [None] * batch_size
        best_delta = [None] * batch_size
        best_const = [None] * batch_size
        Config.BM.mark('imgsetup')
        for outer_step in range(self.BINARY_SEARCH_STEPS):
            Config.BM.mark('init')
            self.sess.run(self.init)
           
            best_loss_inner = [1e10] * batch_size
            best_adv_inner = [None] * batch_size
            best_delta_inner = [None] * batch_size
            best_dist_src = [None] * batch_size
            best_dist_target = [None] * batch_size
            
            # The last iteration (if we run many steps) repeat the search once.
            if self.repeat and outer_step == self.BINARY_SEARCH_STEPS - 1:
                CONST = const_high
            Config.BM.mark('init')
            Config.BM.mark('cw assign')
            self.sess.run(self.setup, {self.assign_timg: imgs,
                                       self.assign_const: CONST,
                                       self.assign_targetdb: face_stack_target,
                                       self.assign_selfdb: face_stack_self,
                                       self.assign_tau: np.zeros(1)})
            Config.BM.mark('cw assign')
            prev = 1e6
            for iteration in range(self.MAX_ITERATIONS):
                _, l, nimg, delta, dist_src, dist_target = self.sess.run([self.train, 
                                                                          self.loss, 
                                                                          self.newimg, 
                                                                          self.modifier_bounded,
                                                                          self.src_loss,
                                                                          self.target_loss])
                                                                          # self.orig_loss])
                if iteration%(self.MAX_ITERATIONS//10) == 0:
                    print('Step: {}, Iteration: {}, Loss: {}'.format(outer_step, iteration, l))
                    # print(dist_src)
                    # print(dist_target)

                if self.ABORT_EARLY and iteration%(self.MAX_ITERATIONS//10) == 0:
                    if l > prev*.9999:
                        break
                    prev = l
                
                for e, (cur_adv, cur_delta) in enumerate(zip(nimg, delta)):
                    if l < best_loss_inner[e]:
                        best_adv_inner[e] = cur_adv
                        best_delta_inner[e] = cur_delta
                        if self.LOSS_IMPL == 'embeddingmean':
                            best_dist_src[e] = dist_src[e]
                            if self.TARGET_FLAG:
                                best_dist_target[e] = dist_target[e]
                            # else:
                                # best_dist_target[e] = dist_orig[e]
                            if best_dist_src[e] - best_dist_target[e] >= self.MARGIN:
                                best_loss_inner[e] = l
                        else:
                            best_dist_src[e] = dist_src
                            if self.TARGET_FLAG:
                                best_dist_target[e] = dist_target
                            # else:
                                # best_dist_target[e] = dist_orig[e]
                            if best_dist_src[e] - best_dist_target >= self.MARGIN:
                                best_loss_inner[e] = l

            for e in range(batch_size):
                if self.TARGET_FLAG:
                    print('Img: {}, Distance(source): {}, Distance(target): {}, Margin: {}'.format(e, best_dist_src[e], best_dist_target[e], self.MARGIN))
                else:
                    print('Img: {}, Distance(adversarial): {}, Distance(original): {}, Margin: {}'.format(e, best_dist_src[e], best_dist_target[e], self.MARGIN))

                if(best_dist_src[e] - best_dist_target[e] >= self.MARGIN):
                    #success condition, decrease const
                    adv_lp = np.linalg.norm(best_delta_inner[e])
                    if adv_lp < best_lp[e]:
                        best_lp[e] = adv_lp
                        best_adv[e] = best_adv_inner[e]
                        best_delta[e] = best_delta_inner[e]
                        best_const[e] = CONST[e]
                    temp_const = CONST[e]
                    const_high[e] = min(const_high[e], CONST[e])
                    if const_high[e] < 1e9:
                        CONST[e] = (const_high[e] + const_low[e]) / 2
                    print('Img: {}, decrease const between {} and {}'.format(e, temp_const, CONST[e]))
                else:
                    #failure condition, increase const
                    temp_const = CONST[e]
                    const_low[e] = max(CONST[e], const_low[e])
                    if const_high[e] < 1e9:
                        CONST[e] = (const_high[e] + const_low[e]) / 2
                    else:
                        CONST[e] *= 10
                    print('Img: {}, increase const between {} and {}'.format(e, temp_const, CONST[e]))
        return best_lp, best_const, best_adv, best_delta


    def attack_batch_linf(self,
                          imgs,
                          target_imgs,
                          src_imgs):
        
        
        def doit(imgs, src_imgs, target_imgs, tt, CONST, batch_size):
            # convert to tanh-space
            '''
            best_loss_inner = [1e10] * batch_size
            best_adv_inner = [None] * batch_size
            best_delta_inner = [None] * batch_size
            best_dist_src = [None] * batch_size
            best_dist_target = [None] * batch_size
            best_lp_inner = [None] * batch_size
            '''

            best_loss_inner = LARGE
            best_adv_inner = None
            best_delta_inner = None
            best_dist_src = None
            best_dist_target = None
            best_lp_inner = None

            imgs = np.arctanh((imgs - self.boxplus) / self.boxmul * 0.999999)
            face_stack_target = np.arctanh((target_imgs - self.boxplus) / self.boxmul * 0.999999)
            face_stack_self = np.arctanh((src_imgs - self.boxplus) / self.boxmul * 0.999999)

            
            # initialize the variables
            self.sess.run(self.init)
            self.sess.run(self.setup, {self.assign_timg: imgs,
                                       self.assign_const: CONST,
                                       self.assign_targetdb: face_stack_target,
                                       self.assign_selfdb: face_stack_self,
                                       self.assign_tau: [tt]})
            #terminate = False
            return_flag = False
            while CONST[0] < self.LARGEST_CONST:
            #while not terminate:
                # try solving for each value of the constant
                print('try const', CONST[0])
                for step in range(self.MAX_ITERATIONS):
                    feed_dict={self.const: CONST}

                    # perform the update step
                    _, l, nimg, delta, dist_src, dist_target = self.sess.run([self.train,
                                                                              self.loss,
                                                                              self.newimg,
                                                                              self.modifier_bounded,
                                                                              self.src_loss,
                                                                              self.target_loss],
                                                                              # self.orig_loss],
                                                                              feed_dict=feed_dict)
                    if step%(self.MAX_ITERATIONS//10) == 0:
                        print('Step: {}, Loss: {}'.format(step, l))
                    
                    if dist_src - dist_target >= self.MARGIN:
                        #print("loss:", l)
                        if l < best_loss_inner:
                            best_loss_inner = l
                            best_adv_inner = nimg
                            best_delta_inner = delta
                            best_dist_src = dist_src
                            best_dist_target = dist_target
                            best_lp_inner = np.linalg.norm(best_delta_inner)
                            return_flag = True
                    
                if not return_flag:
                    CONST[0] *= self.const_factor
                else:
                    return best_lp_inner, CONST, best_adv_inner, best_delta_inner
            
            return best_lp_inner, CONST, best_adv_inner, best_delta_inner


        batch_size = imgs.shape[0]
        
        prev_lp = LARGE
        prev_const = None
        prev_adv = None
        prev_delta = None
        tau = 1.0
        const = self.INITIAL_CONST
        terminate = False
        trials = 0
        while tau > 1./256 or trials <= 2:
            print("trial:", trials)
            # try to solve given this tau value
            res = doit(imgs, src_imgs, target_imgs, tau, const, batch_size)
            lp, const, adv, delta = res
            if lp != None and lp <= prev_lp:
                prev_lp = lp
                prev_const = const
                prev_adv = adv
                prev_delta = delta
                
            if self.REDUCE_CONST: const[0] /= 2

            # the attack succeeded, reduce tau and try again
            if prev_adv is not None:
                actualtau = np.max(np.abs(prev_adv-imgs))

                if actualtau < tau:
                    tau = actualtau
            else:
                print("Attack Failure")
                return [None], [None], [imgs], [None]

            print("Tau",tau)

            tau *= self.DECREASE_FACTOR
            trials += 1
        if prev_lp != LARGE:
            print("Attack Success")
            return [prev_lp], [prev_const], [prev_adv], [prev_delta]
        else:
            print("Attack Failure")
            return [None], [None], [imgs], [None]

