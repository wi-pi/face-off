## lp_attack.py -- attack a network optimizing for l_2 distance
##
## Copyright (C) 2016, Nicholas Carlini <nicholas@carlini.com>.
##
## This program is licenced under the BSD 2-Clause licence,
## contained in the LICENCE file in this directory.

import sys
import tensorflow as tf
import numpy as np
from attacks.tv_loss import get_tv_loss
from utils.recog import *

BINARY_SEARCH_STEPS = 9  # number of times to adjust the constant with binary search
MAX_ITERATIONS = 10000   # number of iterations to perform gradient descent
ABORT_EARLY = False       # if we stop improving, abort gradient descent early
LEARNING_RATE = 1e-2     # larger values converge faster to less accurate results
TARGETED_ATTACK = False          # should we target one specific class? or just be wrong?
CONFIDENCE = 0           # how strong the adversarial example should be
# INITIAL_CONST = 1e0     # the initial constant c to pick as a first guess
MARGIN = 0

class CW:
    def __init__(self, 
            sess, 
            model, 
            model_type = 'small',
            loss_type = 'triplet', 
            batch_size = 1, 
            num_src = 1, 
            num_target = 1, 
            face_stack_self = None,
            face_stack_target = None,
            confidence = CONFIDENCE,
            margin = MARGIN,
            targeted = TARGETED_ATTACK, 
            learning_rate = LEARNING_RATE,
            binary_search_steps = BINARY_SEARCH_STEPS, 
            max_iterations = MAX_ITERATIONS,
            abort_early = ABORT_EARLY, 
            hinge_loss = True,
            initial_const = 1e0,
            boxmin = 0, 
            boxmax = 1,
            p_norm = 2):
        
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

        image_height, image_width, num_channels = model.image_height, model.image_width, model.num_channels
        self.sess = sess
        self.model = model
        self.face_stack_self = face_stack_self
        self.face_stack_target = face_stack_target
        self.model_type = model_type
        self.loss_type = loss_type
        self.TARGETED_ATTACK = targeted
        self.LEARNING_RATE = learning_rate
        self.MAX_ITERATIONS = max_iterations
        self.BINARY_SEARCH_STEPS = binary_search_steps
        self.ABORT_EARLY = abort_early
        self.CONFIDENCE = confidence
        self.MARGIN = margin
        self.initial_const = initial_const
        self.batch_size = batch_size
        self.num_target = num_target
        self.num_src = num_src
        self.hinge_loss = hinge_loss

        self.repeat = binary_search_steps >= 10

        if self.model_type == 'large':
            shape = (batch_size, image_height, image_width, num_channels)
            target_shape = (num_target, image_height, image_width, num_channels)
            self_db_shape = (num_src, image_height, image_width, num_channels)
        else:
            shape = (batch_size, num_channels, image_width, image_height)
            target_shape = (num_target, num_channels, image_width, image_height)
            self_db_shape = (num_src, num_channels, image_width, image_height)
        
        modifier = tf.Variable(tf.random_uniform(shape, minval=-0.1, maxval=0.1, dtype=tf.float32))

        self.timg = tf.Variable(np.zeros(shape), dtype=tf.float32)
        self.const = tf.Variable(np.zeros(batch_size), dtype=tf.float32)
        
        self.targetimg = tf.Variable(np.zeros(target_shape), dtype=tf.float32)
        self.selfdb = tf.Variable(np.zeros(self_db_shape), dtype=tf.float32)

        self.assign_timg = tf.placeholder(tf.float32, shape)
        self.assign_const = tf.placeholder(tf.float32, [batch_size])
        
        self.assign_targetimg = tf.placeholder(tf.float32, target_shape)
        self.assign_selfdb = tf.placeholder(tf.float32, self_db_shape)
        
        # what are the 2 variables below? 
        self.boxmul = (boxmax - boxmin) / 2.
        self.boxplus = (boxmin + boxmax) / 2.
        
        # this condition is different from carlini's original implementation
        self.newimg = tf.tanh(modifier + self.timg) 
        self.newimg = self.newimg * self.boxmul + self.boxplus
        
        self.targetimg_bounded = tf.tanh(self.targetimg) 
        self.targetimg_bounded = self.targetimg_bounded * self.boxmul + self.boxplus
        
        self.selfdb_bounded = tf.tanh(self.selfdb) 
        self.selfdb_bounded = self.selfdb_bounded * self.boxmul + self.boxplus
        
        self.outputNew = model.predict(self.newimg)
        
        self.outputTarg = model.predict(self.targetimg_bounded)
        self.outputSelfdb = model.predict(self.selfdb_bounded)
        
        # distance to the input data
        # missing line from CW implementation

        #modify line below depending on p_norm to support \ell_\infty attacks
        if p_norm == 2:
            self.lpdist = tf.reduce_sum(tf.square(self.newimg-(tf.tanh(self.timg) * self.boxmul + self.boxplus)),[1,2,3])
        else:
            self.lpdist = tf.reduce_sum(tf.norm(self.newimg-(tf.tanh(self.timg) * self.boxmul + self.boxplus), ord=np.inf),[1,2,3]) #verify for correctness
        self.modifier_bounded = self.newimg-(tf.tanh(self.timg) * self.boxmul + self.boxplus)

        # missing lines from CW implementation

        self.outputTargMean = tf.reduce_mean(self.outputTarg, axis=0)
        self.outputSelfdbMean = tf.reduce_mean(self.outputSelfdb, axis=0)

        if self.TARGETED_ATTACK:
            target_loss = tf.reduce_sum(tf.square(self.outputNew - self.outputTargMean),1)
            
            if self.hinge_loss:
                hinge_loss = target_loss - tf.reduce_sum(tf.square(self.outputNew - self.outputSelfdbMean),1) + self.CONFIDENCE
                hinge_loss = tf.maximum(hinge_loss,0)
                loss1 = hinge_loss
            else:
                loss1 = target_loss
        
        else:
            target_loss = tf.reduce_sum(tf.square(self.outputNew - self.outputTargMean),1)
            loss1 = -target_loss
            #why no hinge component here?

        self.loss1 = tf.reduce_sum(self.const * loss1)
        self.loss2 = tf.reduce_sum(self.lpdist)
        
        #add condition to check if smoothing term is needed/not
        #self.loss = self.loss1 + self.loss2 + get_tv_loss(self.outputNew)
        self.loss = self.loss1 + self.loss2

        start_vars = set(x.name for x in tf.global_variables())
        optimizer = tf.train.AdamOptimizer(self.LEARNING_RATE)
        self.train = optimizer.minimize(self.loss, var_list=[modifier])
        end_vars = tf.global_variables()
        new_vars = [x for x in end_vars if x.name not in start_vars]

        self.setup = []
        self.setup.append(self.timg.assign(self.assign_timg))
        self.setup.append(self.const.assign(self.assign_const))
        self.setup.append(self.targetimg.assign(self.assign_targetimg))
        self.setup.append(self.selfdb.assign(self.assign_selfdb))
        
        self.init = tf.variables_initializer(var_list=[modifier]+new_vars)

    def attack(self, 
		imgs, 
		targetimg, 
		selfdb):
        """
        Perform the L_2 attack on the given images for the given targets.

        If self.targeted is true, then the targets represents the target labels.
        If self.targeted is false, then targets are the original class labels.
        """
        r = []
        for i in range(0,len(imgs),self.batch_size):
            r.extend(self.attack_batch(imgs[i:i+self.batch_size], targetimg, selfdb))
        return np.array(r) #(batch_size, 32, 32, 3)

    def attack_batch(self, 
                      imgs, 
                      targetimg, 
                      selfdb):
        """
        Run the attack on a batch of images and labels.
        """
        batch_size = self.batch_size

        imgs = np.arctanh((imgs - self.boxplus) / self.boxmul * 0.999999)
        targetimg = np.arctanh((targetimg - self.boxplus) / self.boxmul * 0.999999)
        selfdb = np.arctanh((selfdb - self.boxplus) / self.boxmul * 0.999999)

        CONST = np.ones(batch_size)*self.initial_const
	
        def modify(img, loss_type, model_type):
            return np.transpose(img, (0,1,3,2))

        for outer_step in range(self.BINARY_SEARCH_STEPS):
            self.sess.run(self.init)
            
            # The last iteration (if we run many steps) repeat the search once.
            if self.repeat and outer_step == self.BINARY_SEARCH_STEPS - 1:
                CONST = upper_bound

            self.sess.run(self.setup, {self.assign_timg: imgs,
                                       self.assign_const: CONST,
                                       self.assign_targetimg: targetimg,
                                       self.assign_selfdb: selfdb})
            
            prev = 1e6

            best_loss = 99999.0
            best_nimg = np.zeros(imgs.shape)
            best_temp_delta = np.zeros(imgs.shape)
            
            const_high = [10.0] * batch_size
            const_low = [0.05] * batch_size
            best_lp = [9999.0] * batch_size
            best_adv = [None] * batch_size
            best_delta = [None] * batch_size
            best_const = [None] * batch_size

            for iteration in range(self.MAX_ITERATIONS):
                _, l, lps, scores, nimg, delta = self.sess.run([self.train, 
                                                                self.loss, 
                                                                self.lpdist, 
                                                                self.outputNew, 
                                                                self.newimg, 
                                                                self.modifier_bounded])

                if iteration%(self.MAX_ITERATIONS//10) == 0:
                    print(iteration, self.sess.run((self.loss,self.loss1,self.loss2)))

                if self.ABORT_EARLY and iteration%(self.MAX_ITERATIONS//10) == 0:
                    if l > prev*.9999:
                        break
                    prev = l

                for e, (cur_nimg, cur_delta) in enumerate(zip(nimg, delta)):
                    best_nimg[e] = cur_nimg
                    best_temp_delta[e] = cur_delta
            
            for e in range(batch_size):
                current_nimg = np.expand_dims(best_nimg[e], axis=0)
                current_nimg = modify(current_nimg, self.model_type, self.loss_type) #@brian, varun: verify correctness across all combinations
                if self.model_type == 'small':
                    if self.loss_type == 'triplet':
                        dist = face_recog(face = current_nimg,
                                face_src = self.face_stack_self,
                                face_target = self.face_stack_target,
                                model = self.model,
                                sess = self.sess)
                    elif self.loss_type == 'center':
                        dist = face_recog_center(face = current_nimg,
                                        face_src = self.face_stack_self,
                                        face_target = self.face_stack_target,
                                        model = self.model,
                                        sess = self.sess)
                elif self.model_type == 'large':
                    dist = face_recog_large(face = current_nimg,
                                        face_src = self.face_stack_self,
                                        face_target = self.face_stack_target,
                                        model = self.model,
                                        sess = self.sess)
                
                dist_src = dist[0]
                dist_target = dist[1]
                if(dist_src - dist_target >= self.MARGIN):
                    #success condition
                    
                    adv_lp = np.linalg.norm(best_temp_delta[e]) #change based on norm
                    if(adv_lp) < best_lp[e]:
                        best_lp[e] = adv_lp
                        best_adv[e] = best_nimg[e]
                        best_delta[e] = delta[e]
                        best_const[e] = CONST[e]
                    # decrease const
                    const_high[e] = min(const_high[e], CONST[e])
                    if const_high[e] < 1e9:
                        CONST[e] = (const_high[e] + const_low[e]) / 2
                else:
                    #failure condition
                    
                    #increase const
                    const_low[e] = max(CONST[e], const_low[e])
                    if const_high[e] < 1e9:
                        CONST[e] = (const_high[e] + const_low[e]) / 2
                    else:
                        CONST[e] *= 10

        return best_lp, best_const, best_adv, best_delta
