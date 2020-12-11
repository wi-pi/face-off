import scipy.misc as ms 
import imageio
import numpy as np 
import os,sys
from utils.crop import *
import tensorflow as tf
import cv2
from keras.models import load_model

#@brian: why dist_src and dist_target measured in terms of the 2 norm?


def face_recog(face, face_src, face_target, model, sess, input_tensor=None):
    embedding = model.predict(input_tensor)

    encoding_face = sess.run(embedding, feed_dict={input_tensor: face})
    encoding_src = sess.run(embedding, feed_dict={input_tensor: face_src})
    encoding_target = sess.run(embedding, feed_dict={input_tensor: face_target})

    encoding_src_mean = np.reshape(np.mean(encoding_src, axis=0), (1, -1))
    encoding_target_mean = np.reshape(np.mean(encoding_target, axis=0), (1, -1))
    dist_src = np.linalg.norm(encoding_face - encoding_src_mean)
    dist_target = np.linalg.norm(encoding_face - encoding_target_mean)
    
    return (dist_src, dist_target)
