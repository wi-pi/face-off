import tensorflow as tf

duplicate_rgb = True

def get_args(read_input: bool = False, duplicate_rgb: bool = False):
    if read_input:
        return [float(input(var + '\n')) for var in ['tv_eps (0.2)', 'tv_lam (0.3)', 'reg_lam(1.0)']]
    if duplicate_rgb:
        return [200.5, 0.000009, 0.02]
    return [0.2, 0.0, 0.0, 0.0001]

def _tv_diff(c):
    x_wise = c[:, :, 1:] - c[:, :, :-1]
    y_wise = c[:, 1:, :] - c[:, :-1, :]
    return x_wise, y_wise

def smooth_tv(c):
    x_wise, y_wise = _tv_diff(c)
    return tf.reduce_sum(tf.multiply(x_wise, x_wise)) + tf.reduce_sum(tf.multiply(y_wise, y_wise))

def get_tv(t):
    return smooth_tv(t[:, 0]) + smooth_tv(t[:, 1]) + smooth_tv(t[:, 2])

def get_ct(x):
    t_shape = x[:, :1, :, :].shape
    t = tf.random_uniform(t_shape, minval=0, maxval=1) - 0.5
    repeat_t = tf.keras.backend.repeat_elements(t, 3, axis=1)
    return repeat_t

def get_tv_loss(x):
    #assumes x is of shape (N,3,b,w), where N is batch size, 3 is num channels & b,w are standard
    ct = get_ct(x)
    tv_eps, tv_lam, reg_lam = get_args(duplicate_rgb=duplicate_rgb)
    tv_loss = tv_lam * get_tv(ct)
    return tv_loss


if __name__ == "__main__":
    x = tf.random_uniform([4,32,32,3])
    print(x.shape)
    y = get_tv_loss(x)
    print(y)
