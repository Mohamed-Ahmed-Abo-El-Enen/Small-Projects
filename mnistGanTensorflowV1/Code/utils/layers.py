import tensorflow.compat.v1 as tf


def _l2normalize(v, eps=1e-12):
    return v / (tf.reduce_sum(v ** 2) ** 0.5 + eps)


def spectral_normed_weight(weights, num_iters=1, update_collection=None, with_sigma=False):
    w_shape = weights.shape.as_list()
    w_mat = tf.reshape(weights, [-1, w_shape[-1]])
    u = tf.get_variable('u', [1, w_shape[-1]],
                        initializer=tf.truncated_normal_initializer(),
                        trainable=False)
    u_ = u
    v_ = tf.ones_like(w_mat)
    for _ in range(num_iters):
        v_ = _l2normalize(tf.matmul(u_, w_mat, transpose_b=True))
        u_ = _l2normalize(tf.matmul(v_, w_mat))

    sigma = tf.squeeze(tf.matmul(tf.matmul(v_, w_mat), u_, transpose_b=True))
    w_mat /= sigma
    if update_collection is None:
        with tf.control_dependencies([u.assign(u_)]):
            w_bar = tf.reshape(w_mat, w_shape)
    else:
        w_bar = tf.reshape(w_mat, w_shape)
        if update_collection != 'NO_OPS':
            tf.add_to_collection(update_collection, u.assign(u_))
    if with_sigma:
        return w_bar, sigma
    else:
        return w_bar


def snconv2d(x, filters, kernel_size=4, strides=2, padding='SAME', use_bias=False, sn=True, name="snconv2d"):
    with tf.variable_scope(name):
        if sn:
            w = tf.get_variable("kernel", shape=[kernel_size, kernel_size, x.get_shape()[-1], filters],
                                initializer=tf.random_normal_initializer(mean=0.0, stddev=0.02),
                                regularizer=None)
            bias = tf.get_variable("bias", [filters], initializer=tf.constant_initializer(0.0))
            x = tf.nn.conv2d(input=x, filter=spectral_normed_weight(w),
                             strides=[1, strides, strides, 1], padding=padding)
            if use_bias:
                x = tf.nn.bias_add(x, bias)

        else:
            x = tf.layers.conv2d(inputs=x, filters=filters,
                                 kernel_size=kernel_size,
                                 strides=strides, use_bias=use_bias, padding=padding)
    return x


def sndeconv2d(x, batch_size, filters, kernel_size=4, strides=2, padding='SAME', use_bias=False, sn=True, name="sndeconv2d"):
    with tf.variable_scope(name):
        x_shape = x.get_shape().as_list()
        output_shape = [batch_size, x_shape[1] * strides, x_shape[2] * strides, filters]
        if sn:
            w = tf.get_variable("kernel", shape=[kernel_size, kernel_size, filters, x.get_shape()[-1]],
                                initializer=tf.random_normal_initializer(mean=0.0, stddev=0.02),
                                regularizer=None)
            x = tf.nn.conv2d_transpose(x, filter=spectral_normed_weight(w), output_shape=output_shape,
                                       strides=[1, strides, strides, 1], padding=padding)

            if use_bias:
                bias = tf.get_variable("bias", [filters], initializer=tf.constant_initializer(0.0))
                x = tf.nn.bias_add(x, bias)

        else:
            x = tf.layers.conv2d_transpose(inputs=x, filters=filters,
                                           kernel_size=kernel_size,
                                           strides=strides, padding=padding, use_bias=use_bias)
        return x

    