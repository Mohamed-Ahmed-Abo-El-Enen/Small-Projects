from models.model import GanModel
from utils.config import Config
from utils.file_handler import create_class_res_directory
import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()


class Gan2DCnn2DBatchNormalization(GanModel):

    def _generator(self, Z, batch_size=tf.Variable(Config.BATCH_SIZE), hsize=Config.CNN_HIDDEN_LAYERS_NEURONS, reuse=False):
        with tf.variable_scope("GAN/Generator", reuse=reuse):
            x = tf.layers.dense(Z, 7 * 7 * hsize[0], use_bias=False)
            x = tf.layers.batch_normalization(x)
            x = tf.nn.leaky_relu(x)

            x = tf.reshape(x, (tf.shape(Z)[0], 7, 7, hsize[0]))
            # print(x.shape)

            x = tf.layers.conv2d_transpose(x, hsize[1], kernel_size=5, strides=1, padding='same', use_bias=False)
            x = tf.layers.batch_normalization(x)
            x = tf.nn.leaky_relu(x)
            # print(x.shape)

            x = tf.layers.conv2d_transpose(x, hsize[2], kernel_size=5, strides=2, padding='same', use_bias=False)
            x = tf.layers.batch_normalization(x)
            x = tf.nn.leaky_relu(x)
            # print(x.shape)

            x = tf.layers.conv2d_transpose(x, 1, kernel_size=5, strides=2, padding='same', use_bias=False)
            outputs = tf.nn.tanh(x)
            # print(outputs.shape)

        return outputs

    def _discriminator(self, inputs, batch_size=tf.Variable(Config.BATCH_SIZE), hsize=Config.CNN_HIDDEN_LAYERS_NEURONS, reuse=False):
        with tf.variable_scope("GAN/Discriminator", reuse=reuse):
            x = tf.layers.conv2d(inputs, hsize[0], kernel_size=5, strides=2, padding='same')
            x = tf.nn.leaky_relu(x)
            x = tf.layers.dropout(x, 0.3)
            # print(x.shape)

            x = tf.layers.conv2d(x, hsize[1], kernel_size=5, strides=2, padding='same')
            x = tf.nn.leaky_relu(x)
            x = tf.layers.dropout(x, 0.3)
            # print(x.shape)

            x = tf.layers.flatten(x)
            outputs = tf.layers.dense(x, 1)
        return outputs, x

    def __init__(self):
        super(Gan2DCnn2DBatchNormalization, self).__init__()

        self.sub_directory = "Gan2DCnn2DBatchNormalization"
        create_class_res_directory(self.sub_directory)

        self._X = tf.placeholder(tf.float32, [None, Config.IMG_H, Config.IMG_W, Config.IMG_C])
        self._Z = tf.placeholder(tf.float32, [None, Config.LATENT_DIM])

        self._G_sample = self._generator(self._Z)
        r_logits, r_rep = self._discriminator(self._X)
        f_logits, g_rep = self._discriminator(self._G_sample, reuse=True)

        real_loss = tf.nn.sigmoid_cross_entropy_with_logits(logits=r_logits,
                                                            labels=tf.ones_like(r_logits))
        fake_loss = tf.nn.sigmoid_cross_entropy_with_logits(logits=f_logits,
                                                            labels=tf.zeros_like(f_logits))
        self._disc_loss = tf.reduce_mean(real_loss + fake_loss)
        self._gen_loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=f_logits,
                                                                                labels=tf.ones_like(f_logits)))

        gen_vars = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope="GAN/Generator")
        disc_vars = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope="GAN/Discriminator")

        self._gen_step = tf.train.RMSPropOptimizer(learning_rate=Config.LEARNING_RATE).minimize(self._gen_loss,
                                                                                                var_list=gen_vars)
        self._disc_step = tf.train.RMSPropOptimizer(learning_rate=Config.LEARNING_RATE).minimize(self._disc_loss,
                                                                                                 var_list=disc_vars)

        self._sess = tf.Session()
        tf.global_variables_initializer().run(session=self._sess)

