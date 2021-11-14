from models.model import GanModel
from utils.config import Config
from utils.file_handler import create_class_res_directory
import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()


class GanLinearNoNormalization(GanModel):

    def _generator(self, Z, batch_size=tf.Variable(Config.BATCH_SIZE), hsize=Config.HIDDEN_LAYERS_NEURONS, reuse=False):
        with tf.variable_scope("GAN/Generator", reuse=reuse):
            h1 = tf.layers.dense(Z, hsize[0])
            h2 = tf.layers.dense(h1, hsize[1])
            out = tf.layers.dense(h2, Config.IMG_FLATTEN_SHAPE)
        return out

    def _discriminator(self, X, batch_size=tf.Variable(Config.BATCH_SIZE), hsize=Config.HIDDEN_LAYERS_NEURONS, reuse=False):
        with tf.variable_scope("GAN/Discriminator", reuse=reuse):
            h1 = tf.layers.dense(X, hsize[1])
            h2 = tf.layers.dense(h1, hsize[0])
            h3 = tf.layers.dense(h2, Config.IMG_FLATTEN_SHAPE)
            out = tf.layers.dense(h3, 1)
        return out, h3

    def __init__(self):
        super(GanLinearNoNormalization, self).__init__()

        self.sub_directory = "GanLinearNoNormalization"
        create_class_res_directory(self.sub_directory)

        self._X = tf.placeholder(tf.float32, [None, Config.IMG_FLATTEN_SHAPE])
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

