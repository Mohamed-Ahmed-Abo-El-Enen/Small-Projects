import os
from utils.config import Config
from utils.generate_latent_dim import sample_Z
from utils.batch_generator import generate_batches
from utils.mnist_read import flatten_x_dataset
from utils.file_handler import save_loss_file, create_loss_file
from utils.visualization import visualize_reconstructed_img
import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()
tf.set_random_seed(1)


class GanModel(object):
    _sess = None

    def _generator(self, Z, batch_size=tf.Variable(Config.BATCH_SIZE), hsize=Config.HIDDEN_LAYERS_NEURONS, reuse=False):
        pass

    def _discriminator(self, X, batch_size=tf.Variable(Config.BATCH_SIZE), hsize=Config.HIDDEN_LAYERS_NEURONS, reuse=False):
        pass

    def __init__(self):
        tf.reset_default_graph()

        self._G_sample = None
        self._Batch_size = tf.Variable(0)
        self._X = None
        self._Z = None
        self._disc_step = None
        self._disc_loss = None
        self._gen_step = None
        self._gen_loss = None
        self.sub_directory = ""

    def predict(self, Z_batch):
        G_yhat = self._sess.run(self._G_sample, feed_dict={self._Z: Z_batch,
                                                           self._Batch_size: Z_batch.shape[0]})
        return G_yhat

    def train(self, X_train, y_train, flatten_images=True):
        f = create_loss_file(self.sub_directory)
        current_batch_size = Config.BATCH_SIZE
        dloss = 0
        gloss = 0
        for i in range(Config.EPOCHS + 1):
            for X_batch, y_batch in generate_batches(X_train, y_train, Config.BATCH_SIZE):
                if flatten_images:
                    X_batch = flatten_x_dataset(X_batch)
                current_batch_size = X_batch.shape[0]
                Z_batch = sample_Z(current_batch_size, Config.LATENT_DIM)

                _, dloss = self._sess.run([self._disc_step, self._disc_loss], feed_dict={self._X: X_batch,
                                                                                         self._Z: Z_batch,
                                                                                         self._Batch_size: current_batch_size})
                _, gloss = self._sess.run([self._gen_step, self._gen_loss], feed_dict={self._Z: Z_batch,
                                                                                       self._Batch_size: current_batch_size})

            save_fig = False

            if i == int((Config.EPOCHS + 1) / 2) or i == Config.EPOCHS:
                save_fig = True

            if i % Config.PLOTTING_SAVING_STEP == 0:
                print("Iterations: %d\t Discriminator loss: %.4f\t Generator loss: %.4f" % (i, dloss, gloss))
                save_loss_file(f, i, dloss, gloss)

            if save_fig or i % Config.PLOTTING_SAVING_STEP == 0:
                Z_batch = sample_Z(current_batch_size, Config.LATENT_DIM)
                G_yhat = self.predict(Z_batch)
                visualize_reconstructed_img(os.path.join(self.sub_directory, "plots"), G_yhat, itr=i, save_fig=save_fig)

        f.close()