from models.GanLinearNoNormalization import GanLinearNoNormalization
from models.GanLinearWithBatchNormalization import GanLinearWithBatchNormalization
from models.Gan2DCnn2DBatchNormalization import Gan2DCnn2DBatchNormalization
from models.Gan2DCnnSpectralNormedWeight import Gan2DCnnSpectralNormedWeight
from utils.generate_latent_dim import sample_Z
from utils.config import Config
from utils.visualization import visualize_reconstructed_img
import os


def select_train_model(X_train, y_train):
    print("********************************************\n")
    print("Mnist Gan Models")
    print("********************************************\n")
    print("Enter \n"
          "1- for model q1 (GanLinearNoNormalization)\n"
          "2- for model q1 (GanLinearWithBatchNormalization)\n"
          "3- for model q1 (Gan2DCnn2DBatchNormalization)\n"
          "4- for model q1 (Gan2DCnnSpectralNormedWeight)\n"
          "else for model q1 (GanLinearNoNormalization)\n")
    model_num = int(input())
    model = None

    if model_num == 2:
        model = GanLinearWithBatchNormalization()
        model.train(X_train, y_train, flatten_images=True)
    elif model_num == 3:
        model = Gan2DCnn2DBatchNormalization()
        model.train(X_train, y_train, flatten_images=False)
    elif model_num == 4:
        model = Gan2DCnnSpectralNormedWeight()
        model.train(X_train, y_train, flatten_images=False)
    else:
        model = GanLinearNoNormalization()
        model.train(X_train, y_train, flatten_images=True)

    Z_batch = sample_Z(Config.NUMBER_TEST, Config.LATENT_DIM)
    G_yhat = model.predict(Z_batch)
    visualize_reconstructed_img(os.path.join(model.sub_directory, "plots"), G_yhat, itr=-1, save_fig=True)
    return model


def model_predict(model):
    print("Enter number of testing samples above 10 just to visualize result in right shape: ")
    num_samples = int(input())
    X_test = sample_Z(num_samples, Config.LATENT_DIM)
    G_yhat = model.predict(X_test)
    visualize_reconstructed_img(os.path.join(model.sub_directory, "plots"), G_yhat, itr=-1, save_fig=False)
