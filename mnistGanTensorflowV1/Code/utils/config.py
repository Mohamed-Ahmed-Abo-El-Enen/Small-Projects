class Config:
    IMG_H = 28
    IMG_W = 28
    IMG_C = 1
    IMG_FLATTEN_SHAPE = IMG_H*IMG_W*IMG_C
    BATCH_SIZE = 256
    EPOCHS = 100
    HIDDEN_LAYERS_NEURONS = [16, 16]
    CNN_HIDDEN_LAYERS_NEURONS = [256, 128, 64]
    LEARNING_RATE = 0.001
    LATENT_DIM = 10
    PLOTTING_SAVING_STEP = 10
    NUMBER_TEST = 10


def recalculate_dim(H, W, C):
    Config.IMG_H = H
    Config.IMG_W = W
    Config.IMG_C = C
    Config.IMG_FLATTEN_SHAPE = H * W * C