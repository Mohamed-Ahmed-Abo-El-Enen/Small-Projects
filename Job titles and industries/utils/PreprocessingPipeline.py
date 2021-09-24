import numpy as np
from os.path import join, dirname, realpath
from joblib import load
from keras.models import load_model
from utils.PadSequencesTransformer import PadSequencesTransformer
from utils.TokenizerTransformer import TokenizerTransformer


def load_label_encoder(encoded_class_path):
    return np.load(join(dirname(realpath(__file__)), "model\\"+encoded_class_path), allow_pickle=True)


def load_preprocessing_pipeline(transform_pipeline_path):
    return load(join(dirname(realpath(__file__)), "model\\"+transform_pipeline_path))


def load_svm_model(model_path):
    return load(join(dirname(realpath(__file__)), "model\\"+model_path))


def load_lstm_model(model_path):
    return load_model(join(dirname(realpath(__file__)), "model\\"+model_path))
