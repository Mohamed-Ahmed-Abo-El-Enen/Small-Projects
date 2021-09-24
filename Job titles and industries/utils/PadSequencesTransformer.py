from sklearn.base import BaseEstimator, TransformerMixin
from keras.preprocessing.sequence import pad_sequences


class PadSequencesTransformer(BaseEstimator, TransformerMixin):

    def __init__(self, maxlen):
        self.maxlen = maxlen

    def fit(self, X, y=None):
        return self

    def transform(self, X, y=None):
        X_padded = pad_sequences(X, maxlen=self.maxlen, padding="post")
        return X_padded
