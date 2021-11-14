import numpy as np


def predict_svm_job_title(X, label_encoder, pipeline, model):
    test_np_input = pipeline.transform([X])
    y_hat = model.predict(test_np_input)[0]
    return str(label_encoder[y_hat])


def predict_lstm_job_title(X, label_encoder, pipeline, model):
    test_np_input = pipeline.transform([X])
    y_hat = np.argmax(model.predict(test_np_input)[0])
    return str(label_encoder[y_hat])