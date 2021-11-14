from flask import Flask, render_template, request
from .utils.util import *


def get_model_transformers(classifier_type):
    if classifier_type == "lstm_dl":
        label_encoder = app.config["encoded_class"]
        pipeline = app.config["lstm_transform_pipeline"]
        model = app.config["lstm_model"]
    else:
        label_encoder = app.config["encoded_class"]
        pipeline = app.config["svm_transform_pipeline"]
        model = app.config["svm_model"]

    return label_encoder, pipeline, model


def predict_job_title(X, label_encoder, pipeline, model, classifier_type):
    if classifier_type == "lstm_dl":
        return predict_lstm_job_title(X, label_encoder, pipeline, model)
    else:
        return predict_svm_job_title(X, label_encoder, pipeline, model)


app = Flask(__name__)

var_dict = {"res":"",
            "text_val":"",
            "classifier_type":"svm"}


@app.route('/', methods=["GET", "POST"])
def run():
    request_type_str = request.method

    if request_type_str == "GET":
        return render_template("index.html")
    else:
        var_dict["text_val"] = request.form['text']
        var_dict["classifier_type"] = request.form['classifier_type']
        label_encoder, pipeline, model = get_model_transformers(var_dict["classifier_type"])
        var_dict["res"] = predict_job_title(var_dict["text_val"], label_encoder, pipeline, model,
                                            var_dict["classifier_type"])

        return render_template("index.html", res=var_dict["res"])

