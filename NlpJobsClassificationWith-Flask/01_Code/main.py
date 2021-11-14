from app.app import *
from utils.PreprocessingPipeline import *

if __name__ == "__main__":
    app.config["encoded_class"] = load_label_encoder("encoded_class.npy")
    app.config["svm_transform_pipeline"] = load_preprocessing_pipeline("SVM_Transorm_Pipleline.joblib")
    app.config["lstm_transform_pipeline"] = load_preprocessing_pipeline("LSTMDL_Transorm_Pipleline.pkl")
    app.config["svm_model"] = load_svm_model("SVM_Model.joblib")
    app.config["lstm_model"] = load_lstm_model("LSTMDL_Model.h5")

    app.run()