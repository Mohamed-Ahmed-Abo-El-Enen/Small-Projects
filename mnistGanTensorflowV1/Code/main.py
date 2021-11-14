from utils.mnist_read import get_mnist_dataset
from utils.models_run import select_train_model, model_predict


if __name__ == "__main__":
    X_train, y_train = get_mnist_dataset()
    model = select_train_model(X_train, y_train)
    model_predict(model)

