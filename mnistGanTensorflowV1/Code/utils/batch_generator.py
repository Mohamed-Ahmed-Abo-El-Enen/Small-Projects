import random


def generate_batches(X_dataset, y_dataset, batch_size):
    ind = 0
    list_index = list(range(0, len(X_dataset)))
    random.shuffle(list_index)
    X_dataset = X_dataset[list_index]
    y_dataset = y_dataset[list_index]
    while ind < X_dataset.shape[0]:
        if ind + batch_size > X_dataset.shape[0]:
            X = X_dataset[ind:]
            y = y_dataset[ind:]
        else:
            X = X_dataset[ind:ind + batch_size]
            y = y_dataset[ind:ind + batch_size]
        ind += batch_size
        yield X, y
