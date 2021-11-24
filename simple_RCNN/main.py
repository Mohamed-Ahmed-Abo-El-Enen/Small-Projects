import os
import pandas as pd
import numpy as np
import cv2
import keras
import matplotlib.pyplot as plt
from keras.layers import Dense
from keras import Model
from keras.preprocessing.image import ImageDataGenerator
from keras.applications.vgg16 import VGG16
from keras.optimizers import Adam
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelBinarizer
from keras.callbacks import ModelCheckpoint, EarlyStopping


DATASET_DIRECTORY = "dataset"
path = "Images"
annot = "Airplanes_Annotations"
cv2.setUseOptimized(True)
ss = cv2.ximgproc.segmentation.createSelectiveSearchSegmentation()


def get_iou(bb1, bb2):
    assert bb1["x1"] < bb1["x2"]
    assert bb1["y1"] < bb1["y2"]
    assert bb2["x1"] < bb2["x2"]
    assert bb2["y1"] < bb2["y2"]

    x_left = max(bb1["x1"], bb2["x1"])
    y_top = max(bb1["y1"], bb2["y1"])
    x_right = min(bb1["x2"], bb2["x2"])
    y_bottom = min(bb1["y2"], bb2["y2"])

    if x_right<x_left or y_bottom < y_top:
        return 0.0

    intersection_area = (x_right - x_left) * (y_bottom - y_top)

    bb1_area = (bb1["x2"] - bb1["x1"]) * (bb1["y2"] - bb1["y1"])
    bb2_area = (bb2["x2"] - bb2["x1"]) * (bb2["y2"] - bb2["y1"])

    iou = intersection_area / float(bb1_area+bb2_area-intersection_area)
    assert iou >= 0.0
    assert iou <= 1.0
    return iou


def get_image_label():
    train_images = []
    train_labels = []

    for e, i in enumerate(os.listdir(os.path.join(DATASET_DIRECTORY, annot))):
        if e >= 20:
            break
        file_name = ""
        try:
            if i.startswith("airplane"):
                file_name = i.split('.')[0] + ".jpg"
                print(e, file_name)

                image = cv2.imread(os.path.join(DATASET_DIRECTORY, path, file_name))
                df = pd.read_csv(os.path.join(DATASET_DIRECTORY, annot, i))
                gt_values = []
                for row in df.iterrows():
                    x1 = int(row[1][0].split(' ')[0])
                    y1 = int(row[1][0].split(' ')[1])

                    x2 = int(row[1][0].split(' ')[2])
                    y2 = int(row[1][0].split(' ')[3])

                    gt_values.append({"x1": x1,
                                      "y1": y1,
                                      "x2": x2,
                                      "y2": y2})

                    ss.setBaseImage(image)
                    ss.switchToSelectiveSearchFast()
                    ss_results = ss.process()
                    im_out = image.copy()
                    counter = 0
                    false_counter = 0
                    flag = 0
                    f_flag = 0
                    b_flag = 0

                    for e, result in enumerate(ss_results):
                        if e < 2000 and flag == 0:
                            for gt_val in gt_values:
                                x, y, w, h = result
                                iou = get_iou(gt_val,
                                              {"x1": x,
                                               "y1": y,
                                               "x2": x + w,
                                               "y2": y + h})
                                if counter < 30:
                                    if iou > 0.7:
                                        t_image = im_out[y:y + h, x:x + w]
                                        resized = cv2.resize(t_image, (224, 224), interpolation=cv2.INTER_AREA)

                                        train_images.append(resized)
                                        train_labels.append(1)
                                        counter += 1
                                else:
                                    f_flag = 1

                                if false_counter < 30:
                                    if iou < 0.3:
                                        t_image = im_out[y:y + h, x:x + w]
                                        resized = cv2.resize(t_image, (224, 224), interpolation=cv2.INTER_AREA)
                                        train_images.append(resized)
                                        train_labels.append(0)
                                        false_counter += 1
                                else:
                                    b_flag = 1

                            if f_flag == 1 and b_flag == 1:
                                print("inside")
                                flag = 1
        except Exception as e:
            print(e)
            print("Error in " + file_name)
            continue

    return np.array(train_images), np.array(train_labels)


def create_model(optimizer=Adam()):
    vgg_model = VGG16(weights="imagenet", include_top=True)
    vgg_model.summary()

    for layers in vgg_model.layers[:15]:
        print(layers)
        layers.trainable = False

    X = vgg_model.layers[-2].output
    predictions = Dense(2, activation="softmax")(X)
    model_final = Model(inputs=vgg_model.input, outputs=predictions)

    model_final.compile(loss=keras.losses.categorical_crossentropy, optimizer=optimizer, metrics=["accuracy"])

    return model_final


class CustomLabelBinarizer(LabelBinarizer):
    def transform(self, y):
        Y = super().transform(y)
        if self.y_type_ == "binary":
            return np.hstack((Y, 1-Y))
        else:
            return Y

    def inverse_transform(self, Y, threshold=None):
        if self.y_type_ == "binary":
            return super().inverse_transform(Y[:, 0], threshold)
        else:
            return super().inverse_transform(Y, threshold)


def train_model(model, train_gen, val_gen, steps_per_epoch, epochs, validation_steps):
    checkpoint = ModelCheckpoint("rcnn_vgg16.h5",
                                 monitor='val_loss',
                                 verbose=1,
                                 save_best_only=True,
                                 save_weights_only=False,
                                 mode='auto',
                                 period=1)
    early = EarlyStopping(monitor='val_loss',
                          min_delta=0,
                          patience=100,
                          verbose=1,
                          mode='auto')
    history = model.fit_generator(generator=train_gen,
                                  steps_per_epoch=steps_per_epoch,
                                  epochs=epochs,
                                  validation_data=val_gen,
                                  validation_steps=validation_steps,
                                  callbacks=[checkpoint, early],
                                  verbose=1)
    return history


def plot_model_history(history):
    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.title("model loss")
    plt.ylabel("Loss")
    plt.xlabel("Epoch")
    plt.legend(["Loss", "Validation Loss"])
    plt.show()


def predict_model_bounding_box(model):
    z = 0
    for e, i in enumerate(os.listdir(path)):
        if i.startswith("4"):
            z += 1
            img = cv2.imread(os.path.join(path, i))
            ss.setBaseImage(img)
            ss.switchToSelectiveSearchFast()
            ss_results = ss.process()
            im_out = img.copy()
            for e, result in enumerate(ss_results):
                if e < 2000:
                    x, y, w, h = result
                    t_image = im_out[y:y + h, x:x + w]
                    resized = cv2.resize(t_image, (224, 224), interpolation=cv2.INTER_AREA)
                    img = np.expand_dims(resized, axis=0)
                    out = model.predict(img)
                    if out[0][0] > 0.70:
                        cv2.rectangle(im_out, (x, y), (x + w, y + h), (0, 255, 0), 1, cv2.LINE_AA)
            plt.figure()
            plt.imshow(im_out)
            break


if __name__ == "__main__":
    X_new, y_new = get_image_label()
    print(X_new.shape)
    model = create_model(Adam(0.001))
    label_bin = CustomLabelBinarizer()
    y_new = label_bin.fit_transform(y_new)
    X_train, X_test, y_train, y_test = train_test_split(X_new, y_new, test_size=0.10)
    print(X_train.shape, y_train.shape, X_test.shape, y_test.shape)

    _train_gen = ImageDataGenerator(horizontal_flip=True,
                                    vertical_flip=True,
                                    rotation_range=90)
    train_gen = _train_gen.flow(x=X_train, y=y_train)

    _val_gen = ImageDataGenerator(horizontal_flip=True,
                                  vertical_flip=True,
                                  rotation_range=90)
    val_gen = _val_gen.flow(x=X_test, y=y_test)
    steps_per_epoch = 10
    epochs = 10
    validation_steps = 2
    history = train_model(model, train_gen, val_gen, steps_per_epoch, epochs, validation_steps)
    plot_model_history(history)

    predict_model_bounding_box(model)




















