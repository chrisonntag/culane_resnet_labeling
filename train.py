import sys
import os
import pandas as pd
import multiprocessing
import models
from numpy import load
from sklearn.model_selection import train_test_split
from keras.preprocessing.image import ImageDataGenerator

import preprocessing as pp
import insights as stat
from metrics import fbeta


def load_dataset():
    # load dataset
    data = load('culane.npz')
    X, y = data['arr_0'], data['arr_1']

    X_train, X_test, y_train, y_test = train_test_split(X, y,
                                                        test_size=0.15,
                                                        random_state=42)
    print(X_train.shape, y_train.shape, X_test.shape, y_test.shape)

    return X_train, y_train, X_test, y_test


if __name__ == "__main__":
    args = sys.argv
    epochs = 50
    if len(args) > 1:
        try:
            epochs = int(sys.argv[1])
        except ValueError as e:
            # in case unvalid type is given
            print("Set a valid number for epochs")

    print("%d epochs will be trained" % epochs)

    X_train, y_train, X_test, y_test = load_dataset()

    train_data = pp.one_hot_to_categorical(y_train)
    train_data['filename'] = X_train
    test_data = pp.one_hot_to_categorical(y_test)
    test_data['filename'] = X_test

    df_train = pd.DataFrame(data=train_data)
    df_test = pd.DataFrame(data=test_data)

    print(df_train.head())

    # create data generator in order to process images in batches and prevent
    # Memory Errors.
    train_datagen = ImageDataGenerator(rescale=1.0 / 255.0, horizontal_flip=True)
    test_datagen = ImageDataGenerator(rescale=1.0 / 255.0)

    # prepare iterators
    abs_path = os.path.dirname(os.path.abspath(__file__))
    train_it = train_datagen.flow_from_dataframe(df_train, directory=abs_path,
                                                 x_col='filename', y_col=pp.label_names,
                                                 target_size=(800, 288),
                                                 batch_size=256, class_mode='raw')
    test_it = test_datagen.flow_from_dataframe(df_test, directory=abs_path,
                                               x_col='filename', y_col=pp.label_names,
                                               target_size=(800, 288),
                                               batch_size=256, class_mode='raw')

    model = models.vgg19_model()
    history = model.fit(train_it, steps_per_epoch=len(train_it),
                        validation_data=test_it, validation_steps=len(test_it),
                        epochs=epochs, verbose=1, use_multiprocessing=True,
                        workers=multiprocessing.cpu_count())
    model.save('culane_model')

    loss, fbeta = model.evaluate(test_it, steps=len(test_it), verbose=1)
    print('> loss=%.3f, fbeta=%.3f' % (loss, fbeta))

    stat.plot_history(history)
