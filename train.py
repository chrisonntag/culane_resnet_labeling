import sys
import os
import pandas as pd
import tensorflow as tf


os.environ['TF_FORCE_GPU_ALLOW_GROWTH'] = "true"
os.environ["CUDA_VISIBLE_DEVICES"] = '0' # Set to -1 if CPU should be used CPU = -1 , GPU = 0

gpus = tf.config.experimental.list_physical_devices('GPU')
cpus = tf.config.experimental.list_physical_devices('CPU')

if gpus:
    try:
        # Currently, memory growth needs to be the same across GPUs
        for gpu in gpus:
            # Allow the GPU to grow memory dynamically and not allocate at the beginning
            tf.config.experimental.set_memory_growth(gpu, True)
        logical_gpus = tf.config.experimental.list_logical_devices('GPU')
        print(len(gpus), "Physical GPUs,", len(logical_gpus), "Logical GPUs")
    except RuntimeError as e:
        # Memory growth must be set before GPUs have been initialized
        print(e)
elif cpus:
    try:
        # Currently, memory growth needs to be the same across GPUs
        logical_cpus= tf.config.experimental.list_logical_devices('CPU')
        print(len(cpus), "Physical CPU,", len(logical_cpus), "Logical CPU")
    except RuntimeError as e:
        # Memory growth must be set before GPUs have been initialized
        print(e)


import multiprocessing
import models
import configparser
from numpy import load
from sklearn.model_selection import train_test_split
from keras.preprocessing.image import ImageDataGenerator
import preprocessing as pp
import insights as stat
from metrics import fbeta
from dl_bot import DLBot
from telegram_bot_callback import TelegramBotCallback


BASE_DIR = os.path.dirname(os.path.abspath(__file__))

config = configparser.RawConfigParser()
config.read(os.path.join(BASE_DIR, 'config.cnf'))

telegram_token = config.get('telegram', 'token')
telegram_user_id = config.getint('telegram', 'user')

# bot = DLBot(token=telegram_token, user_id=telegram_user_id)
#telegram_callback = TelegramBotCallback(bot)


def load_dataset():
    # load dataset
    data = load('culane.npz')
    X, y = data['arr_0'], data['arr_1']

    X_train, X_test, y_train, y_test = train_test_split(X, y,
                                                        test_size=0.2,
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
                                                 target_size=(328, 118),
                                                 batch_size=4, class_mode='raw')
    test_it = test_datagen.flow_from_dataframe(df_test, directory=abs_path,
                                               x_col='filename', y_col=pp.label_names,
                                               target_size=(328, 118),
                                               batch_size=4, class_mode='raw')

    model = models.test_model()
    history = model.fit(train_it, steps_per_epoch=len(train_it),
                        validation_data=test_it, validation_steps=len(test_it),
                        epochs=epochs, verbose=1)
    model.save('culane_model')

    loss, fbeta = model.evaluate(test_it, steps=len(test_it), verbose=1)
    print('> loss=%.3f, fbeta=%.3f' % (loss, fbeta))

    stat.plot_history(history)
