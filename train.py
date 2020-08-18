import sys
import os
import pandas as pd
import multiprocessing
from numpy import load
from sklearn.model_selection import train_test_split
from keras.preprocessing.image import ImageDataGenerator
from keras.models import Sequential
from keras.layers import Conv2D
from keras.layers import MaxPooling2D
from keras.layers import Dense
from keras.layers import Flatten
from keras.optimizers import SGD
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


def define_vgg_model(in_shape=(800, 288, 3), out_shape=9):
    model = Sequential()
    model.add(Conv2D(32, (3, 3), activation='relu', kernel_initializer='he_uniform', padding='same', input_shape=in_shape))
    model.add(Conv2D(32, (3, 3), activation='relu', kernel_initializer='he_uniform', padding='same'))
    model.add(MaxPooling2D((2, 2)))
    model.add(Conv2D(64, (3, 3), activation='relu', kernel_initializer='he_uniform', padding='same'))
    model.add(Conv2D(64, (3, 3), activation='relu', kernel_initializer='he_uniform', padding='same'))
    model.add(MaxPooling2D((2, 2)))
    model.add(Conv2D(128, (3, 3), activation='relu', kernel_initializer='he_uniform', padding='same'))
    model.add(Conv2D(128, (3, 3), activation='relu', kernel_initializer='he_uniform', padding='same'))
    model.add(MaxPooling2D((2, 2)))
    model.add(Flatten())

    # Sigmoid used in order to allow multiple labels. Use softmax for
    # classification.
    model.add(Dense(128, activation='relu', kernel_initializer='he_uniform'))
    model.add(Dense(out_shape, activation='sigmoid'))

    opt = SGD(lr=0.01, momentum=0.9)
    model.compile(optimizer=opt, loss='binary_crossentropy', metrics=[fbeta])

    return model


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
    datagen = ImageDataGenerator(rescale=1.0/255.0)

    # prepare iterators
    abs_path = os.path.dirname(os.path.abspath(__file__))
    train_it = datagen.flow_from_dataframe(df_train, directory=abs_path,
                                           x_col='filename', y_col=pp.label_names,
                                           target_size=(800, 288),
                                           batch_size=16, class_mode='raw')
    test_it = datagen.flow_from_dataframe(df_test, directory=abs_path,
                                          x_col='filename', y_col=pp.label_names,
                                          target_size=(800, 288),
                                          batch_size=16, class_mode='raw')

    model = define_vgg_model()
    history = model.fit(train_it, steps_per_epoch=len(train_it),
                        validation_data=test_it, validation_steps=len(test_it),
                        epochs=epochs, verbose=1, use_multiprocessing=True,
                        workers=multiprocessing.cpu_count())
    model.save('culane_model')

    loss, fbeta = model.evaluate(test_it, steps=len(test_it), verbose=1)
    print('> loss=%.3f, fbeta=%.3f' % (loss, fbeta))

    stat.plot_history(history)

