from keras.models import Sequential
from keras.layers import Conv2D, Activation
from keras.layers import MaxPooling2D, AveragePooling2D
from keras.layers import Dense
from keras.layers import Flatten
from keras.layers import BatchNormalization
from keras.optimizers import SGD
from metrics import fbeta


def test_model(in_shape=(328, 118, 3), out_shape=9):
    model = Sequential()

    model.add(Conv2D(16, (3, 3), use_bias=False, kernel_initializer='he_uniform', padding='same', input_shape=in_shape))
    model.add(BatchNormalization())
    model.add(Activation('relu'))
    model.add(Conv2D(16, (3, 3), use_bias=False, kernel_initializer='he_uniform', padding='same'))
    model.add(BatchNormalization())
    model.add(Activation('relu'))
    model.add(MaxPooling2D((2, 2)))

    model.add(Flatten())
    model.add(Dense(out_shape, activation='sigmoid'))

    opt = SGD(lr=0.01, momentum=0.9)
    model.compile(optimizer=opt, loss='binary_crossentropy', metrics=[fbeta])

    return model


def vgg19_model(in_shape=(328, 118, 3), out_shape=9):
    model = Sequential()

    model.add(Conv2D(16, (3, 3), use_bias=False, kernel_initializer='he_uniform', padding='same', input_shape=in_shape))
    model.add(BatchNormalization())
    model.add(Activation('relu'))
    model.add(Conv2D(16, (3, 3), use_bias=False, kernel_initializer='he_uniform', padding='same'))
    model.add(BatchNormalization())
    model.add(Activation('relu'))
    model.add(MaxPooling2D((2, 2)))

    model.add(Conv2D(32, (3, 3), use_bias=False, kernel_initializer='he_uniform', padding='same'))
    model.add(BatchNormalization())
    model.add(Activation('relu'))
    model.add(Conv2D(32, (3, 3), use_bias=False, kernel_initializer='he_uniform', padding='same'))
    model.add(BatchNormalization())
    model.add(Activation('relu'))
    model.add(MaxPooling2D((2, 2)))

    model.add(Conv2D(64, (3, 3), use_bias=False, kernel_initializer='he_uniform', padding='same'))
    model.add(BatchNormalization())
    model.add(Activation('relu'))
    model.add(Conv2D(64, (3, 3), use_bias=False, kernel_initializer='he_uniform', padding='same'))
    model.add(BatchNormalization())
    model.add(Activation('relu'))
    model.add(Conv2D(64, (3, 3), use_bias=False, kernel_initializer='he_uniform', padding='same'))
    model.add(BatchNormalization())
    model.add(Activation('relu'))
    model.add(Conv2D(64, (3, 3), use_bias=False, kernel_initializer='he_uniform', padding='same'))
    model.add(BatchNormalization())
    model.add(Activation('relu'))
    model.add(MaxPooling2D((2, 2)))

    model.add(Conv2D(128, (3, 3), use_bias=False, kernel_initializer='he_uniform', padding='same'))
    model.add(BatchNormalization())
    model.add(Activation('relu'))
    model.add(Conv2D(128, (3, 3), use_bias=False, kernel_initializer='he_uniform', padding='same'))
    model.add(BatchNormalization())
    model.add(Activation('relu'))
    model.add(Conv2D(128, (3, 3), use_bias=False, kernel_initializer='he_uniform', padding='same'))
    model.add(BatchNormalization())
    model.add(Activation('relu'))
    model.add(Conv2D(128, (3, 3), use_bias=False, kernel_initializer='he_uniform', padding='same'))
    model.add(BatchNormalization())
    model.add(Activation('relu'))
    model.add(MaxPooling2D((2, 2)))

    model.add(Conv2D(256, (3, 3), use_bias=False, kernel_initializer='he_uniform', padding='same'))
    model.add(BatchNormalization())
    model.add(Activation('relu'))
    model.add(Conv2D(256, (3, 3), use_bias=False, kernel_initializer='he_uniform', padding='same'))
    model.add(BatchNormalization())
    model.add(Activation('relu'))
    model.add(Conv2D(256, (3, 3), use_bias=False, kernel_initializer='he_uniform', padding='same'))
    model.add(BatchNormalization())
    model.add(Activation('relu'))
    model.add(Conv2D(256, (3, 3), use_bias=False, kernel_initializer='he_uniform', padding='same'))
    model.add(BatchNormalization())
    model.add(Activation('relu'))
    model.add(MaxPooling2D((2, 2)))

    model.add(AveragePooling2D())
    model.add(Flatten())

    # Sigmoid used in order to allow multiple labels. Use softmax for
    # classification.
    model.add(Dense(out_shape, activation='sigmoid'))

    opt = SGD(lr=0.01, momentum=0.9)
    model.compile(optimizer=opt, loss='binary_crossentropy', metrics=[fbeta])

    return model
