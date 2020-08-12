import sys
from numpy import load
from numpy import ones
from numpy import asarray
from matplotlib import pyplot
from sklearn.model_selection import train_test_split
from sklearn.metrics import fbeta_score
from keras import backend
from keras.preprocessing.image import ImageDataGenerator
from keras.models import Sequential
from keras.layers import Conv2D
from keras.layers import MaxPooling2D
from keras.layers import Dense
from keras.layers import Flatten
from keras.optimizers import SGD
import insights as stat


def load_dataset():
    # load dataset
    data = load('culane.npz')
    X, y = data['arr_0'], data['arr_1']

    trainX, testX, trainY, testY = train_test_split(X, y, test_size=0.15, random_state=1)
    print(trainX.shape, trainY.shape, testX.shape, testY.shape)

    return trainX, trainY, testX, testY

# Keras version based on 
# https://github.com/keras-team/keras/blob/4fa7e5d454dd4f3f33f1d756a2a8659f2e789141/keras/metrics.py#L134
# https://www.kaggle.com/arsenyinfo/f-beta-score-for-keras
def fbeta(y_true, y_pred, beta=2):
    # clip predictions
    y_pred = backend.clip(y_pred, 0, 1)
    # calculate elements
    tp = backend.sum(backend.round(backend.clip(y_true * y_pred, 0, 1)), axis=1)
    fp = backend.sum(backend.round(backend.clip(y_pred - y_true, 0, 1)), axis=1)
    fn = backend.sum(backend.round(backend.clip(y_true - y_pred, 0, 1)), axis=1)
    # calculate precision
    p = tp / (tp + fp + backend.epsilon())
    # calculate recall
    r = tp / (tp + fn + backend.epsilon())
    # calculate fbeta, averaged across each class
    bb = beta ** 2
    fbeta_score = backend.mean((1 + bb) * (p * r) / (bb * p + r + backend.epsilon()))
    return fbeta_score

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
    if (len(args) > 1):
        try:
            epochs = int(sys.argv[1])
        except ValueError as e:
            # in case unvalid type is given
            print("Set a valid number for epochs")

    print("%d epochs will be trained" % epochs)

    trainX, trainY, testX, testY = load_dataset()

    # create data generator
    # TODO: describe use of
    datagen = ImageDataGenerator(rescale=1.0/255.0)

    # prepare iterators
    train_it = datagen.flow(trainX, trainY, batch_size=16)
    test_it = datagen.flow(testX, testY, batch_size=16)

    model = define_vgg_model()
    history = model.fit(train_it, steps_per_epoch=len(train_it),
                        validation_data=test_it, validation_steps=len(test_it),
                        epochs=epochs, verbose=1)
    model.save('culane_model')

    loss, fbeta = model.evaluate_generator(test_it, steps=len(test_it), verbose=1)
    print('> loss=%.3f, fbeta=%.3f' % (loss, fbeta))

    stat.plot_history(history)

