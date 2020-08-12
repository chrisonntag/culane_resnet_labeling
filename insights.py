import sys
import random
import preprocessing as pp
from numpy import savez_compressed
import matplotlib.pyplot as plt
from matplotlib.image import imread


def plot_examples(file_mapping):
    example_images = []
    choices = [6123, 4010, 122, 910, 1997, 211, 23, 785, 3785] # 422 is the smallest of all categories
    for i in range(0, 9):
        filtered_mapping = [(k, v) for k, v in file_mapping.items() if v[i] == 1]
        if len(filtered_mapping) > 0:
            example_images.append(filtered_mapping[choices[i]])

    print(example_images)

    fig = plt.figure()
    fig.suptitle("Example images")
    for i in range(0, len(example_images)):
        sp = fig.add_subplot(330 + 1 + i)

        filename = FOLDER + example_images[i][0]
        image = imread(filename)
        sp.imshow(image)

        label_class = [index for index in range(0, len(example_images[i][1])) if
         example_images[i][1][index] == 1]
        label_name = pp.inv_labels_map[label_class[0]]
        sp.set_title(label_name)
    # show the figure
    plt.show()

def plot_history(history):
    """
    Plot the loss and accuracy using fbeta from a models history.
    """
    # plot loss
    plt.subplot(211)
    plt.title('Cross Entropy Loss')
    plt.plot(history.history['loss'], color='blue', label='train')
    plt.legend(loc="upper right")
    plt.plot(history.history['val_loss'], color='orange', label='test')
    plt.legend(loc="upper right")

    # plot accuracy
    plt.subplot(212)
    plt.title('Fbeta')
    plt.plot(history.history['fbeta'], color='blue', label='train')
    plt.legend(loc="upper right")
    plt.plot(history.history['val_fbeta'], color='orange', label='test')
    plt.legend(loc="upper right")

    filename = sys.argv[0].split('/')[-1]
    plt.savefig(filename + '_plot.png')
    plt.close()


if __name__ == "__main__":
    mapping = pp.one_hot_encode("culane/")
    plot_examples(mapping)
