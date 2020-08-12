import os
import os.path
from os import listdir
from numpy import zeros
from numpy import asarray
from numpy import savez_compressed
from keras.preprocessing.image import load_img
from keras.preprocessing.image import img_to_array


label_names = ["test0_normal", "test1_crowd", "test2_hlight", "test3_shadow",
             "test4_noline", "test5_arrow", "test6_curve", "test7_cross",
             "test8_night"]

labels_map = {label_names[i]:i for i in range(len(label_names))}
inv_labels_map = {i:label_names[i] for i in range(len(label_names))}


# one-hot encoding the images in a dictionary
def one_hot_encode(directory):
    labels = dict()
    for category in label_names:
        with open(directory + "list/test_split/%s.txt" % category, "r") as f:
            count = 1
            cat = int(category[4])
            for img in f:
                count += 1

                if len(img) > 5:
                    end = img[-1:] # Remove newline char
                    if end == '\n':
                        img_name = img[:-1]
                    img_name.strip()
                else:
                    continue

                if img_name in labels.keys():
                    labels[img_name][cat] = 1
                    print("It happened with %s in %s" % (img_name, category))
                else:
                    labels[img_name] = zeros(len(label_names), dtype='uint8')
                    labels[img_name][cat] = 1
    return labels

# load all images into memory
def load_dataset(directory, file_mapping):
    photos, targets = list(), list()

    # Search for *.jpg files in the directory and get mapping
    for dirpath, dirnames, filenames in os.walk(directory):
        for filename in [f for f in filenames if f.endswith(".jpg")]:
            path = os.path.join(dirpath, filename)

            # Scale image down to 800x288
            photo = load_img(path, target_size=(800,288))
            photo = img_to_array(photo, dtype='uint8')

            # Remove directory prefix
            path = path[len(directory):]

            if path in file_mapping.keys():
                photos.append(photo)
                targets.append(file_mapping[path])

    X = asarray(photos, dtype='uint8')
    y = asarray(targets, dtype='uint8')
    return X, y

def prepare_img(path):
    img = load_img(path, target_size=(800,288))
    img = img_to_array(img, dtype='uint8')
    img = img.reshape(1, 800, 288, 3)

    return img

def pred_tags(y):
    vals = y.round()
    return [inv_labels_map[i] for i in range(len(vals)) if vals[i] == 1.0]

if __name__ == "__main__":
    directory = "culane/"
    mapping = one_hot_encode(directory)
    X, y = load_dataset(directory, mapping)
    print(X.shape, y.shape)

    # Save X, y into one compresses file
    savez_compressed('culane.npz', X, y)

