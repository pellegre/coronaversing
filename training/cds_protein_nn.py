import pandas as pd
import os
import numpy as np

from sklearn.utils import class_weight
from sklearn.metrics import classification_report, confusion_matrix

from keras.models import Sequential
from keras.layers import Dense, Dropout
from keras.utils import to_categorical

import argparse

TRAINING_SET = ".cache_training_set.pkl"
TRAINING_FRACTION = 0.8


def main():
    parser = argparse.ArgumentParser(description="protein classification")
    parser.add_argument("-f", "--filename", default=TRAINING_SET, action="store",
                        type=str, help="filename of the training data frame")
    parser.add_argument("-o", "--output", action="store",
                        type=str, help="directory to store the model", required=True)

    print("[+] protein classification")

    args = parser.parse_args()
    training_set_filename = args.filename
    output_directory = args.output

    print("[+] reading file", training_set_filename)
    print("[+] output directory", output_directory)

    if args.filename is not None:
        training_set_filename = args.filename
    else:
        training_set_filename = TRAINING_SET

    if os.path.isfile(training_set_filename):
        training_df = pd.read_pickle(training_set_filename)
    else:
        raise ValueError("file", training_set_filename, "not found")

    training_df = training_df.fillna(0)

    cls = training_df["type"].unique()
    class_map = {tp: i for tp, i in zip(cls, range(0, len(cls)))}

    # get training data
    t_data = training_df.values
    np.random.shuffle(t_data)

    # set index to define the training set
    idx = int(TRAINING_FRACTION * len(t_data))

    # training set
    x_train = t_data[:idx, :-1]
    y_train = np.vectorize(class_map.get)(t_data[:idx, -1])

    # test set
    x_test = t_data[idx:, :-1]
    y_test = np.vectorize(class_map.get)(t_data[idx:, -1])

    class_weights = class_weight.compute_class_weight("balanced", np.unique(y_train), y_train)

    y_train = to_categorical(y_train)
    y_test = to_categorical(y_test)

    network = Sequential()

    network.add(Dense(64, activation="relu", input_shape=(len(x_train[0]),)))
    network.add(Dense(64, activation="relu"))
    network.add(Dense(64, activation="relu"))
    network.add(Dense(64, activation="relu"))
    network.add(Dense(len(y_train[0]), activation="softmax"))

    # compile network
    network.compile(optimizer="rmsprop", loss="categorical_crossentropy", metrics=["accuracy"])

    # train network
    network.fit(x_train, y_train, epochs=500, batch_size=128, class_weight=class_weights)

    # predict our testing set
    y_pred = network.predict(x_test)

    target_names = cls

    print("Confusion Matrix")
    matrix = confusion_matrix(y_test.argmax(axis=1), y_pred.argmax(axis=1))

    mdf = pd.DataFrame(matrix, columns=cls)
    mdf.index = cls
    print(mdf)

    print("Classification Report")
    print(classification_report(y_test.argmax(axis=1), y_pred.argmax(axis=1), target_names=target_names))

    # serialize model to JSON
    model_json = network.to_json()
    with open(output_directory + "/cds_protein_nn.json", "w") as json_file:
        json_file.write(model_json)

    # serialize weights to HDF5
    network.save_weights(output_directory + "/cds_protein_nn.h5")
    print("Saved model to disk")


main()