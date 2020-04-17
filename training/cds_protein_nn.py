import pandas as pd
import subprocess
import pickle
import numpy as np
import os
import json

from sklearn.ensemble import RandomForestClassifier
from sklearn.utils import class_weight
from sklearn.metrics import classification_report, confusion_matrix

from keras.models import Sequential
from keras.layers import Dense, Dropout
from keras.utils import to_categorical
from keras.models import model_from_json

from Bio.Seq import Seq
from Bio.SeqRecord import SeqRecord
from Bio import SeqIO

import argparse

TRAINING_FRACTION = 0.8
CDS_FASTA_FILE = "/tmp/corona_cds.faa"
PFAM_DATABASE = "/Pfam-A.CoV.hmm"

CORONA_REGION_FILE = "./.cache_corona_tagged_db.pkl"
CORONA_CDS_ORFS = "./.cache_cds_orfs.pkl"
CORONA_CDS = "./.cache_cds.pkl"
CORONA = "./.cache_vrs.pkl"

CDS_CLASSES = ["S", "HE", "E", "N", "M", "UNDEF"]

# grid search parameters
CUTOFF = [1e-1, 1e-2, 1e-3, 1e-4, 1e-5, 1e-10, 1e-20]

RF_ESTIMATORS = [60, 80, 100, 120, 140]

NN_HIDDEN_LAYERS = [0, 1, 2, 3]
NN_DROPOUT = [0.1, 0.2, 0.3]
NN_NODES = [8, 16, 32, 64, 128]


class Predictor:
    def __init__(self, name):
        self.name = name

    def fit(self, x_train, y_train):
        pass

    def predict(self, x_pred):
        pass


class RandomForest(Predictor):
    def __init__(self, n_estimators):
        name = "rf-" + str(n_estimators)
        super(RandomForest, self).__init__(name)

        self.n_estimators = n_estimators
        self.forest = None

    def fit(self, x_train, y_train):
        forest = RandomForestClassifier(n_estimators=self.n_estimators, class_weight="balanced", n_jobs=-1)
        forest.fit(x_train, y_train)
        self.forest = forest

    def predict(self, x_pred):
        return self.forest.predict_proba(x_pred)


class NeuralNetwork(Predictor):
    def __init__(self, n_layers, n_nodes, dropout):
        name = "nn-layers" + str(n_layers) + "-nodes" + str(n_nodes)
        super(NeuralNetwork, self).__init__(name)

        self.n_layers = n_layers
        self.n_nodes = n_nodes
        self.dropout = dropout

        self.network = None

    def fit(self, x_train, y_train):
        y_categorical = to_categorical(y_train)
        class_weights = class_weight.compute_class_weight("balanced", np.unique(y_train), y_train)

        network = Sequential()
        network.add(Dense(self.n_nodes, activation="relu", input_shape=(len(x_train[0]),)))

        dropout = self.dropout
        for ly in range(0, self.n_layers):
            network.add(Dropout(dropout))
            network.add(Dense(int(self.n_nodes / 2), activation="relu"))
            dropout /= 2

        network.add(Dense(len(y_categorical[0]), activation="softmax"))

        # compile network
        network.compile(optimizer="rmsprop", loss="categorical_crossentropy", metrics=["accuracy"])

        # train network
        network.fit(x_train, y_categorical, epochs=50, batch_size=8, class_weight=class_weights, verbose=0)

        self.network = network

    def predict(self, x_pred):
        return self.network.predict(x_pred)


def get_training_set(cds_classes, domain_matches, corona_cds_training):
    columns = ["length"]

    domain_count = domain_matches.groupby(["domain", "id"]).size().reset_index(name="count")
    domain_count = domain_count.groupby("domain").max()["count"].reset_index(name="count")
    domain_count = domain_count.set_index("domain").to_dict()

    for domain in domain_count["count"]:
        count = domain_count["count"][domain]
        for i in range(1, count + 1):
            columns += [domain + "_from_" + str(i), domain + "_to_" + str(i), domain + "_score_" + str(i)]

    columns += ["type"]

    dataset_frame = {i: dict() for i in columns}

    for cls in cds_classes:
        # get CDS group
        df = corona_cds_training[cls]

        for index, row in df.iterrows():
            length = float(row["length"]) / 3.0
            dataset_frame["length"][index] = length

            count = {}
            pfam = domain_matches[domain_matches["id"] == index][["domain", "score", "from", "to"]]
            pfam = pfam.set_index("domain")
            for i, r in pfam.iterrows():

                # increment counter
                if i not in count:
                    count[i] = 1
                else:
                    count[i] += 1

                dataset_frame[i + "_score_" + str(count[i])][index] = r["score"]
                dataset_frame[i + "_from_" + str(count[i])][index] = float(r["from"]) / length
                dataset_frame[i + "_to_" + str(count[i])][index] = float(r["to"]) / length

            dataset_frame["type"][index] = cls

    training_df = pd.DataFrame(dataset_frame)
    training_df.index.name = "id"

    return training_df


def print_missed(training_df, corona_cds, classifier):
    df = training_df.copy()

    cls = df["type"].unique()
    class_map = {tp: i for tp, i in zip(cls, range(0, len(cls)))}
    categories_map = {v: k for k, v in class_map.items()}

    x = df.values[:, :-1]
    y_pred = classifier.predict(x)

    for cls in class_map:
        df[cls] = y_pred[:, class_map[cls]]

    y_pred = np.vectorize(categories_map.get)(y_pred.argmax(axis=1))
    df["predicted"] = y_pred

    misses = df[df["type"].values != df["predicted"].values]

    missed_cds = corona_cds[corona_cds.index.isin(misses.index)][
        ["product", "start", "end", "length", "gene", "oid"]]
    missed_cds["type"] = misses["type"]
    missed_cds["predicted"] = misses["predicted"]

    if len(missed_cds) > 0:
        print(missed_cds)


def get_domain_matches(scan_file, ecutoff):
    columns = ["id", "domain", "accession", "score", "from", "to"]
    domains_frame = {i: list() for i in columns}

    with open(scan_file) as matches_file:
        for line in matches_file:
            row = line[:-1]
            if row != "#":
                toks = row.split()

                # domain
                domain = toks[0]

                # accession number
                acc = toks[1].split(".")[0]

                if "PF" in acc:
                    total = int(toks[10])
                    ievalue = float(toks[12])

                    if ievalue > ecutoff:
                        continue

                    domains_frame["accession"].append(acc)
                    domains_frame["domain"].append(domain)

                    # protein id
                    domains_frame["id"].append(toks[3])

                    # score
                    domains_frame["score"].append(float(toks[7]))

                    domains_frame["from"].append(int(toks[17]))
                    domains_frame["to"].append(int(toks[18]))

    domain_matches = pd.DataFrame.from_dict(domains_frame)

    return domain_matches


def main():
    parser = argparse.ArgumentParser(description="CDS ORFs file")
    parser.add_argument("-f", "--filename", default="./.cache_cds_orfs.pkl", action="store",
                        type=str, help="filename of the training data")
    parser.add_argument("-o", "--output", action="store",
                        type=str, help="directory to store the model", required=True)
    parser.add_argument("-d", "--data", action="store",
                        type=str, help="directory with Pfam data", required=True)
    parser.add_argument("-x", "--cross", action="store", default=5,
                        type=str, help="number of folds (k-fold cross validation)")

    print("[+] protein classification")
    args = parser.parse_args()

    training_set_filename = args.filename
    output_directory = args.output
    data_directory = args.data

    print("[+] reading file", training_set_filename)
    print("[+] output directory", output_directory)
    print("[+] data directory", data_directory)

    corona_db = pd.read_pickle(CORONA_REGION_FILE)
    corona_cds_orfs = pd.read_pickle(CORONA_CDS_ORFS)

    lst = set(corona_cds_orfs["S"]["oid"]) & set(corona_cds_orfs["E"]["oid"]) & \
          set(corona_cds_orfs["M"]["oid"]) & set(corona_cds_orfs["N"]["oid"])

    for cds in CDS_CLASSES:
        corona_cds_orfs[cds] = corona_cds_orfs[cds][corona_cds_orfs[cds]["oid"].isin(lst)]

    # load complete CDS file
    with open(CORONA_CDS, "rb") as handle:
        corona_cds = pickle.load(handle)
    corona_cds = corona_cds.set_index("protein_id")

    k_fold = int(args.cross)

    scan_file = data_directory + "/matches_cds.scan"

    classifiers = []
    for n in RF_ESTIMATORS:
        classifiers.append(RandomForest(n_estimators=n))

    for layers in NN_HIDDEN_LAYERS:
        for dropout in NN_DROPOUT:
            for nodes in NN_NODES:
                classifiers.append(NeuralNetwork(n_layers=layers, n_nodes=nodes, dropout=dropout))

    cross_validation = {classifier.name: {} for classifier in classifiers}

    for cutoff in CUTOFF:
        cutoff_value = str(cutoff)

        domain_matches = get_domain_matches(scan_file, cutoff)

        training_df = get_training_set(CDS_CLASSES, domain_matches, corona_cds_orfs)
        training_df = training_df.fillna(0)

        for classifier in classifiers:
            classifier_name = classifier.name

            average_accuracy = 0.0

            # run for each cross validation case
            for run in range(0, k_fold):
                print("[===] Running case", run, "for", classifier_name, "with cutoff", cutoff)

                # get classes
                cls = training_df["type"].unique()
                class_map = {tp: i for tp, i in zip(cls, range(0, len(cls)))}

                # get training data
                t_data = np.copy(training_df.values)

                # shuffle the data
                np.random.shuffle(t_data)

                # set index to define the training set
                idx = int(TRAINING_FRACTION * len(t_data))

                # training set
                x_train = t_data[:idx, :-1]
                y_train = np.vectorize(class_map.get)(t_data[:idx, -1])

                # test set
                x_test = t_data[idx:, :-1]
                y_test = to_categorical(np.vectorize(class_map.get)(t_data[idx:, -1]))

                # train network
                classifier.fit(x_train, y_train)

                # predict our testing set
                y_pred = classifier.predict(x_test)

                target_names = cls

                hits = y_test.argmax(axis=1) == y_pred.argmax(axis=1)

                print("Result for case - estimators", n)
                accuracy = float(np.count_nonzero(hits)) / float(len(hits))
                print("Accuracy", accuracy)
                average_accuracy += accuracy
                print("Confusion Matrix")
                matrix = confusion_matrix(y_test.argmax(axis=1), y_pred.argmax(axis=1))

                mdf = pd.DataFrame(matrix, columns=cls)
                mdf.index = cls
                print(mdf)

                print("Classification Report")
                print(classification_report(y_test.argmax(axis=1), y_pred.argmax(axis=1), target_names=target_names))

                print_missed(training_df, corona_cds, classifier)

            # get average accuracy
            average_accuracy /= float(k_fold)
            print("Average accuracy for ", run, "for", classifier_name, "with cutoff", cutoff, ":", average_accuracy)

            cross_validation[classifier_name][cutoff_value] = format(100 * average_accuracy, '.2f')

    # store cross validation results
    cross_validation = pd.DataFrame(cross_validation)
    cross_validation.index.name = "cutoff"

    print(cross_validation)
    cross_validation.to_pickle(output_directory + "/cds_protein_nn_xval.pkl")


main()
