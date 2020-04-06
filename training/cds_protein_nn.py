import pandas as pd
import subprocess
import pickle
import numpy as np
import os

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
CDS_FASTA_FILE = "/corona_cds.faa"
PFAM_DATABASE = "/Pfam-A.CoV.hmm"
CDS_CLASSES = ["S", "HE", "E", "N", "M", "UNDEF"]


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

        df = corona_cds_training[cls]

        for index, row in df.iterrows():
            dataset_frame["length"][index] = row["length"]

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
                dataset_frame[i + "_from_" + str(count[i])][index] = float(r["from"]) / float(row["length"])
                dataset_frame[i + "_to_" + str(count[i])][index] = float(r["to"]) / float(row["length"])

            dataset_frame["type"][index] = cls

    training_df = pd.DataFrame(dataset_frame)
    training_df.index.name = "id"

    return training_df


def print_missed(training_df, loaded_model):
    # load complete CDS file
    cache_cds_file = ".cache_cds.pkl"
    with open(cache_cds_file, "rb") as handle:
        corona_cds = pickle.load(handle)

    corona_cds = corona_cds.set_index("protein_id")

    cls = training_df["type"].unique()
    class_map = {tp: i for tp, i in zip(cls, range(0, len(cls)))}
    categories_map = {v: k for k, v in class_map.items()}

    x = training_df.values[:, :-1]
    y_pred = loaded_model.predict(x)

    y_pred = np.vectorize(categories_map.get)(y_pred.argmax(axis=1))
    training_df["predicted"] = y_pred

    misses = training_df[training_df["type"].values != training_df["predicted"].values]

    missed_cds = corona_cds[corona_cds.index.isin(misses.index)][
        ["product", "start", "end", "length", "gene", "oid"]]
    missed_cds["type"] = misses["type"]
    missed_cds["predicted"] = misses["predicted"]

    print(missed_cds)


def main():
    parser = argparse.ArgumentParser(description="CDS ORFs file")
    parser.add_argument("-f", "--filename", default="./.cache_cds_orfs.pkl", action="store",
                        type=str, help="filename of the training data")
    parser.add_argument("-o", "--output", action="store",
                        type=str, help="directory to store the model", required=True)
    parser.add_argument("-d", "--data", action="store",
                        type=str, help="directory with Pfam data", required=True)
    parser.add_argument("-l", action="store_true", default=False, help="load classifier")

    print("[+] protein classification")
    args = parser.parse_args()

    training_set_filename = args.filename
    output_directory = args.output
    data_directory = args.data

    print("[+] reading file", training_set_filename)
    print("[+] output directory", output_directory)
    print("[+] data directory", data_directory)

    # load training set
    with open(training_set_filename, "rb") as handle:
        corona_cds_training = pickle.load(handle)

    # load classifier and predict
    if args.l:
        training_df = pd.read_pickle(".cache_cds_training_set.pkl")
        training_df = training_df.fillna(0)

        # load json and create model
        json_file = open(output_directory + "/cds_protein_nn.json", "r")
        loaded_model_json = json_file.read()
        json_file.close()
        loaded_model = model_from_json(loaded_model_json)
        # load weights into new model
        loaded_model.load_weights(output_directory + "/cds_protein_nn.h5")
        print("Loaded model from disk")

        print_missed(training_df, loaded_model)

        return

    records = []
    for cls in corona_cds_training:
        for index, row in corona_cds_training[cls].iterrows():
            record = SeqRecord(Seq(row["translation"]), id=index, description=row["product"])
            records.append(record)

    cds_fasta_file = data_directory + CDS_FASTA_FILE
    SeqIO.write(records, cds_fasta_file, "fasta")

    print("[+] fasta file written :", cds_fasta_file)

    pfam_database_file = data_directory + PFAM_DATABASE
    scan_file = data_directory + "/matches_cds.scan"
    cutoff = "--cut_ga"
    cutoff = "-E 0.001"
    command = ["hmmscan", "--domtblout", scan_file, cutoff, "--cpu", "8", pfam_database_file, cds_fasta_file]

    with open(os.devnull, "w") as f:
        subprocess.call(command, stdout=f)

    print("[+] finish running hmmscan")

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
                    domains_frame["accession"].append(acc)
                    domains_frame["domain"].append(domain)

                    # protein id
                    domains_frame["id"].append(toks[3])

                    # score
                    domains_frame["score"].append(float(toks[7]))

                    # from
                    domains_frame["from"].append(int(toks[17]))
                    domains_frame["to"].append(int(toks[18]))

    domain_matches = pd.DataFrame.from_dict(domains_frame)

    training_df = get_training_set(CDS_CLASSES, domain_matches, corona_cds_training)
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

    network.add(Dense(32, activation="relu", input_shape=(len(x_train[0]),)))
    network.add(Dense(16, activation="relu"))
    network.add(Dense(16, activation="relu"))
    network.add(Dense(len(y_train[0]), activation="softmax"))

    # compile network
    network.compile(optimizer="rmsprop", loss="categorical_crossentropy", metrics=["accuracy"])

    # train network
    network.fit(x_train, y_train, epochs=50, batch_size=8, class_weight=class_weights)

    # predict our testing set
    y_pred = network.predict(x_test)

    target_names = cls

    hits = y_test.argmax(axis=1) == y_pred.argmax(axis=1)
    print("Accuracy :", float(np.count_nonzero(hits)) / float(len(hits)))

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

    # store confusion matrix
    mdf.to_pickle(output_directory + "/cds_protein_nn_matrix.pkl")

    print("Saved model to disk")

    print_missed(training_df, network)


main()
