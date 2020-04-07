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
ORFS_PREDICTION_SET="./.cache_orfs_prediction_set.pkl"
CDS_CLASSES = ["S", "HE", "E", "N", "M", "UNDEF"]

# grid search parameters
CUTOFF = ["-E 1e-50", "-E 1e-30", "-E 1e-20", "-E 1e-10", "-E 1e-5", "-E 1e-3", "-E 1e-2", "--cut_ga"]
HIDDEN_LAYERS = [0, 1, 2]


def get_training_set(domain_matches, corona_cds_training):
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

    for cls in corona_cds_training:

        # get CDS group
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
    df = training_df.copy()

    # load complete CDS file
    cache_cds_file = ".cache_cds.pkl"
    with open(cache_cds_file, "rb") as handle:
        corona_cds = pickle.load(handle)

    corona_cds = corona_cds.set_index("protein_id")

    cls = df["type"].unique()
    class_map = {tp: i for tp, i in zip(cls, range(0, len(cls)))}
    categories_map = {v: k for k, v in class_map.items()}

    x = df.values[:, :-1]
    y_pred = loaded_model.predict(x)

    y_pred = np.vectorize(categories_map.get)(y_pred.argmax(axis=1))
    df["predicted"] = y_pred

    misses = df[df["type"].values != df["predicted"].values]

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
    parser.add_argument("-x", "--cross", action="store", default=5,
                        type=str, help="number of folds (k-fold cross validation)")
    parser.add_argument("-l", action="store_true", default=False, help="load classifier")
    parser.add_argument("-g", action="store_true", default=False, help="grid search (optimize e cut-off)")

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

    # domain database
    pfam_database_file = data_directory + PFAM_DATABASE

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
        if cls == "PRED":
            for index, row in corona_cds_training[cls].iterrows():
                record = SeqRecord(row["protein"], id=index, description=row["id"])
                records.append(record)
        else:
            for index, row in corona_cds_training[cls].iterrows():
                record = SeqRecord(Seq(row["translation"]), id=index, description=row["product"])
                records.append(record)

    cds_fasta_file = data_directory + CDS_FASTA_FILE
    SeqIO.write(records, cds_fasta_file, "fasta")

    print("[+] fasta file written :", cds_fasta_file)

    if args.g:
        cutoff_values = CUTOFF
        hidden_layers = HIDDEN_LAYERS
        k_fold = args.cross
    else:
        # we know this is the optimal values
        cutoff_values = ["-E 1e-2"]
        hidden_layers = [2]
        k_fold = 1

    # run HMMSCAN for all the cases we need
    training_cases = {}
    for cutoff in cutoff_values:
        suffix = str(cutoff).replace(' ', '')
        if args.g:
            scan_file = data_directory + "/matches_cds.scan"
        else:
            scan_file = "/tmp/matches_cds" + suffix + ".scan"

        command = ["hmmscan", "--domtblout", scan_file, cutoff, "--cpu", "64", pfam_database_file, cds_fasta_file]

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

        training_df = get_training_set(domain_matches, corona_cds_training)
        training_df = training_df.fillna(0)

        training_cases[suffix] = training_df[training_df["type"].isin(CDS_CLASSES)]

        if args.g:
            # get ORFs for prediction
            predict_df = training_df["type"][training_df["type"] == "PRED"]
            print("Writing", ORFS_PREDICTION_SET, "file")
            predict_df.to_pickle(ORFS_PREDICTION_SET)

    columns = ["cutoff"]
    for hidden_number in hidden_layers:
        columns += ["hidden-" + str(hidden_number)]

    cross_validation = pd.DataFrame(None, columns=columns)
    cross_validation["cutoff"] = cutoff_values

    # run over each case
    for j, case in zip(range(0, len(training_cases)), training_cases):
        for hidden_number in hidden_layers:

            case_hidden_column = "hidden-" + str(hidden_number)
            average_accuracy = 0.0

            # run for each cross validation case
            for run in range(0, k_fold):
                print(" =========== Running case", case, case_hidden_column, run)

                # get training dataset
                training_df = training_cases[case]

                # get classes
                cls = training_df["type"].unique()
                class_map = {tp: i for tp, i in zip(cls, range(0, len(cls)))}

                # get training data
                t_data = np.copy(training_df.values)

                # shuffle the data
                np.random.shuffle(t_data)

                # set index to define the training set
                if args.g:
                    idx = int(TRAINING_FRACTION * len(t_data))
                else:
                    idx = len(t_data)

                # training set
                x_train = t_data[:idx, :-1]
                y_train = np.vectorize(class_map.get)(t_data[:idx, -1])

                # test set
                if args.g:
                    x_test = t_data[idx:, :-1]
                    y_test = np.vectorize(class_map.get)(t_data[idx:, -1])
                else:
                    print("Running single case with the FULL dataset")
                    x_test = x_train
                    y_test = y_train

                class_weights = class_weight.compute_class_weight("balanced", np.unique(y_train), y_train)

                encoded = to_categorical(y_train)
                y_test = to_categorical(y_test)

                network = Sequential()

                network.add(Dense(32, activation="relu", input_shape=(len(x_train[0]),)))

                for nh in range(0, hidden_number):
                    network.add(Dense(16, activation="relu"))

                network.add(Dense(len(encoded[0]), activation="softmax"))

                # compile network
                network.compile(optimizer="rmsprop", loss="categorical_crossentropy", metrics=["accuracy"])

                # train network
                network.fit(x_train, encoded, epochs=50, batch_size=8, class_weight=class_weights, verbose=0)

                # predict our testing set
                y_pred = network.predict(x_test)

                target_names = cls

                hits = y_test.argmax(axis=1) == y_pred.argmax(axis=1)

                print("Result for case", case, "- hidden layers", hidden_number)
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

                print_missed(training_df, network)

                if not args.g:
                    # serialize model to JSON
                    model_json = network.to_json()
                    with open(output_directory + "/cds_protein_nn.json", "w") as json_file:
                        json_file.write(model_json)

                    # serialize weights to HDF5
                    network.save_weights(output_directory + "/cds_protein_nn.h5")

                    print("Saved model to disk")

            # get average accuracy
            average_accuracy /= float(k_fold)
            print("Average accuracy for case :", case, case_hidden_column)

            # set value on the data frame
            cross_validation.iloc[j][case_hidden_column] = average_accuracy

    # store cross validation results
    print(cross_validation)
    cross_validation.to_pickle(output_directory + "/cds_protein_nn_xval.pkl")


main()
