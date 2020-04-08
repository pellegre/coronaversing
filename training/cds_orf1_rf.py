import pandas as pd
from sklearn.externals import joblib
import numpy as np
import pickle
import matplotlib.pyplot as plt

import seaborn as sns

from Bio.Seq import Seq
from Bio.SeqRecord import SeqRecord
from Bio import SeqIO

from sklearn.utils import class_weight
from sklearn.metrics import classification_report, confusion_matrix

from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.ensemble import RandomForestClassifier

import argparse

TRAINING_FRACTION = 0.8
CORONA_REGION_FILE = "./.cache_regions.pkl"
CORONA_CDS_ORFS = "./.cache_cds_orfs.pkl"
CORONA_CDS = "./.cache_cds.pkl"
CORONA = "./.cache_vrs.pkl"

CDS_CLASSES = ["ORF1A", "UNDEF"]
MODEL_NAME = "/cds_orf1_rf"

SLIPPERY_THRESHOLD = 0.9


def get_training_set(cds_classes, corona, corona_cds, corona_cds_orfs):
    columns = ["start", "end"]
    orf1ab_frame = {i: dict() for i in columns}

    orf1ab_cds = corona_cds[corona_cds["length"] > 15000]

    for protein_id, row in orf1ab_cds.iterrows():
        sample = corona[corona.index == row["oid"]]
        genome = sample["sequence"].values[0]

        # start / end codon of the gene
        start = row["start"]
        end = row["end"]

        orf1ab_frame["start"][protein_id] = float(start) / float(len(genome))
        orf1ab_frame["end"][protein_id] = float(end) / float(len(genome))

    orf1ab = pd.DataFrame(orf1ab_frame)
    orf1ab.index.name = "id"

    min_orf1ab_end = orf1ab["end"].min()
    max_orf1ab_end = orf1ab["end"].max()

    columns = ["start", "end", "frameshifted", "length", "slippery", "type"]
    slippery_sequence = "TTTAAAC"

    dataset_frame = {i: dict() for i in columns}

    for cls in cds_classes:
        # get CDS group
        df = corona_cds_orfs[cls]

        for index, row in df.iterrows():
            sample = corona[corona.index == row["oid"]]
            genome = sample["sequence"].values[0]

            # start / end codon of the gene
            start = row["start"]
            end = row["end"]

            gene = genome[start:end]

            # slippery sequence relative position
            slippery_when_wet = gene.rfind(slippery_sequence)
            if slippery_when_wet == -1:
                # if it doesn't have a slippery sequence, it's not something we want to tag as ORF1A
                dataset_frame["slippery"][index] = 0.0
                dataset_frame["type"][index] = "UNDEF"
                dataset_frame["frameshifted"][index] = float(end) / float(len(genome))
            else:
                slippery_position = float(slippery_when_wet) / float(row["length"])
                dataset_frame["slippery"][index] = slippery_position

                til_end_gene = genome[:end]
                slippery_when_wet = til_end_gene.rfind(slippery_sequence)

                slippery_when_wet += len(slippery_sequence)
                shifted_gene = genome[start:slippery_when_wet] + genome[slippery_when_wet - 1:len(genome)]
                rfs_length = 3.0 * len(shifted_gene.translate(to_stop=True))

                extended_length = float(rfs_length + start) / float(len(genome))
                dataset_frame["frameshifted"][index] = extended_length

                if (slippery_position > SLIPPERY_THRESHOLD) and \
                        (min_orf1ab_end < extended_length < max_orf1ab_end):
                    dataset_frame["type"][index] = cls
                else:
                    dataset_frame["type"][index] = "UNDEF"

            dataset_frame["length"][index] = row["length"]
            dataset_frame["start"][index] = float(start) / float(len(genome))
            dataset_frame["end"][index] = float(end) / float(len(genome))

    training_df = pd.DataFrame(dataset_frame)
    training_df.index.name = "id"

    return training_df


def print_missed(training_df, corona_cds, loaded_model):
    df = training_df.copy()

    cls = df["type"].unique()
    class_map = {tp: i for tp, i in zip(cls, range(0, len(cls)))}
    categories_map = {v: k for k, v in class_map.items()}

    x = df.values[:, :-1]
    y_pred = loaded_model.predict(x)

    y_pred = np.vectorize(categories_map.get)(y_pred)
    df["predicted"] = y_pred

    misses = df[df["type"].values != df["predicted"].values]

    missed_cds = corona_cds[corona_cds.index.isin(misses.index)][
        ["product", "start", "end", "length", "gene", "oid"]]
    missed_cds["type"] = misses["type"]
    missed_cds["predicted"] = misses["predicted"]

    print(missed_cds)


def main():
    parser = argparse.ArgumentParser(description="CDS ORFs file")
    parser.add_argument("-o", "--output", action="store",
                        type=str, help="directory to store the model", required=True)
    parser.add_argument("-x", "--cross", action="store", default=5,
                        type=str, help="number of folds (k-fold cross validation)")
    parser.add_argument("-l", action="store_true", default=False, help="load classifier")
    parser.add_argument("-g", action="store_true", default=False, help="grid search (optimize e cut-off)")

    print("[+] ORF1 classification")
    args = parser.parse_args()
    output_directory = args.output

    corona_db = pd.read_pickle(CORONA_REGION_FILE)
    corona_cds_orfs = pd.read_pickle(CORONA_CDS_ORFS)
    corona = pd.read_pickle(CORONA)
    corona = corona.set_index("id")

    # load complete CDS file
    with open(CORONA_CDS, "rb") as handle:
        corona_cds = pickle.load(handle)
    corona_cds = corona_cds.set_index("protein_id")

    # at this point we would like to analyze pretty much complete genomes
    corona_incomplete = corona[(corona["unknown"] > 0) | (corona["length"] < 27000)]
    corona_incomplete_cds = corona_cds[(corona_cds["unknown"] > 0)]

    # drop duplicates
    corona = corona[~corona.index.duplicated(keep='first')]
    corona_cds = corona_cds[~corona_cds.index.duplicated(keep='first')]

    corona_db = corona_db[~corona_db["id"].isin(corona_incomplete.index)]
    corona_cds = corona_cds[~corona_cds["oid"].isin(corona_incomplete.index)]
    corona_cds = corona_cds[~corona_cds.index.isin(corona_incomplete_cds.index)]

    tdf = get_training_set(CDS_CLASSES, corona, corona_cds, corona_cds_orfs)

    # get classes
    cls = tdf["type"].unique()
    class_map = {tp: i for tp, i in zip(cls, range(0, len(cls)))}

    average_accuracy = 0.0

    # run for each cross validation case
    if args.g:
        k_fold = args.cross
    else:
        k_fold = 1

    for run in range(0, k_fold):
        print(" =========== Running case", run)
        # get training data
        t_data = np.copy(tdf.values)

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

        # tree = DecisionTreeClassifier()
        tree = RandomForestClassifier()
        tree.fit(x_train, y_train)

        y_pred = tree.predict(x_test)

        target_names = cls
        hits = (y_test == y_pred)

        features = tdf.columns[:-1]
        print("Importances")
        for i, tn in zip(range(0, len(features)), features):
            print("  -", tn, "=", tree.feature_importances_[i])

        accuracy = float(np.count_nonzero(hits)) / float(len(hits))
        average_accuracy += accuracy

        print("Accuracy", accuracy)
        print("Confusion Matrix")
        matrix = confusion_matrix(y_test, y_pred)

        mdf = pd.DataFrame(matrix, columns=cls)
        mdf.index = cls
        print(mdf)

        print("Classification Report")
        print(classification_report(y_test, y_pred, target_names=target_names))

        print_missed(tdf, corona_cds, tree)

        if not args.g:
            # serialize classifier
            joblib.dump(tree, output_directory + MODEL_NAME + ".pkl", compress=9)
            print("Saved model to disk")

    # get average accuracy
    average_accuracy /= float(k_fold)

    print("Average accuracy for case :", average_accuracy)


main()
