import argparse
import json
from collections import defaultdict
from typing import Sequence

import numpy
import numpy as np
from sklearn.metrics import roc_curve, auc
from tqdm import tqdm

from leaderboard import SUPPORTED_METRICS


def _get_CMs(i, probabilities, references, thresholds):
    confusion_matrices = []
    for threshold in thresholds[i]:
        TP = 0
        FP = 0
        TN = 0
        FN = 0
        for j in range(len(probabilities)):
            if probabilities[j][i] >= threshold:
                if references[j] == i:
                    TP += 1
                else:
                    FP += 1
            else:
                if references[j] == i:
                    FN += 1
                else:
                    TN += 1
        cm = {"TP": TP, "FP": FP, "TN": TN, "FN": FN, "threshold": threshold, "class": i}
        confusion_matrices.append(cm)

    return confusion_matrices


def compute_significance_avg_mcauroc(probsA: Sequence[Sequence[float]], referencesA: Sequence[int],
                                     probsB: Sequence[Sequence[float]], referencesB: Sequence[int]):
    # compute MC-AUC for model A
    model_A_scores = get_mc_auc_samples(probsA, referencesA, Nsamples=1_000)
    model_B_scores = get_mc_auc_samples(probsB, referencesB, Nsamples=1_000)

    # one-tailed test
    p_value = ((model_A_scores[:, np.newaxis] <= model_B_scores[np.newaxis, :]).sum()
               / (len(model_A_scores) * len(model_B_scores)))

    delta = np.mean(model_A_scores) - np.mean(model_B_scores)
    return p_value, delta


def get_mc_auc_samples(probs, references, Nsamples=1_000_000):
    n_classes = list(range(len(probs[0])))
    fpr = dict()
    thresholds = dict()
    # compute AUC for every class
    auc_scores_per_class = []
    for i in range(len(n_classes)):
        # for i-th class vs all others
        fpr[i], _, thresholds[i] = roc_curve(y_true=[1 if x == n_classes[i] else 0 for x in references],
                                             y_score=[prob[i] for prob in probs])

        confusion_matrices = _get_CMs(i, probs, references, thresholds)

        λ = 1.0  # <- Flat prior
        # λ = 0.5  # <- Jeffrey's prior

        # sample variates for every threshold
        tpr_variates_for_each_fpr = []
        for k in range(len(thresholds[i])):
            tpr_variates_for_each_fpr.append(
                numpy.random.beta(confusion_matrices[k]["TP"] + λ, confusion_matrices[k]["FN"] + λ, Nsamples))

        # fprs x tpr_variates
        tpr_variates_for_each_fpr = np.array(tpr_variates_for_each_fpr)

        # now pick 1 variate for each fpr, and compute AUC
        auc_scores = []
        for tpr_variates in tqdm(tpr_variates_for_each_fpr.T,
                                 desc=f"Computing AUCs for class {i + 1}/{len(n_classes)}"):
            auc_score = auc(fpr[i], tpr_variates)
            # if numpy.isnan(auc_score):
            #     auc_score = 0
            auc_scores.append(auc_score)
        auc_scores_per_class.append(auc_scores)

    auc_scores_per_class = np.array(auc_scores_per_class)
    mcauc_scores = np.mean(auc_scores_per_class, axis=0)
    return mcauc_scores


def read_json(file_path):
    data = defaultdict(list)
    with open(file_path, "r") as f:
        fc = json.load(f)
    for task, results in fc["predictions"].items():
        # determine the metric
        metric = None
        for key in SUPPORTED_METRICS:
            if key in results[0]:
                metric = key
                break
        if metric is None:
            raise ValueError(f"Unsupported metric in {file_path}")

        if metric == "avg_mcauroc":
            local_data = [line[metric] for line in fc["predictions"][task]]
            unzipped_list = list(zip(*local_data))
            golds = unzipped_list[0]
            probs = unzipped_list[1]
            data[task] = (golds, probs), metric
    return data, fc["metadata"]


def check_significance(fileA, fileB, significance_level):
    dataA, metadataA = read_json(fileA)
    dataB, metadataB = read_json(fileB)

    decisions = dict()
    for task in dataA.keys():
        metricA = dataA[task][1]
        metricB = dataB[task][1]
        assert metricA == metricB
        assert len(dataA[task]) == len(dataB[task])

        if metricA == "avg_mcauroc":
            p_value, delta = compute_significance_avg_mcauroc(probsA=dataA[task][0][1], referencesA=dataA[task][0][0],
                                                              probsB=dataB[task][0][1], referencesB=dataB[task][0][0])
            decisions[task] = {
                "significant": not (p_value > significance_level),
                "p_value": p_value,
                "delta": delta,
            }
    return decisions


def main():
    parser = argparse.ArgumentParser(description="One-tailed test if model A improves over model B.")
    parser.add_argument("--modelA", help="ModelA JSONL file from lm harness.")
    parser.add_argument("--modelB", help="ModelB JSONL file from lm harness.")
    parser.add_argument("--significance_level", type=float, default=0.05, help="Significance level (e.g., 0.05)")
    args = parser.parse_args()

    result = check_significance(args.modelA, args.modelB, args.significance_level)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
