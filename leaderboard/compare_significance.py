import argparse
import concurrent
import json
import random
from collections import defaultdict
from typing import Sequence

import numpy
import numpy as np
from scipy.stats import ttest_rel
from sklearn.metrics import roc_curve, auc
from tqdm import tqdm
from numba import njit, prange

from leaderboard import SUPPORTED_METRICS

# Set the random seed
SEED = 42
random.seed(SEED)
np.random.seed(SEED)


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


def compute_significance_ttest(scores_A, scores_B):
    delta = np.mean(scores_A) - np.mean(scores_B)
    t, p = ttest_rel(scores_A, scores_B)
    # correct for one-tailed test
    p_value = p / 2
    return p_value, delta


@njit(parallel=True)
def compute_significance_bootstrap(scores_A, scores_B):
    n = len(scores_A)
    R = 1_000
    delta_orig = np.mean(scores_A) - np.mean(scores_B)

    r = 0
    for _ in prange(R):
        samples = np.random.choice(n, n, replace=True)
        temp_A = scores_A[samples]
        temp_B = scores_B[samples]
        delta = np.mean(temp_A) - np.mean(temp_B)
        if delta > 2 * delta_orig:
            r += 1

    pval = r / R
    return pval, delta_orig


@njit(parallel=True)
def compute_significance_bootstrap_pp(scores_A, scores_B):
    n = len(scores_A)
    R = 1_000

    scores_A, words_A = scores_A[:, 0], scores_A[:, 1]
    scores_B, words_B = scores_B[:, 0], scores_B[:, 1]

    def get_pp(scores, words):
        return np.exp(-scores.sum() / words.sum())

    # inverted as lower is better
    delta_orig = get_pp(scores_B, words_B) - get_pp(scores_A, words_A)

    r = 0
    for _ in prange(R):
        samples = np.random.choice(n, n, replace=True)
        temp_A = scores_A[samples]
        temp_words_A = words_A[samples]
        temp_B = scores_B[samples]
        temp_words_B = words_B[samples]
        delta = get_pp(temp_B, temp_words_B) - get_pp(temp_A, temp_words_A)
        if delta > 2 * delta_orig:
            r += 1

    pval = r / R
    return pval, delta_orig


def compute_significance_avg_mcauroc(probsA: Sequence[Sequence[float]], referencesA: Sequence[int],
                                     probsB: Sequence[Sequence[float]], referencesB: Sequence[int]):
    # compute MC-AUC for model A
    model_A_scores = get_mc_auc_samples(probsA, referencesA, Nsamples=1_000)
    model_B_scores = get_mc_auc_samples(probsB, referencesB, Nsamples=1_000)
    delta = np.mean(model_A_scores) - np.mean(model_B_scores)

    # one-tailed test
    p_value = ((model_A_scores[:, np.newaxis] <= model_B_scores[np.newaxis, :]).sum()
               / (len(model_A_scores) * len(model_B_scores)))

    return p_value, delta


# Helper function to convert confusion matrices to numba-compatible arrays
def convert_confusion_matrices(confusion_matrices):
    num_thresholds = len(confusion_matrices)
    tp = np.empty(num_thresholds)
    fn = np.empty(num_thresholds)
    for k in range(num_thresholds):
        tp[k] = confusion_matrices[k]["TP"]
        fn[k] = confusion_matrices[k]["FN"]
    return tp, fn


@njit(parallel=True)
def compute_tpr_variates(tp, fn, λ, Nsamples, num_thresholds):
    tpr_variates_for_each_fpr = np.empty((num_thresholds, Nsamples))
    for k in prange(num_thresholds):
        tpr_variates_for_each_fpr[k, :] = np.random.beta(tp[k] + λ, fn[k] + λ, Nsamples)
    return tpr_variates_for_each_fpr


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
        tp, fn = convert_confusion_matrices(confusion_matrices)

        λ = 1.0  # <- Flat prior
        # λ = 0.5  # <- Jeffrey's prior

        # sample variates for every threshold
        # tpr_variates_for_each_fpr = []
        # for k in range(len(thresholds[i])):
        #     tpr_variates_for_each_fpr.append(
        #         numpy.random.beta(confusion_matrices[k]["TP"] + λ, confusion_matrices[k]["FN"] + λ, Nsamples))
        tpr_variates_for_each_fpr = compute_tpr_variates(tp, fn, λ, Nsamples, len(thresholds[i]))

        # fprs x tpr_variates
        # tpr_variates_for_each_fpr = np.array(tpr_variates_for_each_fpr)

        # now pick 1 variate for each fpr, and compute AUC
        auc_scores = []
        for tpr_variates in tpr_variates_for_each_fpr.T:
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
        else:
            scores = [line[metric] for line in fc["predictions"][task]]
            data[task] = scores, metric
    data['results'] = fc['results']

    # make sure all tasks are submitted
    METADATA_FILE = "leaderboard/metadata.json"
    with open(METADATA_FILE, "r") as f:
        metadata = json.load(f)

    all_tasks = list(metadata["tasks"].keys())
    all_missing_tasks = []
    for task in all_tasks:
        if task not in data:
            all_missing_tasks.append(task)
    if len(all_missing_tasks) > 0:
        EOLN = "\n"
        raise ValueError(f"Missing tasks in {file_path}: {EOLN.join(all_missing_tasks)}")
    return data


def process_task(task, dataA, dataB, significance_level):
    metricA = dataA[task][1]
    metricB = dataB[task][1]

    # hotfix for rouge_raw naming,
    # TODO: MF - to be refactored, after deprecating _without_bootstrap suffix
    def hotfix_metric_name(m, d):
        if m == 'rouge_raw_r2_mid_f_without_bootstrap':
            newm = "rouge_raw_r2_mid_f"
            d['results'][task][newm] = d['results'][task][m]
            m = newm
        return m

    metricA = hotfix_metric_name(metricA, dataA)
    metricB = hotfix_metric_name(metricB, dataB)
    ##

    assert metricA == metricB
    assert len(dataA[task]) == len(dataB[task])

    A_result = dataA['results'][task][metricA]
    B_result = dataB['results'][task][metricB]

    delta_measured = A_result - B_result
    if metricA == "word_perplexity":
        A_result = -A_result
        B_result = -B_result
        delta_measured = -delta_measured

    if A_result < B_result:
        return task, {
            "significant": False,
            "p_value": 1,
            "delta": delta_measured,
        }

    if metricA == "avg_mcauroc":
        p_value, delta = compute_significance_avg_mcauroc(probsA=dataA[task][0][1], referencesA=dataA[task][0][0],
                                                          probsB=dataB[task][0][1], referencesB=dataB[task][0][0])
    elif metricA in ["acc", "exact_match"]:
        # t-test is symmetric
        p_value, delta = compute_significance_ttest(scores_A=dataA[task][0], scores_B=dataB[task][0])
    elif metricA in ["rouge_raw_r2_mid_f"]:
        p_value, delta = compute_significance_bootstrap(scores_A=np.array(dataA[task][0]),
                                                        scores_B=np.array(dataB[task][0]))
    elif metricA in ["word_perplexity"]:
        p_value, delta = compute_significance_bootstrap_pp(scores_A=np.array(dataA[task][0]),
                                                           scores_B=np.array(dataB[task][0]))
    else:
        raise ValueError(f"Unsupported metric {metricA}")

    return task, {
        "significant": not (p_value > significance_level),
        "p_value": p_value,
        "delta": delta,
    }


def check_significance(fileA, fileB, significance_level):
    dataA = read_json(fileA)
    dataB = read_json(fileB)

    decisions = dict()
    tasks = list(dataA.keys())

    with concurrent.futures.ProcessPoolExecutor() as executor:
        futures = {executor.submit(process_task, task, dataA, dataB, significance_level): task for task in tasks if
                   task != 'results'}
        _iter = tqdm(concurrent.futures.as_completed(futures), total=len(tasks))
        for future in _iter:
            task, result = future.result()
            _iter.set_description(f"Processing task: {task}")
            decisions[task] = result

    return decisions


def main():
    parser = argparse.ArgumentParser(description="One-tailed test if model A improves over model B.")
    parser.add_argument("--modelA", help="ModelA JSON file from lm harness.")
    parser.add_argument("--modelB", help="ModelB JSON file from lm harness.")
    parser.add_argument("--significance_level", type=float, default=0.05, help="Significance level (e.g., 0.05)")
    parser.add_argument("--output", help="Output file for the results.", default="significance.json")
    args = parser.parse_args()

    result = check_significance(args.modelA, args.modelB, args.significance_level)
    print(json.dumps(result, indent=2))
    with open(args.output, "w", encoding='utf-8') as f:
        json.dump(result, f, indent=2)


# harness already returns stderr estimate for sampling distribution
# see https://github.com/EleutherAI/lm-evaluation-harness/blob/6433bd3fe3033d302b22cdcd53af237e9039ef29/lm_eval/api/metrics.py#L213

if __name__ == "__main__":
    main()
