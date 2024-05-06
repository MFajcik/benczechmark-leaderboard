import argparse
import copy
import os
import json

from tqdm import tqdm

from leaderboard import SUPPORTED_METRICS


def process_harness_logs(input_folder, output_file, model_name, model_description):
    """
    - Selects best prompt for each task
    - Extract data for that prompt, necessary for targe/mnt/data/ifajcik/micromamba/envs/envs/lmharnest metrics
    """

    per_task_results = {}
    # read all files in input_folder
    with open(os.path.join(input_folder, "results.json"), "r") as f:
        harness_results = json.load(f)

    for name, result in harness_results['results'].items():
        if result['alias'].startswith('  - prompt-'):
            # process taskname
            taskname = name[:-1]
            if taskname.endswith("_"):
                taskname = taskname[:-1]
            # process metric names
            for k, v in copy.deepcopy(result).items():
                if "," in k:
                    name, key = k.split(",")
                    del result[k]
                    result[name] = v
            if taskname not in per_task_results:
                per_task_results[taskname] = [result]
            else:
                per_task_results[taskname].append(result)

    metric_per_task = dict()
    # get best result according to metric priority given in SUPPORTED_METRICS list
    for taskname, results in per_task_results.items():
        best_result = None
        target_metric = None
        for m in SUPPORTED_METRICS:
            if m in results[0]:
                target_metric = m
                break
        if target_metric is None:
            raise ValueError(f"No supported metric found in {taskname}")
        metric_per_task[taskname] = target_metric

        for result in results:
            if best_result is None:
                best_result = result
            else:
                if result[target_metric] > best_result[target_metric]:
                    best_result = result
        per_task_results[taskname] = best_result

    predictions = dict()
    for file in tqdm(os.listdir(input_folder), desc="Loading files"):
        if file == "results.json":
            continue
        for taskname in per_task_results.keys():
            if taskname in file:
                # check this file corresponds to same prompt
                winning_prompt = per_task_results[taskname]['alias'][-1]
                # 'pretrained__BUT-FIT__CSMPT7b,dtype__bfloat16,max_length__2048,truncation__True,trust_remote_code__True_propaganda_rusko3.jsonl'
                current_prompt = file[:-len(".jsonl")][-1]
                if winning_prompt == current_prompt:
                    with open(os.path.join(input_folder, file), "r") as f:
                        # load file contents
                        predictions[taskname] = json.load(f)
                        # only keep data necessary for metrics
                        for prediction in predictions[taskname]:
                            for key in list(prediction.keys()):
                                if key not in SUPPORTED_METRICS:
                                    del prediction[key]
    aggregated_predictions = dict()
    aggregated_predictions["predictions"] = predictions
    aggregated_predictions["results"] = harness_results
    aggregated_predictions["metadata"] = {
        "model_name": model_name,
        "model_description": model_description,
    }
    with open(output_file, "w") as f:
        json.dump(aggregated_predictions, f)


def main():
    parser = argparse.ArgumentParser(
        description="Process outputs of lm harness into minimum compatible format necessary for leaderboard submission.")
    parser.add_argument("-i", "-f", "--input_folder", "--folder",
                        help="Folder with unprocessed results from lm harness.")
    parser.add_argument("-o", "--output_file", help="File to save processed results.")
    parser.add_argument("--model_name", help="Name of the model.")
    parser.add_argument("--model_description", help="Description of the model.")
    args = parser.parse_args()

    process_harness_logs(args.input_folder, args.output_file, args.model_name, args.model_description)


if __name__ == "__main__":
    main()
