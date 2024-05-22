import argparse
import copy
import glob
import os
import json
from tqdm import tqdm

from leaderboard import SUPPORTED_METRICS


def process_harness_logs(input_folders, output_file, model_name, model_url, model_description):
    """
    - Selects best prompt for each task
    - Extract data for that prompt, necessary for targe/mnt/data/ifajcik/micromamba/envs/envs/lmharnest metrics
    """

    def expand_input_folders(input_folders):
        # Check if input_folders is a wildcard pattern
        if '*' in input_folders or '?' in input_folders:
            # Expand the wildcard into a list of matching directories
            matching_directories = [f for f in glob.glob(input_folders) if os.path.isdir(f)]
            return matching_directories
        else:
            # If it's not a wildcard, return the input as a single-item list if it's a valid directory
            if os.path.isdir(input_folders):
                return [input_folders]
            else:
                return []

    input_folders = expand_input_folders(input_folders)

    per_task_results = {}
    metric_per_task = {}
    predictions = {}

    for input_folder in tqdm(input_folders, desc="Loading files"):
        # read all files in input_folder
        with open(os.path.join(input_folder, "results.json"), "r") as f:
            harness_results = json.load(f)

        current_tasknames = []
        for name, result in harness_results['results'].items():
            if result['alias'].strip().startswith('- prompt-'):
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
                    current_tasknames.append(taskname)
                else:
                    per_task_results[taskname].append(result)

        # get best result according to metric priority given in SUPPORTED_METRICS list
        for taskname, results in per_task_results.items():
            if not taskname in current_tasknames:
                continue
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

        for file in os.listdir(input_folder):
            if file == "results.json":
                continue
            for taskname in per_task_results.keys():
                if taskname in file:
                    # check this file corresponds to same prompt
                    winning_prompt = per_task_results[taskname]['alias'][-1]
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
    aggregated_predictions["results"] = per_task_results
    aggregated_predictions["metadata"] = {
        "model_name": model_name,
        "model_description": model_description,
        "model_url": model_url
    }
    with open(output_file, "w") as f:
        json.dump(aggregated_predictions, f)


def main():
    parser = argparse.ArgumentParser(
        description="Process outputs of lm harness into minimum compatible format necessary for leaderboard submission.")
    parser.add_argument("-i", "-f", "--input_folder", "--folder",
                        help="Folder with unprocessed results from lm harness.", required=True)
    parser.add_argument("-o", "--output_file", help="File to save processed results.", required=True)
    parser.add_argument("--name", help="Name of the model.", required=True)
    parser.add_argument("--url", help="URL of the model.", required=False)
    parser.add_argument("--description", help="Description of the model.", required=True)
    args = parser.parse_args()

    process_harness_logs(args.input_folder, args.output_file, args.name, args.url, args.description)


if __name__ == "__main__":
    main()
