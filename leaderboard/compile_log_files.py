import argparse
import copy
import glob
import hashlib
import os
import json
import re

import jsonlines
from tqdm import tqdm

from leaderboard import SUPPORTED_METRICS, EXTRA_INFO_RELEASE_KEYS

with open("leaderboard/metadata.json", "r") as f:
    METADATA = json.load(f)

# TASK MAP
# from promptname to taskname
MAP = {
    'benchmark_agree': 'benczechmark_agree',
    'benchmark_belebele': 'benczechmark_belebele',
    'benchmark_czechnews': 'benczechmark_czechnews',
    'benchmark_subjectivity': 'benczechmark_subjectivity',
    'benczechmark_snli': 'benczechmark_snli',
    'propaganda_argumentace': 'benczechmark_propaganda_argumentace',
    'propaganda_fabulace': 'benczechmark_propaganda_fabulace',
    'propaganda_nazor': 'benczechmark_propaganda_nazor',
    'propaganda_strach': 'benczechmark_propaganda_strach',
    'propaganda_zamereni': 'benczechmark_propaganda_zamereni',
    'propaganda_demonizace': 'benczechmark_propaganda_demonizace',
    'propaganda_lokace': 'benczechmark_propaganda_lokace',
    'propaganda_relativizace': 'benczechmark_propaganda_relativizace',
    'propaganda_vina': 'benczechmark_propaganda_vina',
    'propaganda_zanr': 'benczechmark_propaganda_zanr',
    'propaganda_emoce': 'benczechmark_propaganda_emoce',
    'propaganda_nalepkovani': 'benczechmark_propaganda_nalepkovani',
    'propaganda_rusko': 'benczechmark_propaganda_rusko',
    'benczechmark_sentiment_mall': 'benczechmark_sentiment_mall',
    'benczechmark_sentiment_fb': 'benczechmark_sentiment_fb',
    'benczechmark_sentiment_csfd': 'benczechmark_sentiment_csfd',
    'benczechmark_summarization': 'benczechmark_summarization',
    'gec': 'benczechmark_grammarerrorcorrection',
    'cs_nq_open': 'benczechmark_cs_naturalquestions',
    'cs_sqad_open': 'benczechmark_cs_sqad32',
    'cs_triviaqa': 'benczechmark_cs_triviaQA',
    'csfever': 'benczechmark_csfever_nli',
    'ctkfacts': 'benczechmark_ctkfacts_nli',
    'cnec_ner': 'benczechmark_cs_ner',
    'cdec_ner': 'benczechmark_cs_court_decisions_ner',
    'klokan_qa': 'benczechmark_klokan_qa',
    'umimeto_biology': 'benczechmark_umimeto_biology',
    'umimeto_chemistry': 'benczechmark_umimeto_chemistry',
    'umimeto_czech': 'benczechmark_umimeto_czech',
    'umimeto_history': 'benczechmark_umimeto_history',
    'umimeto_informatics': 'benczechmark_umimeto_informatics',
    'umimeto_math': 'benczechmark_umimeto_math',
    'umimeto_physics': 'benczechmark_umimeto_physics',
    'cermat_czech_open': 'benczechmark_cermat_czech_open',
    'cermat_czech_mc': 'benczechmark_cermat_czech_mc',
    'cermat_czech_tf': 'benczechmark_cermat_czech_tf',
    'cermat_czmath_open': 'benczechmark_cermat_czmath_open',
    'cermat_czmath_mc': 'benczechmark_cermat_czmath_mc',
    'history_ir': 'benczechmark_history_ir',
    'benczechmark_histcorpus': "benczechmark_histcorpus",
    'benczechmark_hellaswag': "benczechmark_hellaswag",
    'benczechmark_essay': 'benczechmark_essay',
    'benczechmark_fiction': 'benczechmark_fiction',
    'benczechmark_capek': 'benczechmark_capek',
    'benczechmark_correspondence': 'benczechmark_correspondence',
    'benczechmark_havlicek': 'benczechmark_havlicek',
    'benczechmark_speeches': 'benczechmark_speeches',
    'benczechmark_spoken': 'benczechmark_spoken',
    'benczechmark_dialect': 'benczechmark_dialect'
}

NO_PROMPT_TASKS = ["benczechmark_histcorpus",
                   "benczechmark_hellaswag",
                   "benczechmark_essay",
                   "benczechmark_fiction",
                   "benczechmark_capek",
                   "benczechmark_correspondence",
                   "benczechmark_havlicek",
                   "benczechmark_speeches",
                   "benczechmark_spoken",
                   "benczechmark_dialect"]


def resolve_taskname(taskname):
    if taskname not in MAP:
        raise ValueError(f"Taskname {taskname} not found.")
    return MAP[taskname]


def rename_keys(d, resolve_taskname):
    orig_len = len(d)
    for k, v in list(d.items()):
        new_key = resolve_taskname(k)
        d[new_key] = d.pop(k)

    # make sure list length didnt changed
    assert len(d) == orig_len


def process_harness_logs(input_folders, output_file):
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
        # consider first folder within this folder
        input_folder = os.path.join(input_folder, os.listdir(input_folder)[0])
        # find file which starts with results... prefix in the input_folder
        result_file = [f for f in os.listdir(input_folder) if f.startswith("results")][0]
        with open(os.path.join(input_folder, result_file), "r") as f:
            harness_results = json.load(f)

        current_multipleprompt_tasknames = []
        for name, result in harness_results['results'].items():
            if name in NO_PROMPT_TASKS:
                # not prompts
                taskname = name
                # process metric names
                for k, v in copy.deepcopy(result).items():
                    if "," in k:
                        name, _ = k.split(",")
                        del result[k]
                        result[name] = v
                per_task_results[taskname] = result

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
                    current_multipleprompt_tasknames.append(taskname)
                else:
                    per_task_results[taskname].append(result)

        # get best result according to metric priority given in SUPPORTED_METRICS list
        for taskname, results in per_task_results.items():
            if not taskname in current_multipleprompt_tasknames:
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

            all_measured_results = []
            for result in results:
                all_measured_results.append(result[target_metric])
                if best_result is None:
                    best_result = result
                else:
                    if result[target_metric] > best_result[target_metric]:
                        best_result = result
            # Compute max-centered variance
            max_value = best_result[target_metric]
            squared_diffs = [(x * 100.0 - max_value * 100.0) ** 2 for x in all_measured_results]
            max_centered_variance = sum(squared_diffs) / (len(squared_diffs) - 1)
            best_result['max_centered_variance'] = max_centered_variance

            per_task_results[taskname] = best_result

        for file in os.listdir(input_folder):
            if file == result_file:
                continue
            for taskname in per_task_results.keys():
                if taskname in file:
                    print(f"Processing {os.path.join(input_folder, file)} for {taskname}")
                    # check this file corresponds to same prompt
                    winning_prompt = per_task_results[taskname]['alias'][-1]
                    if taskname in NO_PROMPT_TASKS:
                        current_prompt = "-1"
                    else:
                        try:
                            current_prompt = re.search(rf"{taskname}_(\d+)_", file).group(1)
                        except AttributeError:
                            raise ValueError(f"Prompt not found in {file}")
                    if winning_prompt == current_prompt or taskname in NO_PROMPT_TASKS:
                        # load file contents
                        predictions[taskname] = list(jsonlines.open(os.path.join(input_folder, file)))
                        # only keep data necessary for metrics
                        for prediction in predictions[taskname]:
                            for key in list(prediction.keys()):
                                if key not in SUPPORTED_METRICS + EXTRA_INFO_RELEASE_KEYS:
                                    del prediction[key]

    # rename keys (tasknames) using resolve_tasknames:
    rename_keys(predictions, resolve_taskname)
    rename_keys(per_task_results, resolve_taskname)

    # assert keys in predictions and results are the same
    # assert set(predictions.keys()) == set(per_task_results.keys())
    if not set(predictions.keys()) == set(per_task_results.keys()):
        # print missing keys
        print("Missing keys in predictions:")
        print(set(predictions.keys()) - set(per_task_results.keys()))
        # print extra keys
        print("Extra keys in predictions:")
        print(set(per_task_results.keys()) - set(predictions.keys()))
        raise ValueError("Keys in predictions and results are not the same")

    aggregated_predictions = dict()
    aggregated_predictions["predictions"] = predictions
    aggregated_predictions["results"] = per_task_results
    aggregated_predictions["metadata"] = {
        'git_hash': harness_results['git_hash'],
        'transformers_version': harness_results['transformers_version'],
        'tokenizer_pad_token': harness_results['tokenizer_pad_token'],
        'tokenizer_eos_token': harness_results['tokenizer_eos_token'],
        'tokenizer_bos_token': harness_results['tokenizer_bos_token'],
        'eot_token_id': harness_results['eot_token_id'],
        'max_length': harness_results['max_length'],
        'task_hashes': harness_results['task_hashes'],
        'model_source': harness_results['model_source'],
        'model_name': harness_results['model_name'],
        'model_name_sanitized': harness_results['model_name_sanitized'],
        'system_instruction': harness_results['system_instruction'],
        'system_instruction_sha': harness_results['system_instruction_sha'],
        'fewshot_as_multiturn': harness_results['fewshot_as_multiturn'],
        'chat_template': harness_results['chat_template'],
        'chat_template_sha': harness_results['chat_template_sha'],
        'start_time': harness_results['start_time'],
        'end_time': harness_results['end_time'],
        'total_evaluation_time_seconds': harness_results['total_evaluation_time_seconds']
    }

    # make sure all tasks are present
    all_tasks = set(METADATA["tasks"].keys())
    all_expected_tasks = set(per_task_results.keys())
    all_missing_tasks = all_tasks - all_expected_tasks
    all_extra_tasks = all_expected_tasks - all_tasks
    if len(all_missing_tasks) > 0:
        EOLN = "\n"
        # print(f"Missing tasks: {EOLN.join(all_missing_tasks)}")
        raise Exception(f"Missing tasks: {EOLN.join(all_missing_tasks)}")  # TODO: uncomment
    if len(all_extra_tasks) > 0:
        EOLN = "\n"
        raise Exception(f"Extra tasks: {EOLN.join(all_extra_tasks)}")
    with open(output_file, "w") as f:
        json.dump(aggregated_predictions, f)
    print("Success!")
    print("Output saved to", output_file)


def main():
    parser = argparse.ArgumentParser(
        description="Process outputs of lm harness into minimum compatible format necessary for leaderboard submission.")
    parser.add_argument("-i", "-f", "--input_folder", "--folder",
                        help="Folder with unprocessed results from lm harness.", required=True)
    parser.add_argument("-o", "--output_file", help="File to save processed results.", required=True)
    args = parser.parse_args()

    process_harness_logs(args.input_folder, args.output_file)


if __name__ == "__main__":
    main()
