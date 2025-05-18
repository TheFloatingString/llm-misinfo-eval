import argparse
import json
import numpy as np


def generate_prompt_and_answer_for_x_fact_mcq(claim, answer):
    prompt = f"""Given the following claim:
"{claim}"

determine whether this statement is:

A. true
B. mostly true
C. partly true/misleading
D. complicated/hard to categorize
E. other
F. mostly false
G. false

Only respond with the corresponding uppercase letter (A to G). Answer with a single letter, do not write anything else:"""
    return prompt, answer


def calc_mean_f1_from_jsonl_file(jsonl_filepath):
    with open(jsonl_filepath) as f:
        data = [json.loads(line) for line in f]
    f1_score_list = np.zeros(3320)
    print(f1_score_list.shape)
    print(len(data))
    for i in range(len(data)):
        f1_score_list[i] = data[i]["score"]
    print(
        f"mean score: {round(np.mean(f1_score_list), 5)} +/- {round(np.std(f1_score_list), 5)}"
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("jsonl_filepath")
    args = parser.parse_args()
    calc_mean_f1_from_jsonl_file(args.jsonl_filepath)
