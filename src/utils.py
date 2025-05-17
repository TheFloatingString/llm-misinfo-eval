import argparse
import json
import numpy as np


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
