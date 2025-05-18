import argparse
import json
import numpy as np


def calc_mean_f1_from_jsonl_file(jsonl_filepath):
    with open(jsonl_filepath) as f:
        data = [json.loads(line) for line in f]
    N = len(data)
    f1_score_list = []
    print(len(data))
    for i in range(len(data)):
        f1_score_list.append(data[i]["score"])
    print(
        f"mean score: {round(np.mean(f1_score_list), 5)} +/- {round(np.std(f1_score_list), 5)}"
    )
    data = [item for item in data if 'language' in item.keys()]
    unique_lang_list = list(set([item["language"] for item in data]))
    print(unique_lang_list)
    for language in unique_lang_list:
        f1_score_list = []
        for item in data:
            if 'language' in item.keys() and item["language"] == language:
                f1_score_list.append(item["score"])
        print("language:", language)
        print(
            f"{round(np.mean(f1_score_list), 5)}\t+/-\t{round(np.std(f1_score_list), 5)} \tN={len(f1_score_list)}"
        )
    print(f"failure rate: {(N-len(data))/(N)*100} %")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--jsonl-filepath")
    args = parser.parse_args()
    calc_mean_f1_from_jsonl_file(args.jsonl_filepath)
