import together
import argparse
from dotenv import load_dotenv
import os
import pandas as pd

load_dotenv()


def score(pred, gt):
    score_mat = {
        "true": {
            "true": 1.0,
            "mostly true": 0.75,
            "partly true/misleading": 0.5,
            "complicated/hard to categorise": 0.0,
            "other": 0.0,
            "mostly false": 0.0,
            "false": 0.0,
        },
        "mostly true": {
            "true": 0.75,
            "mostly true": 1.0,
            "partly true/misleading": 0.75,
            "complicated/hard to categorise": 0.0,
            "other": 0.0,
            "mostly false": 0.0,
            "false": 0.0,
        },
        "partly true/misleading": {
            "true": 0.25,
            "mostly true": 0.50,
            "partly true/misleading": 1.0,
            "complicated/hard to categorise": 0.5,
            "other": 0.5,
            "mostly false": 0.50,
            "false": 0.25,
        },
        "complicated/hard to categorise": {
            "true": 0.0,
            "mostly true": 0.0,
            "partly true/misleading": 0.0,
            "complicated/hard to categorise": 1.0,
            "other": 0.0,
            "mostly false": 0.0,
            "false": 0.0,
        },
        "other": {
            "true": 0.0,
            "mostly true": 0.0,
            "partly true/misleading": 0,
            "complicated/hard to categorise": 0,
            "other": 1.0,
            "mostly false": 0.0,
            "false": 0.0,
        },
        "mostly false": {
            "true": 0.25,
            "mostly true": 0.5,
            "partly true/misleading": 0.5,
            "complicated/hard to categorise": 0.5,
            "other": 0.5,
            "mostly false": 1.00,
            "false": 0.75,
        },
        "false": {
            "true": 0.0,
            "mostly true": 0.25,
            "partly true/misleading": 0.5,
            "complicated/hard to categorise": 0.5,
            "other": 0.5,
            "mostly false": 0.75,
            "false": 1.0,
        },
    }
    return score_mat[gt][pred]


def generate_prompt_and_answer_for_x_fact(claim, answer):
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


def run_eval(ds_name, provider, model):
    df = pd.read_csv(
        "data/x_fact_dataset/x-fact/zeroshot.tsv", delimiter="\t", on_bad_lines="skip"
    )
    print(df.head())
    print(df.label.unique())

    print(df.shape)
    df = df.dropna(subset=["label"])
    print(df.shape)

    IDX = 500

    claim = df.iloc[IDX]["claim"]
    print(claim)
    answer = df.iloc[IDX]["label"]

    prompt, answer = generate_prompt_and_answer_for_x_fact(claim, answer)

    if provider == "together":
        client = together.Together(api_key=os.getenv("TOGETHER_API_KEY"))
        completion = client.chat.completions.create(
            model=model,
            messages=[
                {
                    "role": "user",
                    "content": prompt,
                }
            ],
        )

        pred = completion.choices[0].message.content
        print("---")
        print(pred, answer)

        pred = pred.split("</think>")[-1].strip()
        print(pred)

        letter_to_label = {
            "A": "true",
            "B": "mostly true",
            "C": "partly true/misleading",
            "D": "complicated/hard to categorise",
            "E": "other",
            "F": "mostly true",
            "G": "false",
        }

        print(score(letter_to_label[pred.strip()], answer))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--ds")
    parser.add_argument("--prov")
    parser.add_argument("--model")
    args = parser.parse_args()
    if args.ds not in ["x-fact", "mumin"]:
        raise ValueError("--ds must be either 'x-fact' or 'mumin'")
    print(args.ds)
    print(args.model)
    run_eval(ds_name=args.ds, provider=args.prov, model=args.model)
