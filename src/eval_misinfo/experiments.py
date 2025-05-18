from eval_misinfo import model_loader, utils
import pandas as pd
import tqdm
import json
from concurrent.futures import ThreadPoolExecutor, as_completed
import datetime
import metrics


def number_to_label(number: int):
    if 0 <= number < 20:
        return "false"
    if 20 <= number < 40:
        return "mostly false"
    if 40 <= number < 60:
        return "partly true/misleading"
    if 60 <= number < 80:
        return "mostly true"
    if 80 <= number <= 100:
        return "true"
    else:
        return "complicated/hard to categorise"


def run_single_eval(df, IDX, provider, model, jsonl_filepath, prompt_type, ds_name):
    claim = df.iloc[IDX]["claim"]
    answer = df.iloc[IDX]["label"]

    if prompt_type == "mcq":
        prompt, answer = utils.generate_prompt_and_answer_for_x_fact_mcq(claim, answer)
    elif prompt_type == "numerical":
        prompt, answer = utils.generate_prompt_and_answer_for_x_fact_numerical(
            claim, answer
        )

    letter_to_label = {
        "A": "true",
        "B": "mostly true",
        "C": "partly true/misleading",
        "D": "complicated/hard to categorise",
        "E": "other",
        "F": "mostly false",
        "G": "false",
    }

    if provider == "together":
        pred = model_loader.call_together(model=model, prompt=prompt)
        pred = pred.strip()

    elif provider == "openai":
        pred = model_loader.call_openai(model=model, prompt=prompt)
        pred = pred.strip()

    elif provider == "gemini":
        pred = model_loader.call_gemini(model=model, prompt=prompt)
        pred = pred.strip()

    elif provider == "cohere":
        pred = model_loader.call_cohere(model=model, prompt=prompt)
        pred = pred.strip()

    try:
        pred = pred.split("</think>")[-1].strip()
        score = None
        if prompt_type == "numerical":
            score = metrics.score(number_to_label(int(pred.strip())), answer)
            pred_for_json = number_to_label(int(pred.strip()))
        elif prompt_type == "mcq":
            score = metrics.score(letter_to_label[pred.strip()], answer)
            pred_for_json = letter_to_label[pred.strip()]
        else:
            raise ValueError(
                f"prompt_type `{prompt_type}` must be either 'numerical' or 'mcq'"
            )

        new_data = {
            "dataset": ds_name,
            "idx_after_dropna": IDX,
            "raw_pred": pred,
            "pred": pred_for_json,
            "ground_truth": answer,
            "time": str(datetime.datetime.now()),
            "score": score,
            "model": model,
            "provider": provider,
            "language": df.iloc[IDX]["language"],
            "prompt_type": prompt_type,
        }
        with open(jsonl_filepath, "a", encoding="utf-8") as f:
            f.write(json.dumps(new_data) + "\n")

    except KeyboardInterrupt:
        with open(jsonl_filepath, "a", encoding="utf-8") as f:
            f.write(json.dumps({"score": 0}) + "\n")
            print(f"error at {IDX}: `{pred}`")


def run_eval(
    provider: str,
    model: str,
    jsonl_filepath: str,
    ds_name: str,
    prompt_type: str,
    max_workers: int = 10,
):
    if ds_name == "x-fact-zero-shot":
        df = pd.read_csv(
            "data/x_fact_dataset/x-fact/zeroshot.tsv",
            delimiter="\t",
            on_bad_lines="skip",
        )
    elif ds_name == "x-fact-in-domain":
        df = pd.read_csv(
            "data/x_fact_dataset/x-fact/test.all.tsv",
            sep="\t",
            quotechar='"',
            engine="python",
            on_bad_lines="skip",
        )

    else:
        raise ValueError(f"dataset name `{ds_name}`not recogized")

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = {
            executor.submit(
                run_single_eval,
                df,
                IDX,
                provider,
                model,
                jsonl_filepath,
                prompt_type,
                ds_name,
            ): IDX
            for IDX in range(df.shape[0])
        }

        for future in tqdm.tqdm(
            as_completed(futures), total=len(futures), desc="Running evaluations"
        ):
            try:
                result = future.result()
                # results.append(result)
            except KeyboardInterrupt as e:
                print(f"Error: {e}")
