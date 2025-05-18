from eval_misinfo import model_loader, utils
import pandas as pd
import tqdm
import json
from concurrent.futures import ThreadPoolExecutor, as_completed
import datetime
import metrics


def run_experiment(provider: str, model: str, prompt_type: str, dataset: str):
    # load dataset
    # run multithread eval
    # get correct model
    if provider == "together":
        pass
    elif provider == "openai":
        pass
    elif provider == "cohere":
        pass
    elif provider == "gemini":
        pass


def run_single_eval(df, IDX, provider, model, jsonl_filepath, prompt_type):
    claim = df.iloc[IDX]["claim"]
    answer = df.iloc[IDX]["label"]

    prompt, answer = utils.generate_prompt_and_answer_for_x_fact_mcq(claim, answer)

    letter_to_label = {
        "A": "true",
        "B": "mostly true",
        "C": "partly true/misleading",
        "D": "complicated/hard to categorise",
        "E": "other",
        "F": "mostly true",
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

        new_data = {
            "dataset": "x-fact-zeroshot.tsv",
            "idx_after_dropna": IDX,
            "raw_pred": pred,
            "pred": letter_to_label[pred.strip()],
            "ground_truth": answer,
            "time": str(datetime.datetime.now()),
            "score": metrics.score(letter_to_label[pred.strip()], answer),
            "model": model,
            "provider": provider,
            "languge": df.iloc[IDX]["language"],
        }
        with open(jsonl_filepath, "a", encoding="utf-8") as f:
            f.write(json.dumps(new_data) + "\n")

    except:
        with open(jsonl_filepath, "a", encoding="utf-8") as f:
            f.write(json.dumps({"score": 0}) + "\n")
            print(f"error at {IDX}: `{pred}`")


def run_eval(
    provider: str,
    model: str,
    jsonl_filepath: str,
    ds_name: str,
    prompt_type: str,
    max_workers: int = 5,
):
    if ds_name == "x-fact":
        df = pd.read_csv(
            "data/x_fact_dataset/x-fact/zeroshot.tsv",
            delimiter="\t",
            on_bad_lines="skip",
        )
    else:
        raise ValueError(f"dataset name `{ds_name}`not recogized")

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = {
            executor.submit(
                run_single_eval, df, IDX, provider, model, jsonl_filepath,
                prompt_type
            ): IDX
            for IDX in range(df.shape[0])
        }

        for future in tqdm.tqdm(
            as_completed(futures), total=len(futures), desc="Running evaluations"
        ):
            try:
                result = future.result()
                # results.append(result)
            except Exception as e:
                print(f"Error: {e}")
