from eval_misinfo import experiments
import argparse

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--ds")
    parser.add_argument("--prov")
    parser.add_argument("--model")
    parser.add_argument("--jsonl-filepath")
    parser.add_argument("--prompt-type")
    parser.add_argument("--max-workers", default=5)
    args = parser.parse_args()
    if args.ds not in ["x-fact-in-domain", "x-fact-zero-shot", "mumin"]:
        raise ValueError(
            "--ds must be either 'x-fact-in-domain', 'x-fact-zero-shot' or 'mumin'"
        )
    print(args.ds)
    print(args.model)
    experiments.run_eval(
        provider=args.prov,
        model=args.model,
        jsonl_filepath=args.jsonl_filepath,
        prompt_type=args.prompt_type,
        ds_name=args.ds,
        max_workers=int(args.max_workers)
    )
