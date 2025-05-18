# llama 3.1 model scaling

# zero-shot dataset split

uv run src/eval_misinfo/cli.py --ds x-fact-zero-shot --prov together --model meta-llama/Meta-Llama-3.1-8B-Instruct-Turbo --jsonl-filepath x-fact-zero-shot-together-llama-3.1-8b-mcq.jsonl --prompt-type mcq --max-workers 10
uv run src/eval_misinfo/cli.py --ds x-fact-zero-shot --prov together --model meta-llama/Meta-Llama-3.1-8B-Instruct-Turbo --jsonl-filepath x-fact-zero-shot-together-llama-3.1-8b-numerical.jsonl --prompt-type numerical --max-workers 10

# in-domain dataset split

uv run src/eval_misinfo/cli.py --ds x-fact-in-domain --prov together --model meta-llama/Meta-Llama-3.1-8B-Instruct-Turbo --jsonl-filepath x-fact-in-domain-together-llama-3.1-8b-mcq.jsonl --prompt-type mcq --max-workers 10
uv run src/eval_misinfo/cli.py --ds x-fact-in-domain --prov together --model meta-llama/Meta-Llama-3.1-8B-Instruct-Turbo --jsonl-filepath x-fact-in-domain-together-llama-3.1-8b-numerical.jsonl --prompt-type numerical --max-workers 10
