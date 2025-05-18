# llama 3.2 model scaling

# zero-shot dataset split

uv run src/eval_misinfo/cli.py --ds x-fact-zero-shot --prov together --model meta-llama/Llama-3.2-3B-Instruct-Turbo --jsonl-filepath x-fact-zero-shot-together-llama-3.2-3B-mcq.jsonl --prompt-type mcq --max-workers 10
uv run src/eval_misinfo/cli.py --ds x-fact-zero-shot --prov together --model meta-llama/Llama-3.2-3B-Instruct-Turbo --jsonl-filepath x-fact-zero-shot-together-llama-3.2-3B-numerical.jsonl --prompt-type numerical --max-workers 10

# in-domain dataset split

uv run src/eval_misinfo/cli.py --ds x-fact-in-domain --prov together --model meta-llama/Llama-3.2-3B-Instruct-Turbo --jsonl-filepath x-fact-in-domain-together-llama-3.2-3B-mcq.jsonl --prompt-type mcq --max-workers 10
uv run src/eval_misinfo/cli.py --ds x-fact-in-domain --prov together --model meta-llama/Llama-3.2-3B-Instruct-Turbo --jsonl-filepath x-fact-in-domain-together-llama-3.2-3B-numerical.jsonl --prompt-type numerical --max-workers 10
