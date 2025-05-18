# zero-shot dataset split

uv run src/eval_misinfo/cli.py --ds x-fact-zero-shot --prov together --model deepseek-ai/DeepSeek-R1 --jsonl-filepath x-fact-zero-shot-together-deepseek-r1-mcq.jsonl --prompt-type mcq --max-workers 30
uv run src/eval_misinfo/cli.py --ds x-fact-zero-shot --prov together --model deepseek-ai/DeepSeek-R1 --jsonl-filepath x-fact-zero-shot-together-deepseek-r1-numerical.jsonl --prompt-type numerical --max-workers 30

# in-domain dataset split

uv run src/eval_misinfo/cli.py --ds x-fact-in-domain --prov together --model deepseek-ai/DeepSeek-R1 --jsonl-filepath x-fact-in-domain-together-deepseek-r1-mcq.jsonl --prompt-type mcq --max-workers 30
uv run src/eval_misinfo/cli.py --ds x-fact-in-domain --prov together --model deepseek-ai/DeepSeek-R1 --jsonl-filepath x-fact-in-domain-together-deepseek-r1-numerical.jsonl --prompt-type numerical --max-workers 30
