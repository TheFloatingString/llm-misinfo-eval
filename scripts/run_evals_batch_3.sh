# openai o4-mini

# zero-shot dataset split

uv run src/eval_misinfo/cli.py --ds x-fact-zero-shot --prov openai --model o4-mini --jsonl-filepath x-fact-zero-shot-openai-o4-mini-mcq.jsonl --prompt-type mcq --max-workers 60
uv run src/eval_misinfo/cli.py --ds x-fact-zero-shot --prov openai --model o4-mini --jsonl-filepath x-fact-zero-shot-openai-o4-mini-numerical.jsonl --prompt-type numerical --max-workers 60

# in-domain dataset split

uv run src/eval_misinfo/cli.py --ds x-fact-in-domain --prov openai --model o4-mini --jsonl-filepath x-fact-in-domain-openai-o4-mini-mcq.jsonl --prompt-type mcq --max-workers 60
uv run src/eval_misinfo/cli.py --ds x-fact-in-domain --prov openai --model o4-mini --jsonl-filepath x-fact-in-domain-openai-o4-mini-numerical.jsonl --prompt-type numerical --max-workers 60
