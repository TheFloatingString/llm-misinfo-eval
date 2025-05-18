# For in-domain

# Gemini 2.0 flash

uv run src/eval_misinfo/cli.py --ds x-fact-in-domain --prov gemini --model gemini-2.0-flash --jsonl-filepath x-fact-in-domain-gemini-2.0-flash-mcq.jsonl --prompt-type mcq
uv run src/eval_misinfo/cli.py --ds x-fact-in-domain --prov gemini --model gemini-2.0-flash --jsonl-filepath x-fact-in-domain-gemini-2.0-flash-numerical.jsonl --prompt-type numerical


# Llama models

uv run src/eval_misinfo/cli.py --ds x-fact-in-domain --prov together --model meta-llama/Llama-3.3-70B-Instruct-Turbo --jsonl-filepath x-fact-in-domain-together-llama-3.3-70b-mcq.jsonl --prompt-type mcq
uv run src/eval_misinfo/cli.py --ds x-fact-in-domain --prov together --model meta-llama/Llama-3.3-70B-Instruct-Turbo --jsonl-filepath x-fact-in-domain-together-llama-3.3-70b-numerical.jsonl --prompt-type numerical
