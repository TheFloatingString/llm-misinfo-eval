# llm-misinfo-eval

Repository for using LLMs to evaluate misinformation in non-English datasets.

### Setup

This repository uses `uv` as a package manager:

```bash
pip install uv
uv sync
```

Ensure that all the environment variables are defined, either directly in the terminal, or in a `.env` file at the project's root folder:

```bash
OPENAI_API_KEY="<KEY>"
TOGETHER_API_KEY="<KEY>"
GEMINI_API_KEY="<KEY>"
```

### Experiments

Run the following for each experiment from the project root folder:

```bash
source ./scripts/<name of .sh file to run>
```

where each `.sh` shell file in `/scripts` contains the runtime arguments for an experiment of interest. 

### Analysis

Run the following from the project root folder:

```bash
uv run src/eval_misinfo/analyze.py --jsonl-filepath "<.jsonl filepath to file that contains experiment outputs>"
```

