# llm-misinfo-eval

Repository for using LLMs to evaluate misinformation in non-English datasets.

### Setup

In the root directory, run the following command to install dependencies. 

```bash
pip install -r requirements.txt
```

Please include the `.env` file in the project root folder (the same folder as the `requirements.txt` file)

One example with OpenAI's o3-mini would be:

```bash
python scripts/run_zero_shot_model.py --ds x-fact --prov openai --model o4-mini --jsonl-filepath o4-mini-reuslts.jsonl
```

One example with Together AI (which is preferable due to model speed and model offerings) using `meta-llama/Meta-Llama-3.1-8B-Instruct-Turbo` would be:

```bash
python scripts/run_zero_shot_model.py --ds x-fact --prov together --model meta-llama/Meta-Llama-3.1-8B-Instruct-Turbo --jsonl-filepath together-llama-3.1-8b-reuslts.jsonl
```

Please see the following link for models offered on Together AI:

https://www.together.ai/models

The next step would then be to caclculate averages from the f1 scores in the jsonl after the run is complete. 




### Running Evaluations (2025 Update)

```
python scripts/run_zero_shot_model.py --
```


### Notebooks

* `notebooks/load_dataset.ipynb`: Sample code to load the x-fact and mumin datasets
* `notebooks/openai_example_code.ipynb`: Example OpenAI code
* `notebooks/cohere_example_code.ipynb`: Example Cohere code
