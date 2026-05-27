# LLMGuardrails-finetune


## Idea

The goal of this project is to fine-tune a small, efficient BERT-like model specifically for prompt injection detection.

For prompt injection safety, Large Language Models (LLMs) cannot be fully trusted as judges; adversarial prompts can easily manipulate the evaluator's behavior. Training a specialized LoRA adapter on top of a lightweight base model (such as DeBERTa or RoBERTa) offers a robust and cost-effective alternative.

**fastino/gliguard-LLMGuardrails-300M** is a small (0.3B) multi-labels, multi-domains, tasks model:
- Prompts safety (and Text Classification)
- NER (Named Entity Extraction),
- Structured data extraction
- Etc,

Source model: [fastino/gliguard-LLMGuardrails-300M](https://huggingface.co/fastino/gliguard-LLMGuardrails-300M),
documentation: [GLiNER2-Tutorial](https://github.com/fastino-ai/GLiNER2/tree/main/tutorial)

We selected this model because it is relatively small (0.3B parameters), CPU/GPU friendly, and uses a modern architecture: **[GLiNER2](https://github.com/fastino-ai/GLiNER2)** (a Generalist Model for Named Entity Recognition), which is built on the bidirectional **[DeBERTa](https://huggingface.co/microsoft/deberta-v3-base)** encoder architecture.

It provides a 2048-token context window—more than sufficient for typical prompt-injection tests and multi-turn conversations—and the baseline model is already pre-trained for **prompt safety**. 

**License:** Apache 2.0

### Datasets

For training and validation, we aggregated three prominent prompt injection datasets. These records were deduplicated and standardized under a single unified `prompt_safety` classification task, for a total of **23,563 prompts**. Each source dataset is available on Hugging Face:

- [neuralchemy/Prompt-injection-dataset](https://huggingface.co/datasets/neuralchemy/Prompt-injection-dataset)
- [S-Labs/prompt-injection-dataset](https://huggingface.co/datasets/S-Labs/prompt-injection-dataset)
- [xTRam1/safe-guard-prompt-injection](https://huggingface.co/datasets/xTRam1/safe-guard-prompt-injection)

## Setup

Local setup for training the model

1. **Project**:
   - Install `uv`: `curl -LsSf https://astral-sh/uv/install.sh | sh`
   - Sync deps: `uv sync --all-extras`

2. Setup wandb:
   - Login:
     ```bash
     wandb login
     ```

## Running

- **Streamlit frontend**: `uv run streamlit run src/streamlit_app/app.py`

- **CLI dataset preparation**: `uv run python -m src.finetuner.data.prepare_data`

- **CLI training**: `uv run python -m src.finetuner.train`

- **CLI validation eval**: 
  - `uv run python -m src.finetuner.eval --validate --max-samples 100` # load adapters/final/ by default
  - `uv run python -m src.finetuner.eval --validate --adapter None --max-samples 100` # for base model

- **CLI single inference**: `uv run python -m src.finetuner.eval --prompt "Write a script to hack a database."`

## Results

On prompt classification "prompt_safety" task:

| Model                                            | Accuracy | F1 Score | Precision | Recall |
| ------------------------------------------------ | :------- | :------- | --------- | ------ |
| fastino/gliguard-LLMGuardrails-300M (base model) | 72.00%   | 51.70%   | 78.95%    | 38.46% |
| final adapter (in adapters/final/)               | 97.00%   | 96.30%   | 92.86%    | 100.0% |

### Training logs
https://wandb.ai/corentin-l/guardrail-finetune/runs/1fux5p6j

### Limits

![val_loss](doc/eval_loss.png)

- The **val loss** is still decreasing (1.68) after 2 epochs, we could potentially try to train on another epoch before over-fitting.
- The adapter model is a bit aggressive against short prompts:
   - We could add more harmless short prompts in the training dataset, but:
   - It is designed to be used as a tier-2 security layer, with a simpler tier-1 (for instance, a naive bayesian model) that filters out obvious prompts.
- We could benefit to use multi-turn prompts injections in the training dataset.
- It's a short experiment, we could potentially improve the results by tuning hyperparameters (epochs, learning rate, batch size, etc.)