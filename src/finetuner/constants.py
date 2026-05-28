"""Global constants for the Guardrails finetuner."""

from enum import Enum

# Model Identifiers
BASE_MODEL_ID = "fastino/gliguard-LLMGuardrails-300M"

# Hugging Face Datasets
DATASET_NEURALCHEMY = "neuralchemy/Prompt-injection-dataset"
DATASET_SAFEGUARD = "xTRam1/safe-guard-prompt-injection"
DATASET_SLABS = "S-Labs/prompt-injection-dataset"

ALL_DATASETS = [
    DATASET_NEURALCHEMY,
    DATASET_SAFEGUARD,
    DATASET_SLABS,
]

# Tracking and Logging
DEFAULT_WANDB_PROJECT = "guardrail-finetune"
LOG_FORMAT = "%(asctime)s [%(levelname)s] %(name)s: %(message)s"

# Label mapping
LABEL_SAFE = 0
LABEL_UNSAFE = 1
LABEL_NAMES = {
    LABEL_SAFE: "safe",
    LABEL_UNSAFE: "unsafe",
}

# Classification configurations
CLASSIFICATION_TASK = "prompt_safety"
CLASSIFICATION_LABELS = [LABEL_NAMES[LABEL_SAFE], LABEL_NAMES[LABEL_UNSAFE]]

# Inference and Evaluation
EVAL_BATCH_SIZE = 8
DEFAULT_THRESHOLD = 0.9


class LogType(Enum):
    REQUEST = "REQUEST"
    RESPONSE = "RESPONSE"
    TOOL = "TOOL"
    TRAIN = "TRAIN"
    SYSTEM = "SYSTEM"
    ERROR = "ERROR"
    METRIC = "METRIC"


# Theme-harmonious color map for log categories rendered in the console component
LOG_CATEGORY_COLORS = {
    LogType.REQUEST: "#3A86FF",
    LogType.RESPONSE: "#38B000",
    LogType.TOOL: "#FFB703",
    LogType.TRAIN: "#FF006E",
    LogType.SYSTEM: "#8338EC",
    LogType.ERROR: "#FF3333",
    LogType.METRIC: "#00F5D4",
}

# Fallback color when category color mapping is unrecognized
LOG_DEFAULT_COLOR = "#8E9AA8"
