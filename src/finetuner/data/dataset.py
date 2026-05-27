"""Dataset preprocessing and ingestion pipeline."""

import os
from typing import Any, Dict, List, Tuple, cast

import pandas as pd
from datasets import Dataset, load_dataset

from src.finetuner.config import GlobalConfig
from src.finetuner.constants import ALL_DATASETS, LABEL_SAFE, LABEL_UNSAFE, LogType
from src.finetuner.core.tracker import TrainingTracker

_HARMLESS_PRESETS: List[str] = [
    "say hi",
    "hello",
    "hi",
    "hi there",
    "hey",
    "yo",
    "good morning",
    "good afternoon",
    "good evening",
    "how are you?",
    "who are you?",
    "what is your name?",
    "can you help me?",
    "test",
    "whats up?",
    "tell me a joke",
    "what is 2+2?",
    "how's the weather today?",
    "thank you",
    "thanks",
    "bye",
    "see you later",
    "how is it going?",
    "help me write an email",
    "what is the capital of France?",
    "write a short poem about coding",
    "tell me a story",
    "what is machine learning?",
]


class DatasetPipeline:
    """Orchestrates dataset downloading, normalization, deduplication, and partitioning.

    This class provides robust merging of heterogeneous dataset columns into a
    single schema: {"text": str, "label": int}.

    Attributes:
        config (GlobalConfig): Target hyperparameters and split ratio configuration.
        tracker (TrainingTracker): Observability tracker.
    """

    def __init__(self, config: GlobalConfig, tracker: TrainingTracker) -> None:
        """Initializes the dataset pipeline.

        Args:
            config (GlobalConfig): Running configuration settings.
            tracker (TrainingTracker): Observability tracker.
        """
        self.config: GlobalConfig = config
        self.tracker: TrainingTracker = tracker

    def prepare_dataset(self) -> Tuple[Dataset, Dataset, Dataset]:
        """Downloads, standardizes, deduplicates, and splits the target datasets.

        Returns:
            Tuple[Dataset, Dataset, Dataset]: train_ds, test_ds, val_ds.
        """
        self.tracker.add_log(LogType.REQUEST, "Starting dataset preparation and cleaning.")
        combined = self._load_combined_records()
        df_cleaned = self._clean_and_deduplicate(pd.DataFrame(combined))

        os.makedirs("data", exist_ok=True)

        full_ds = Dataset.from_pandas(df_cleaned.reset_index(drop=True))
        raw_splits = full_ds.train_test_split(test_size=0.1, seed=self.config.seed)

        train_ds, test_ds, val_ds = self._cache_and_split_dataset(
            cast(Dataset, raw_splits["train"]), cast(Dataset, raw_splits["test"])
        )

        self.tracker.add_log(
            LogType.RESPONSE,
            f"Dataset splits created: {len(train_ds)} train, "
            f"{len(test_ds)} test, {len(val_ds)} validation.",
        )
        return train_ds, test_ds, val_ds

    def load_from_cache(self, train_path: str, val_path: str) -> Tuple[Dataset, Dataset, Dataset]:
        """Loads pre-built JSONL splits from local disk, skipping HuggingFace downloads.

        Args:
            train_path (str): Path to the training JSONL file (e.g. 'data/train.jsonl').
            val_path (str): Path to the validation JSONL file (e.g. 'data/valid.jsonl').

        Returns:
            Tuple[Dataset, Dataset, Dataset]: train_ds, test_ds, val_ds.
        """
        self.tracker.add_log(LogType.TOOL, f"Loading cached datasets: {train_path}, {val_path}")
        raw_train_ds = load_dataset("json", data_files=train_path, split="train")  # nosec B615
        val_ds = load_dataset("json", data_files=val_path, split="train")  # nosec B615

        # Split cached train pool into train_ds and test_ds based on config.test_split
        train_test_splits = raw_train_ds.train_test_split(
            test_size=self.config.test_split, seed=self.config.seed
        )
        train_ds = cast(Dataset, train_test_splits["train"])
        test_ds = cast(Dataset, train_test_splits["test"])

        self.tracker.add_log(
            LogType.RESPONSE,
            f"Loaded from cache and split: {len(train_ds)} train, "
            f"{len(test_ds)} test, {len(val_ds)} validation rows.",
        )
        return train_ds, test_ds, val_ds

    def _inject_harmless_presets(self, records: List[Dict[str, Any]]) -> None:
        """Injects a diverse set of simple, harmless conversational prompts as safe (label 0).

        High level role: Negative sampling data augmentor. Appends common everyday
        safe chatbot inputs to the dataset list to prevent classification over-sensitivity.

        Args:
            records (List[Dict[str, Any]]): Accumulator list of dataset records.
        """
        self.tracker.add_log(
            LogType.TOOL,
            f"Injecting {len(_HARMLESS_PRESETS)} harmless "
            "conversational presets as safe baseline examples.",
        )
        for prompt in _HARMLESS_PRESETS:
            records.append({"text": prompt, "label": 0})

    def _load_and_standardize_dataset(self, dataset_id: str, records: List[Dict[str, Any]]) -> None:
        """Downloads and extracts normalized rows from a Hugging Face dataset.

        Args:
            dataset_id (str): Hugging Face dataset ID.
            records (List[Dict[str, Any]]): In-out accumulator of records.
        """
        try:
            self.tracker.add_log(LogType.TOOL, f"Loading dataset from HF: {dataset_id}")
            # Load default split
            ds = load_dataset(dataset_id, split="train")  # nosec B615
            text_col, label_col = self._detect_columns(ds.column_names)

            for row in ds:
                row_dict = cast(Dict[str, Any], row)
                text_val = str(row_dict[text_col]).strip()
                # Parse labels securely
                raw_label = row_dict[label_col]
                label_val = self._normalize_label(raw_label)
                records.append({"text": text_val, "label": label_val})

            self.tracker.add_log(LogType.RESPONSE, f"Standardized {len(ds)} rows from {dataset_id}")
        except Exception as err:
            self.tracker.add_log(
                LogType.ERROR, f"Failed to ingest dataset {dataset_id}: {str(err)}"
            )

    def _detect_columns(self, column_names: List[str]) -> Tuple[str, str]:
        """Identifies text and label column names from dataset schema.

        Args:
            column_names (List[str]): Columns available in dataset.

        Returns:
            Tuple[str, str]: Identified (text_column, label_column).
        """
        text_candidates = ["prompt", "text", "input", "instruction", "sentence"]
        label_candidates = ["label", "class", "injection", "safe", "target", "output"]

        text_col = next((c for c in column_names if c in text_candidates), column_names[0])
        label_col = next((c for c in column_names if c in label_candidates), column_names[-1])
        return text_col, label_col

    def _normalize_label(self, raw_label: Any) -> int:
        """Standardizes dynamic label formats into safe (0) / unsafe (1) integers.

        Args:
            raw_label (Any): String or integer label value.

        Returns:
            int: 0 (safe) or 1 (unsafe).
        """
        if isinstance(raw_label, bool):
            return LABEL_UNSAFE if raw_label else LABEL_SAFE
        label_str = str(raw_label).lower().strip()
        if label_str in ["1", "true", "unsafe", "injection", "flagged", "yes"]:
            return LABEL_UNSAFE
        return LABEL_SAFE

    def _clean_and_deduplicate(self, df: pd.DataFrame) -> pd.DataFrame:
        """Removes duplicate text entries and null values from dataset.

        Args:
            df (pd.DataFrame): Dataframe of records.

        Returns:
            pd.DataFrame: Deduplicated dataframe.
        """
        if df.empty:
            self.tracker.add_log(LogType.ERROR, "Dataset dataframe is empty.")
            return pd.DataFrame(columns=["text", "label"])

        before = len(df)
        df = df.dropna(subset=["text"])
        df = df.drop_duplicates(subset=["text"])
        self.tracker.add_log(
            LogType.RESPONSE, f"Deduplication complete. Retained {len(df)} of {before} rows."
        )
        return df

    def _load_combined_records(self) -> List[Dict[str, Any]]:
        """Downloads HF datasets and injects harmless safe presets.

        High level role: Dataset downloader and baseline presets aggregator.

        Returns:
            List[Dict[str, Any]]: Combined standardized record dictionaries.
        """
        combined_records: List[Dict[str, Any]] = []
        for dataset_id in ALL_DATASETS:
            self._load_and_standardize_dataset(dataset_id, combined_records)
        self._inject_harmless_presets(combined_records)
        return combined_records

    def _cache_and_split_dataset(
        self, raw_train_ds: Dataset, val_ds: Dataset
    ) -> Tuple[Dataset, Dataset, Dataset]:
        """Caches training pool and splits it into train/test sets.

        High level role: Dataset caching and partition orchestrator.

        Args:
            raw_train_ds (Dataset): Raw training pool.
            val_ds (Dataset): Validation dataset.

        Returns:
            Tuple[Dataset, Dataset, Dataset]: train_ds, test_ds, val_ds.
        """
        raw_train_ds.to_json("data/train.jsonl")
        val_ds.to_json("data/valid.jsonl")

        train_test_splits = raw_train_ds.train_test_split(
            test_size=self.config.test_split, seed=self.config.seed
        )
        train_ds = cast(Dataset, train_test_splits["train"])
        test_ds = cast(Dataset, train_test_splits["test"])
        return train_ds, test_ds, val_ds
