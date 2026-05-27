"""CLI utility to manually trigger the dataset pipeline, preprocessing, and serialization."""

from src.finetuner.config import GlobalConfig
from src.finetuner.constants import LogType
from src.finetuner.core.tracker import TrainingTracker
from src.finetuner.data.dataset import DatasetPipeline


def print_stats(train_ds, test_ds, val_ds, tracker) -> None:
    """Prints clean distribution statistics for the processed datasets.

    Args:
        train_ds: The tokenized train dataset.
        test_ds: The tokenized test dataset.
        val_ds: The tokenized validation dataset.
        tracker: Active session logger tracker.
    """
    tracker.add_log("SYSTEM", "Analyzing class distribution statistics...")

    for split_name, ds in [("Train", train_ds), ("Test", test_ds), ("Validation", val_ds)]:
        df = ds.to_pandas()
        total = len(df)
        unsafe_count = int(df["label"].sum()) if "label" in df.columns else 0
        safe_count = total - unsafe_count

        print(f"\n📈 Split: {split_name}")
        print(f"  - Total Count: {total}")
        print(f"  - Safe Prompts: {safe_count} ({safe_count / total * 100:.1f}%)")
        print(f"  - Unsafe / Injections: {unsafe_count} ({unsafe_count / total * 100:.1f}%)")


def main() -> None:
    """Main execution function loading and preprocessing target datasets."""
    config = GlobalConfig(use_wandb=False)
    tracker = TrainingTracker(config)

    # 1. Initialize Pipeline
    pipeline = DatasetPipeline(config, tracker)
    train_ds, test_ds, val_ds = pipeline.prepare_dataset()

    tracker.add_log(
        LogType.RESPONSE,
        "Deduplicated dataset splits saved to: data/train.jsonl and data/valid.jsonl",
    )

    # 2. Print metrics distribution
    print_stats(train_ds, test_ds, val_ds, tracker)


if __name__ == "__main__":
    main()
