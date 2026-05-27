"""CLI entry point for running safety classifier inference and validation suite."""

import argparse
import os

from src.finetuner.config import GlobalConfig
from src.finetuner.constants import BASE_MODEL_ID, DEFAULT_THRESHOLD
from src.finetuner.core.inference import InferenceEngine
from src.finetuner.core.metrics import MetricsCalculator
from src.finetuner.core.tracker import TrainingTracker


def _parse_args() -> argparse.Namespace:
    """Parses command-line arguments for safety inference.

    High level role: Command-line argument parser.
    Description: Configures and triggers argparse to capture arguments for
    evaluating prompts and running dataset validations from the terminal.

    Args:
        None.

    Returns:
        argparse.Namespace: Parsed terminal arguments.

    Raises:
        None.
    """
    parser = argparse.ArgumentParser(description="GLIGuard Safety Inference CLI")
    parser.add_argument("--model-id", type=str, default=BASE_MODEL_ID, help="Model ID")
    parser.add_argument(
        "--adapter",
        type=str,
        default="adapters/final",
        help="Adapter path (adapters/xxx). 'None' uses base model",
    )
    parser.add_argument("--prompt", type=str, help="Single prompt to evaluate")
    parser.add_argument("--threshold", type=float, default=DEFAULT_THRESHOLD, help="Threshold")
    parser.add_argument("--validate", action="store_true", help="Run batch validation")
    parser.add_argument("--max-samples", type=int, default=100, help="Max validation samples")
    return parser.parse_args()


def _run_single_inference(
    engine: InferenceEngine, model_id: str, prompt: str, threshold: float
) -> None:
    """Evaluates a single prompt and outputs the verdict to the console.

    High level role: Terminal prompt evaluator.
    Description: Executes safety classification for a single prompt input
    via the inference engine, displaying the safe/unsafe verdict.

    Args:
        engine (InferenceEngine): Active inference engine.
        model_id (str): Target model or adapter ID.
        prompt (str): Text prompt to check.
        threshold (float): Threshold classification level.

    Returns:
        None.

    Raises:
        Exception: If inference fails.

    Examples:
        >>> _run_single_inference(engine, "adapters/best", "Hello", 0.9)
    """
    res = engine.evaluate_text(model_id, prompt, threshold=threshold)
    print(f"\nVerdict: {res['verdict'].upper()}")
    print(f"Unsafe Score: {res['score']:.4f} (Threshold: {threshold})")


def _run_dataset_validation(
    engine: InferenceEngine, model_id: str, max_samples: int, threshold: float
) -> None:
    """Loads the validation dataset and executes batch safety evaluation.

    High level role: Batch validation orchestrator.
    Description: Parses the evaluation dataset, loads the optimized model,
    executes batch classification, and prints performance metrics.

    Args:
        engine (InferenceEngine): Active inference engine.
        model_id (str): Target model or adapter ID.
        max_samples (int): Limit on loaded sample size.
        threshold (float): Unsafe classification threshold.

    Returns:
        None.

    Raises:
        FileNotFoundError: If the validation dataset is missing.
        Exception: If batch inference fails.

    Examples:
        >>> _run_dataset_validation(engine, "base-model", 50, 0.9)
    """
    records = engine.load_validation_records(max_samples)
    model = engine.get_model(model_id)
    print(f"\nRunning batch validation on {len(records)} samples...")
    verdicts = engine.run_batched_inference(model, records)
    print(f"Completed processing {len(verdicts)} validation samples.")

    probs = [engine.get_unsafe_probability(v) for v in verdicts]
    calc = MetricsCalculator()
    metrics = calc.compute_metrics(records, probs, threshold=threshold)

    print("\n============================================================")
    print("📈 Evaluation Metrics")
    print("============================================================")
    print(f"Accuracy  : {metrics['accuracy']:.4f}")
    print(f"Precision : {metrics['precision']:.4f}")
    print(f"Recall    : {metrics['recall']:.4f}")
    print(f"F1 Score  : {metrics['f1_score']:.4f}")
    print(f"Total Ev  : {metrics['total_evaluated']}")
    print(f"TP/FP/TN/FN: {metrics['tp']}/{metrics['fp']}/{metrics['tn']}/{metrics['fn']}")
    print("============================================================")


def main() -> None:
    """CLI entry point for executing safety classifier inference and validation.

    High level role: CLI main launcher.
    Description: Instantiates standard configurations, registers a training tracker,
    and routes input arguments to single prompt or validation execution routines.

    Args:
        None.

    Returns:
        None.

    Raises:
        ValueError: If neither a prompt nor validation is requested.

    Examples:
        >>> main()
    """
    args = _parse_args()
    if not args.prompt and not args.validate:
        raise ValueError("Must specify either --prompt or --validate.")

    target_model = args.model_id
    if args.adapter and args.adapter != "None":
        target_model = (
            f"adapters/{args.adapter}"
            if not os.path.exists(args.adapter) and os.path.exists(f"adapters/{args.adapter}")
            else args.adapter
        )

    config = GlobalConfig(model_id=target_model, use_wandb=False)
    tracker = TrainingTracker(config)
    engine = InferenceEngine(tracker)

    if args.prompt:
        _run_single_inference(engine, target_model, args.prompt, args.threshold)
    elif args.validate:
        _run_dataset_validation(engine, target_model, args.max_samples, args.threshold)


if __name__ == "__main__":
    main()
