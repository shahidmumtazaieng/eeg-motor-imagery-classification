"""Command-line entry point for the EEG motor-imagery experiments."""

import argparse
from pathlib import Path

import pandas as pd

from .data import load_dataset
from .evaluation import (
    evaluate_subject_dependent,
    evaluate_subject_independent,
    summarize_results,
)


def run_experiment(
    subjects: list[int],
    data_dir: str | Path | None = None,
    output_dir: str | Path = "results",
    include_eegnet: bool = True,
    eegnet_epochs: int = 30,
):
    if len(set(subjects)) < 2:
        raise ValueError("Run both evaluation designs with at least two subjects.")
    data, labels, groups, channel_names, sampling_frequency = load_dataset(
        subjects, data_dir=data_dir
    )
    dependent = evaluate_subject_dependent(
        data, labels, groups, channel_names, sampling_frequency
    )
    independent = evaluate_subject_independent(
        data,
        labels,
        groups,
        channel_names,
        sampling_frequency,
        include_eegnet=include_eegnet,
        eegnet_epochs=eegnet_epochs,
    )
    results = pd.concat([dependent, independent], ignore_index=True)

    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    results.to_csv(output_path / "model_comparison.csv", index=False)
    summarize_results(results).to_csv(output_path / "model_summary.csv")
    results.to_json(output_path / "fold_metrics.json", orient="records", indent=2)
    return results


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--subjects",
        type=int,
        nargs="+",
        default=[1, 2, 3, 4, 5],
        help="EEGBCI participant IDs (1-109); subject-independent CV needs 2+.",
    )
    parser.add_argument("--data-dir", type=Path, default=None)
    parser.add_argument("--output-dir", type=Path, default=Path("results"))
    parser.add_argument("--skip-eegnet", action="store_true")
    parser.add_argument("--eegnet-epochs", type=int, default=30)
    args = parser.parse_args()

    results = run_experiment(
        args.subjects,
        data_dir=args.data_dir,
        output_dir=args.output_dir,
        include_eegnet=not args.skip_eegnet,
        eegnet_epochs=args.eegnet_epochs,
    )
    print(results.groupby(["evaluation", "model"])[
        ["accuracy", "balanced_accuracy", "macro_f1", "roc_auc"]
    ].mean().round(3))
    print(f"\nSaved fold-level results to {args.output_dir}")


if __name__ == "__main__":
    main()
