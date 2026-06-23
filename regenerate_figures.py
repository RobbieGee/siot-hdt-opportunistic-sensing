import argparse
import os
import tempfile
from dataclasses import replace
from pathlib import Path

import pandas as pd

temp_root = Path(tempfile.gettempdir())
os.environ.setdefault("MPLCONFIGDIR", str(temp_root / "siot_hdt_mplconfig"))
os.environ.setdefault("XDG_CACHE_HOME", str(temp_root / "siot_hdt_cache"))

from sim_hdt_siot.config import ExperimentConfig
from sim_hdt_siot.simulation import build_figures


def _read_csv(data_dir: Path, filename: str) -> pd.DataFrame:
    path = data_dir / filename
    if not path.exists():
        raise FileNotFoundError(f"Required CSV not found: {path}")
    return pd.read_csv(path)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Regenerate manuscript figures from existing CSV outputs.")
    parser.add_argument("--output-dir", type=Path, default=Path("."), help="Repository/output root containing data/ and figures/.")
    parser.add_argument("--dpi", type=int, default=ExperimentConfig.plot_dpi)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = replace(ExperimentConfig(), results_dir=args.output_dir, plot_dpi=args.dpi)
    root_dir = config.results_dir
    data_dir = root_dir / config.data_dir_name
    figures_dir = root_dir / config.figures_dir_name
    figures_dir.mkdir(parents=True, exist_ok=True)

    outputs = build_figures(
        config,
        _read_csv(data_dir, config.metrics_name),
        _read_csv(data_dir, config.episode_summary_name),
        _read_csv(data_dir, config.robustness_name),
        figures_dir,
        _read_csv(data_dir, "convergence_cost_results.csv"),
        _read_csv(data_dir, "budget_matched_results.csv"),
        _read_csv(data_dir, "pareto_policy_results.csv"),
        _read_csv(data_dir, "cost_effectiveness_results.csv"),
        _read_csv(data_dir, "scalability_results.csv"),
        _read_csv(data_dir, "discovery_mode_comparison.csv"),
    )

    print("Figures regenerated from existing CSV outputs.")
    for name, path in outputs.items():
        print(f"{name}: {path}")


if __name__ == "__main__":
    main()
