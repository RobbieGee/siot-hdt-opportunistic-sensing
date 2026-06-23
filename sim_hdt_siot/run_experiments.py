import os
import sys
import argparse
import tempfile
from pathlib import Path


def _bootstrap_vendor_path() -> None:
    repo_root = Path(__file__).resolve().parent.parent
    vendor_path = repo_root / ".vendor"
    if vendor_path.exists():
        sys.path.append(str(vendor_path))
    temp_root = Path(tempfile.gettempdir())
    os.environ.setdefault("MPLCONFIGDIR", str(temp_root / "siot_hdt_mplconfig"))
    os.environ.setdefault("XDG_CACHE_HOME", str(temp_root / "siot_hdt_cache"))


_bootstrap_vendor_path()

from sim_hdt_siot.config import ExperimentConfig
from sim_hdt_siot.simulation import run_experiment


def parse_args() -> ExperimentConfig:
    parser = argparse.ArgumentParser(description="Run the SIoT HDT simulator experiment suite.")
    parser.add_argument("--episodes", type=int, default=ExperimentConfig.episodes)
    parser.add_argument("--timesteps", type=int, default=ExperimentConfig.timesteps)
    parser.add_argument("--seed", type=int, default=None, help="Backward-compatible single-seed override.")
    parser.add_argument("--seeds", nargs="+", type=int, default=None, help="Deterministic seed list.")
    parser.add_argument("--max-hops", type=int, default=None, help="Backward-compatible baseline hop radius override.")
    parser.add_argument("--baseline-hop-radius", type=int, default=ExperimentConfig.baseline_hop_radius)
    parser.add_argument("--adaptive-hop-radius", type=int, default=ExperimentConfig.adaptive_hop_radius)
    parser.add_argument("--trigger-threshold", type=float, default=ExperimentConfig.trigger_threshold)
    parser.add_argument("--source-nodes", type=int, default=ExperimentConfig.source_node_count)
    parser.add_argument("--selection-cap", type=int, default=ExperimentConfig.default_selection_cap)
    parser.add_argument("--adaptive-selection-cap", type=int, default=ExperimentConfig.adaptive_selection_cap)
    parser.add_argument("--degraded-external-fraction", type=float, default=ExperimentConfig.degraded_external_fraction)
    parser.add_argument("--graph-density", choices=["sparse", "default", "dense"], default=ExperimentConfig.graph_density_mode)
    parser.add_argument("--no-sensitivity", action="store_true", help="Skip compact sensitivity analyses.")
    parser.add_argument("--no-scalability", action="store_true", help="Skip scalability analyses.")
    parser.add_argument("--budget-matched-episodes", type=int, default=ExperimentConfig.budget_matched_episodes)
    parser.add_argument("--budget-matched-timesteps", type=int, default=ExperimentConfig.budget_matched_timesteps)
    parser.add_argument("--budget-matched-seeds", nargs="+", type=int, default=list(ExperimentConfig.budget_matched_seeds))
    parser.add_argument("--discovery-mode-comparison-episodes", type=int, default=ExperimentConfig.discovery_mode_comparison_episodes)
    parser.add_argument("--discovery-mode-comparison-timesteps", type=int, default=ExperimentConfig.discovery_mode_comparison_timesteps)
    parser.add_argument(
        "--discovery-mode-comparison-seeds",
        nargs="+",
        type=int,
        default=list(ExperimentConfig.discovery_mode_comparison_seeds),
    )
    parser.add_argument("--sensitivity-episodes", type=int, default=ExperimentConfig.sensitivity_episodes)
    parser.add_argument("--sensitivity-timesteps", type=int, default=ExperimentConfig.sensitivity_timesteps)
    parser.add_argument("--sensitivity-seeds", nargs="+", type=int, default=list(ExperimentConfig.sensitivity_seeds))
    parser.add_argument("--scalability-episodes", type=int, default=ExperimentConfig.scalability_episodes)
    parser.add_argument("--scalability-timesteps", type=int, default=ExperimentConfig.scalability_timesteps)
    parser.add_argument("--scalability-seeds", nargs="+", type=int, default=list(ExperimentConfig.scalability_seeds))
    parser.add_argument("--scenarios", nargs="+", default=list(ExperimentConfig.scenarios))
    parser.add_argument("--policies", nargs="+", default=list(ExperimentConfig.policies))
    parser.add_argument("--output-dir", type=Path, default=ExperimentConfig.results_dir)
    args = parser.parse_args()
    seeds = tuple(args.seeds) if args.seeds is not None else ((args.seed,) if args.seed is not None else ExperimentConfig.seeds)
    baseline_hop_radius = args.max_hops if args.max_hops is not None else args.baseline_hop_radius
    return ExperimentConfig(
        seed=seeds[0],
        seeds=seeds,
        episodes=args.episodes,
        timesteps=args.timesteps,
        baseline_hop_radius=baseline_hop_radius,
        adaptive_hop_radius=args.adaptive_hop_radius,
        max_hops=baseline_hop_radius,
        trigger_threshold=args.trigger_threshold,
        source_node_count=args.source_nodes,
        default_selection_cap=args.selection_cap,
        adaptive_selection_cap=args.adaptive_selection_cap,
        degraded_external_fraction=args.degraded_external_fraction,
        graph_density_mode=args.graph_density,
        run_sensitivity=not args.no_sensitivity,
        run_scalability=not args.no_scalability,
        budget_matched_episodes=args.budget_matched_episodes,
        budget_matched_timesteps=args.budget_matched_timesteps,
        budget_matched_seeds=tuple(args.budget_matched_seeds),
        discovery_mode_comparison_episodes=args.discovery_mode_comparison_episodes,
        discovery_mode_comparison_timesteps=args.discovery_mode_comparison_timesteps,
        discovery_mode_comparison_seeds=tuple(args.discovery_mode_comparison_seeds),
        sensitivity_episodes=args.sensitivity_episodes,
        sensitivity_timesteps=args.sensitivity_timesteps,
        sensitivity_seeds=tuple(args.sensitivity_seeds),
        scalability_episodes=args.scalability_episodes,
        scalability_timesteps=args.scalability_timesteps,
        scalability_seeds=tuple(args.scalability_seeds),
        scenarios=tuple(args.scenarios),
        policies=tuple(args.policies),
        results_dir=args.output_dir,
    )


def main() -> None:
    outputs = run_experiment(parse_args())
    print("Simulation completed.")
    for name, path in outputs.items():
        print(f"{name}: {path}")


if __name__ == "__main__":
    main()
