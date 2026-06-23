from pathlib import Path

import matplotlib
import numpy as np
import pandas as pd

from sim_hdt_siot.config import (
    BUDGET_MATCHED_POLICIES,
    CONTEXT_VARIABLES,
    DEFAULT_POLICIES,
    DEFAULT_SCENARIOS,
    DISCOVERY_MODE_COMPARISON_POLICIES,
    POLICY_LABELS_SHORT,
    SCENARIO_LABELS,
)


matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D


POLICY_COLORS = {
    "ego_only": "#374151",
    "opportunistic_all": "#D97706",
    "siot_aware": "#0F766E",
    "siot_aware_trust_privacy": "#1D4ED8",
    "random_k": "#7C3AED",
    "budgeted_opportunistic_k": "#B45309",
    "siot_aware_exhaustive_discovery": "#115E59",
    "siot_aware_bounded_discovery": "#14B8A6",
    "siot_aware_trust_privacy_exhaustive_discovery": "#1E40AF",
    "siot_aware_trust_privacy_bounded_discovery": "#60A5FA",
}

FIGURE_SIZE = (7.1, 6.2)
SUBPLOT_TITLE_SIZE = 10.5
TITLE_SIZE = 12.5
LABEL_SIZE = 10.5
TICK_SIZE = 9.5
LEGEND_SIZE = 9.5
VARIABLE_LABELS = {
    "place": "Place",
    "activity": "Activity",
    "env_load": "Env. load",
    "resource_state": "Resource state",
}
SCENARIO_TITLES = SCENARIO_LABELS

def _ordered_metric_frame(frame: pd.DataFrame) -> pd.DataFrame:
    ordered = frame.copy()
    if "scenario" in ordered.columns:
        ordered["scenario"] = pd.Categorical(ordered["scenario"], categories=list(DEFAULT_SCENARIOS), ordered=True)
    if "policy" in ordered.columns:
        policy_order = list(dict.fromkeys(list(DEFAULT_POLICIES) + list(BUDGET_MATCHED_POLICIES) + list(DISCOVERY_MODE_COMPARISON_POLICIES)))
        ordered["policy"] = pd.Categorical(ordered["policy"], categories=policy_order, ordered=True)
    return ordered.sort_values([column for column in ["scenario", "policy"] if column in ordered.columns])


def _style_axes(ax: plt.Axes) -> None:
    ax.grid(axis="y", linestyle="--", alpha=0.25, linewidth=0.8)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.tick_params(axis="both", labelsize=TICK_SIZE)


def _policy_label(policy: str) -> str:
    return POLICY_LABELS_SHORT.get(policy, policy)


def _scenario_label(scenario: str) -> str:
    return SCENARIO_TITLES.get(scenario, scenario.replace("_", " ").title())


def _legend_for_policies(policies: list[str]) -> tuple[list[plt.Rectangle], list[str]]:
    handles = [plt.Rectangle((0, 0), 1, 1, color=POLICY_COLORS[policy]) for policy in policies]
    labels = [_policy_label(policy) for policy in policies]
    return handles, labels


def _bootstrap_confidence_interval(values: np.ndarray, seed: int = 7, samples: int = 2000) -> tuple[float, float]:
    if len(values) == 0:
        return float("nan"), float("nan")
    if len(values) == 1:
        return float(values[0]), float(values[0])
    rng = np.random.default_rng(seed)
    sampled_means = np.empty(samples)
    for index in range(samples):
        sampled_means[index] = rng.choice(values, size=len(values), replace=True).mean()
    low, high = np.quantile(sampled_means, [0.025, 0.975])
    return float(low), float(high)


def _missingness_midpoint(label: str) -> float:
    cleaned = label.strip().strip("[]()")
    lower_text, upper_text = [part.strip() for part in cleaned.split(",")]
    return (float(lower_text) + float(upper_text)) / 2.0


def plot_overall_accuracy(
    metrics_df: pd.DataFrame,
    output_path: Path,
    dpi: int,
    episode_summary_df: pd.DataFrame | None = None,
) -> None:
    metrics_df = _ordered_metric_frame(metrics_df)
    scenarios = [scenario for scenario in DEFAULT_SCENARIOS if scenario in set(metrics_df["scenario"].astype(str))]
    episode_summary_df = _ordered_metric_frame(episode_summary_df) if episode_summary_df is not None else None
    fig, axes = plt.subplots(2, 2, figsize=FIGURE_SIZE, sharex=False, sharey=True)
    axes_list = list(axes.flatten())
    all_values: list[float] = []
    for scenario in scenarios:
        subset = metrics_df[metrics_df["scenario"].astype(str) == scenario]
        policies = [policy for policy in DEFAULT_POLICIES if policy in set(subset["policy"].astype(str))]
        for policy in policies:
            value = float(subset.loc[subset["policy"].astype(str) == policy, "overall_context_accuracy"].iloc[0])
            all_values.append(value)
            if episode_summary_df is not None:
                series = episode_summary_df[
                    (episode_summary_df["scenario"].astype(str) == scenario) & (episode_summary_df["policy"].astype(str) == policy)
                ]["overall_context_accuracy"].to_numpy(dtype=float)
                low, high = _bootstrap_confidence_interval(series, seed=7 + len(all_values))
                all_values.extend([low, high])
    y_upper = min(1.0, max(all_values) + 0.04) if all_values else 1.0

    for ax, scenario in zip(axes_list, scenarios):
        subset = metrics_df[metrics_df["scenario"].astype(str) == scenario]
        policies = [policy for policy in DEFAULT_POLICIES if policy in set(subset["policy"].astype(str))]
        x_positions = np.arange(len(policies))
        values = []
        errors_low = []
        errors_high = []
        for policy in policies:
            value = float(subset.loc[subset["policy"].astype(str) == policy, "overall_context_accuracy"].iloc[0])
            values.append(value)
            if episode_summary_df is not None:
                series = episode_summary_df[
                    (episode_summary_df["scenario"].astype(str) == scenario) & (episode_summary_df["policy"].astype(str) == policy)
                ]["overall_context_accuracy"].to_numpy(dtype=float)
                low, high = _bootstrap_confidence_interval(series, seed=11 + len(values))
                errors_low.append(max(0.0, value - low))
                errors_high.append(max(0.0, high - value))
            else:
                errors_low.append(0.0)
                errors_high.append(0.0)
        ax.bar(
            x_positions,
            values,
            color=[POLICY_COLORS[policy] for policy in policies],
            width=0.72,
            edgecolor="none",
            yerr=np.vstack([errors_low, errors_high]),
            error_kw={"elinewidth": 0.9, "capsize": 2.5, "capthick": 0.9, "ecolor": "#111827"},
        )
        ax.set_title(_scenario_label(scenario), fontsize=SUBPLOT_TITLE_SIZE, pad=4)
        ax.set_xticks(x_positions, [_policy_label(policy) for policy in policies], rotation=0)
        ax.set_ylim(0.0, y_upper)
        _style_axes(ax)
    for ax in axes[:, 0]:
        ax.set_ylabel("OCA", fontsize=LABEL_SIZE)
    for ax in axes[1, :]:
        ax.set_xlabel("Policy", fontsize=LABEL_SIZE)
    handles, labels = _legend_for_policies(list(DEFAULT_POLICIES))
    fig.legend(handles, labels, loc="upper center", ncol=4, frameon=False, fontsize=LEGEND_SIZE, bbox_to_anchor=(0.5, 1.01))
    fig.subplots_adjust(left=0.10, right=0.99, bottom=0.10, top=0.87, wspace=0.20, hspace=0.26)
    fig.savefig(output_path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)


def plot_macro_f1_by_variable(metrics_df: pd.DataFrame, output_path: Path, dpi: int) -> None:
    metrics_df = _ordered_metric_frame(metrics_df)
    scenarios = [scenario for scenario in DEFAULT_SCENARIOS if scenario in set(metrics_df["scenario"].astype(str))]
    fig, axes = plt.subplots(2, 2, figsize=(10.8, 8.0), sharey=True)
    axes_list = list(axes.flatten())
    variables = list(CONTEXT_VARIABLES.keys())
    x_positions = range(len(scenarios))
    width = 0.18
    for ax, variable in zip(axes_list, variables):
        for offset_idx, policy in enumerate(DEFAULT_POLICIES):
            values = []
            for scenario in scenarios:
                subset = metrics_df[
                    (metrics_df["scenario"].astype(str) == scenario) & (metrics_df["policy"].astype(str) == policy)
                ]
                values.append(float(subset[f"macro_f1_{variable}"].iloc[0]) if not subset.empty else 0.0)
            shifted_positions = [x + ((offset_idx - 1.5) * width) for x in x_positions]
            ax.bar(shifted_positions, values, width=width, color=POLICY_COLORS[policy], label=_policy_label(policy))
        ax.set_title(VARIABLE_LABELS[variable], fontsize=SUBPLOT_TITLE_SIZE)
        ax.set_xticks(list(x_positions), [_scenario_label(scenario).replace(" ", "\n") for scenario in scenarios])
        ax.set_ylim(0.0, 1.0)
        ax.set_ylabel("Macro-F1", fontsize=LABEL_SIZE)
        _style_axes(ax)
    handles, labels = _legend_for_policies(list(DEFAULT_POLICIES))
    fig.legend(handles, labels, loc="upper center", ncol=4, frameon=False, fontsize=LEGEND_SIZE, bbox_to_anchor=(0.5, 1.02))
    fig.subplots_adjust(left=0.08, right=0.99, bottom=0.08, top=0.90, wspace=0.18, hspace=0.28)
    fig.savefig(output_path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)


def plot_accuracy_vs_missing_ego(robustness_df: pd.DataFrame, output_path: Path, dpi: int) -> None:
    robustness_df = _ordered_metric_frame(robustness_df)
    scenarios = [scenario for scenario in DEFAULT_SCENARIOS if scenario in set(robustness_df["scenario"].astype(str))]
    fig, axes = plt.subplots(2, 2, figsize=FIGURE_SIZE, sharex=True, sharey=True)
    axes_list = list(axes.flatten())
    x_range = (0.0, 1.0)
    y_max = min(1.0, float(robustness_df["overall_context_accuracy"].max()) + 0.06)
    for ax, scenario in zip(axes_list, scenarios):
        scenario_df = robustness_df[robustness_df["scenario"].astype(str) == scenario]
        for policy in DEFAULT_POLICIES:
            subset = scenario_df[scenario_df["policy"].astype(str) == policy].sort_values("missingness_bin")
            if subset.empty:
                continue
            x_values = [_missingness_midpoint(label) for label in subset["missingness_bin"].tolist()]
            ax.plot(
                x_values,
                subset["overall_context_accuracy"],
                marker="o",
                markersize=4.2,
                linewidth=1.9,
                color=POLICY_COLORS.get(policy, "#111827"),
                label=_policy_label(policy),
            )
        ax.set_xlim(*x_range)
        ax.set_xticks([0.0, 0.25, 0.50, 0.75, 1.0])
        ax.set_ylim(0.0, y_max)
        ax.set_title(_scenario_label(scenario), fontsize=SUBPLOT_TITLE_SIZE, pad=4)
        _style_axes(ax)
    for ax in axes[:, 0]:
        ax.set_ylabel("OCA", fontsize=LABEL_SIZE)
    for ax in axes[1, :]:
        ax.set_xlabel("Ego Missing Rate", fontsize=LABEL_SIZE)
    handles = [Line2D([0], [0], color=POLICY_COLORS[policy], marker="o", linewidth=2.0) for policy in DEFAULT_POLICIES]
    labels = [_policy_label(policy) for policy in DEFAULT_POLICIES]
    fig.legend(handles, labels, loc="upper center", ncol=4, frameon=False, fontsize=LEGEND_SIZE, bbox_to_anchor=(0.5, 1.01))
    fig.subplots_adjust(left=0.10, right=0.99, bottom=0.10, top=0.87, wspace=0.20, hspace=0.26)
    fig.savefig(output_path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)


def plot_discovery_overhead(metrics_df: pd.DataFrame, output_path: Path, dpi: int) -> None:
    summary = _ordered_metric_frame(metrics_df).groupby("policy", as_index=False, observed=False).agg(
        avg_estimated_latency=("avg_estimated_latency", "mean"),
        avg_nodes_visited=("avg_nodes_visited", "mean"),
        avg_selected_source_count=("avg_selected_source_count", "mean"),
    )
    policies = [policy for policy in DEFAULT_POLICIES if policy in set(summary["policy"].astype(str))]
    summary = summary.set_index("policy").loc[policies].reset_index()
    fig, axes = plt.subplots(1, 3, figsize=(12.2, 4.6))
    axes[0].bar([_policy_label(policy) for policy in policies], summary["avg_nodes_visited"], color=[POLICY_COLORS[p] for p in policies])
    axes[0].set_title("Nodes Visited", fontsize=SUBPLOT_TITLE_SIZE)
    axes[0].set_ylabel("Average Nodes", fontsize=LABEL_SIZE)
    _style_axes(axes[0])
    axes[1].bar([_policy_label(policy) for policy in policies], summary["avg_selected_source_count"], color=[POLICY_COLORS[p] for p in policies])
    axes[1].set_title("Selected Sources", fontsize=SUBPLOT_TITLE_SIZE)
    axes[1].set_ylabel("Average Count", fontsize=LABEL_SIZE)
    _style_axes(axes[1])
    axes[2].bar([_policy_label(policy) for policy in policies], summary["avg_estimated_latency"], color=[POLICY_COLORS[p] for p in policies])
    axes[2].set_title("Estimated Latency", fontsize=SUBPLOT_TITLE_SIZE)
    axes[2].set_ylabel("Average Latency", fontsize=LABEL_SIZE)
    _style_axes(axes[2])
    fig.suptitle("Discovery Overhead by Policy", fontsize=TITLE_SIZE, y=1.03)
    fig.tight_layout()
    fig.savefig(output_path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)


def plot_privacy_exposure(metrics_df: pd.DataFrame, output_path: Path, dpi: int) -> None:
    summary = _ordered_metric_frame(metrics_df).groupby("policy", as_index=False, observed=False)["avg_privacy_exposure_cost"].mean()
    policies = [policy for policy in DEFAULT_POLICIES if policy in set(summary["policy"].astype(str))]
    summary = summary.set_index("policy").loc[policies].reset_index()
    fig, ax = plt.subplots(figsize=FIGURE_SIZE)
    ax.bar([_policy_label(policy) for policy in policies], summary["avg_privacy_exposure_cost"], color=[POLICY_COLORS[p] for p in policies])
    ax.set_title("Privacy Exposure by Policy", fontsize=TITLE_SIZE)
    ax.set_ylabel("Privacy Exposure Cost", fontsize=LABEL_SIZE)
    _style_axes(ax)
    fig.tight_layout()
    fig.savefig(output_path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)


def plot_convergence_time(metrics_df: pd.DataFrame, output_path: Path, dpi: int) -> None:
    metrics_df = _ordered_metric_frame(metrics_df)
    scenarios = [scenario for scenario in DEFAULT_SCENARIOS if scenario in set(metrics_df["scenario"].astype(str))]
    policies = [policy for policy in DEFAULT_POLICIES if policy in set(metrics_df["policy"].astype(str))]
    width = 0.18
    x_positions = list(range(len(scenarios)))
    fig, ax = plt.subplots(figsize=(9.2, 4.8))
    for offset_idx, policy in enumerate(policies):
        values = []
        for scenario in scenarios:
            subset = metrics_df[
                (metrics_df["scenario"].astype(str) == scenario) & (metrics_df["policy"].astype(str) == policy)
            ]
            values.append(float(subset["convergence_time"].iloc[0]) if not subset.empty else 0.0)
        shifted_positions = [x + ((offset_idx - 1.5) * width) for x in x_positions]
        ax.bar(shifted_positions, values, width=width, color=POLICY_COLORS[policy], label=_policy_label(policy))
    ax.set_xticks(x_positions, [_scenario_label(scenario).replace(" ", "\n") for scenario in scenarios])
    ax.set_title("Convergence Time by Policy and Scenario", fontsize=TITLE_SIZE)
    ax.set_ylabel("Average Steps After Transition", fontsize=LABEL_SIZE)
    ax.legend(frameon=False, ncol=4, labels=[_policy_label(policy) for policy in policies], fontsize=LEGEND_SIZE)
    _style_axes(ax)
    fig.tight_layout()
    fig.savefig(output_path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)


def plot_effective_recovery_cost(frame: pd.DataFrame, output_path: Path, dpi: int) -> None:
    frame = _ordered_metric_frame(frame)
    _plot_grouped_bars(
        frame,
        "effective_recovery_cost",
        "Effective Recovery Cost",
        output_path,
        dpi,
        ylabel="Cost",
    )


def plot_budget_matched_oca_cost(frame: pd.DataFrame, output_path: Path, dpi: int) -> None:
    frame = _ordered_metric_frame(frame)
    fig, axes = plt.subplots(2, 2, figsize=(9.4, 7.2), sharex=False, sharey=True)
    scenarios = [scenario for scenario in DEFAULT_SCENARIOS if scenario in set(frame["scenario"].astype(str))]
    policies = [policy for policy in BUDGET_MATCHED_POLICIES if policy in set(frame["policy"].astype(str))]
    for ax, scenario in zip(axes.flatten(), scenarios):
        subset = frame[frame["scenario"].astype(str) == scenario]
        for policy in policies:
            policy_df = subset[subset["policy"].astype(str) == policy]
            if policy_df.empty:
                continue
            ax.scatter(
                policy_df["mean_total_operational_cost"],
                policy_df["OCA"],
                s=70,
                color=POLICY_COLORS.get(str(policy), "#111827"),
                label=_policy_label(str(policy)),
                edgecolor="white",
                linewidth=0.8,
            )
        ax.set_title(_scenario_label(scenario), fontsize=SUBPLOT_TITLE_SIZE)
        ax.set_xlabel("Total operational cost", fontsize=LABEL_SIZE)
        ax.set_ylabel("OCA", fontsize=LABEL_SIZE)
        _style_axes(ax)
    handles, labels = axes.flatten()[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=4, frameon=False, fontsize=LEGEND_SIZE)
    fig.subplots_adjust(left=0.10, right=0.99, bottom=0.08, top=0.86, wspace=0.24, hspace=0.30)
    fig.savefig(output_path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)


def plot_discovery_mode_oca_cost(frame: pd.DataFrame, output_path: Path, dpi: int) -> None:
    if frame.empty:
        fig, ax = plt.subplots(figsize=(8.2, 4.8))
        ax.text(0.5, 0.5, "Discovery-mode comparison not run", ha="center", va="center", fontsize=LABEL_SIZE)
        ax.set_axis_off()
        fig.tight_layout()
        fig.savefig(output_path, dpi=dpi, bbox_inches="tight")
        plt.close(fig)
        return
    frame = _ordered_metric_frame(frame)
    fig, axes = plt.subplots(2, 2, figsize=(9.6, 7.3), sharey=True)
    scenarios = [scenario for scenario in DEFAULT_SCENARIOS if scenario in set(frame["scenario"].astype(str))]
    markers = {
        "exhaustive_hop": "o",
        "bounded_relationship_guided": "D",
    }
    for ax, scenario in zip(axes.flatten(), scenarios):
        subset = frame[frame["scenario"].astype(str) == scenario]
        for _, row in subset.iterrows():
            policy = str(row["policy"])
            mode = str(row.get("discovery_mode", ""))
            ax.scatter(
                row["mean_total_operational_cost"],
                row["OCA"],
                s=78,
                marker=markers.get(mode, "o"),
                color=POLICY_COLORS.get(policy, "#111827"),
                edgecolor="white",
                linewidth=0.8,
                label=_policy_label(policy),
            )
        ax.set_title(_scenario_label(scenario), fontsize=SUBPLOT_TITLE_SIZE)
        ax.set_xlabel("Total operational cost", fontsize=LABEL_SIZE)
        ax.set_ylabel("OCA", fontsize=LABEL_SIZE)
        _style_axes(ax)
    handles, labels = axes.flatten()[0].get_legend_handles_labels()
    unique = dict(zip(labels, handles))
    fig.legend(unique.values(), unique.keys(), loc="upper center", ncol=4, frameon=False, fontsize=LEGEND_SIZE)
    fig.subplots_adjust(left=0.10, right=0.99, bottom=0.08, top=0.84, wspace=0.24, hspace=0.30)
    fig.savefig(output_path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)


def plot_pareto(frame: pd.DataFrame, output_path: Path, dpi: int, x_col: str, x_label: str) -> None:
    frame = _ordered_metric_frame(frame)
    fig, axes = plt.subplots(2, 2, figsize=(9.4, 7.2), sharey=True)
    scenarios = [scenario for scenario in DEFAULT_SCENARIOS if scenario in set(frame["scenario"].astype(str))]
    for ax, scenario in zip(axes.flatten(), scenarios):
        subset = frame[frame["scenario"].astype(str) == scenario]
        for _, row in subset.iterrows():
            ax.scatter(
                row[x_col],
                row["OCA"],
                s=80,
                color=POLICY_COLORS.get(str(row["policy"]), "#111827"),
                marker="D" if row.get("pareto_efficient_oca_cost", False) else "o",
                edgecolor="white",
                linewidth=0.8,
                label=_policy_label(str(row["policy"])),
            )
        ax.set_title(_scenario_label(scenario), fontsize=SUBPLOT_TITLE_SIZE)
        ax.set_xlabel(x_label, fontsize=LABEL_SIZE)
        ax.set_ylabel("OCA", fontsize=LABEL_SIZE)
        _style_axes(ax)
    handles, labels = axes.flatten()[0].get_legend_handles_labels()
    unique = dict(zip(labels, handles))
    fig.legend(unique.values(), unique.keys(), loc="upper center", ncol=4, frameon=False, fontsize=LEGEND_SIZE)
    fig.subplots_adjust(left=0.10, right=0.99, bottom=0.08, top=0.86, wspace=0.24, hspace=0.30)
    fig.savefig(output_path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)


def plot_oca_per_cost(frame: pd.DataFrame, output_path: Path, dpi: int) -> None:
    _plot_grouped_bars(
        frame,
        "OCA_per_total_cost",
        "OCA per total operational cost",
        output_path,
        dpi,
        ylabel="OCA / Cost",
    )


def plot_scalability(frame: pd.DataFrame, output_path: Path, dpi: int, y_col: str, y_label: str) -> None:
    if frame.empty or "graph_size" not in frame.columns:
        fig, ax = plt.subplots(figsize=(8.2, 4.8))
        ax.text(0.5, 0.5, "Scalability analysis not run", ha="center", va="center", fontsize=LABEL_SIZE)
        ax.set_axis_off()
        fig.tight_layout()
        fig.savefig(output_path, dpi=dpi, bbox_inches="tight")
        plt.close(fig)
        return
    summary = frame.groupby(["graph_size", "policy"], as_index=False, observed=False)[y_col].mean()
    policies = [policy for policy in DEFAULT_POLICIES if policy in set(summary["policy"].astype(str))]
    fig, ax = plt.subplots(figsize=(8.2, 4.8))
    graph_sizes = sorted(int(value) for value in summary["graph_size"].unique())
    x_lookup = {size: index for index, size in enumerate(graph_sizes)}
    for policy in policies:
        subset = summary[summary["policy"].astype(str) == policy].sort_values("graph_size")
        x_values = [x_lookup[int(value)] for value in subset["graph_size"]]
        ax.plot(
            x_values,
            subset[y_col],
            marker="o",
            linewidth=2.0,
            color=POLICY_COLORS.get(policy, "#111827"),
            label=_policy_label(policy),
        )
    ax.set_xlabel("Source nodes", fontsize=LABEL_SIZE)
    ax.set_ylabel(y_label, fontsize=LABEL_SIZE)
    if y_col == "mean_total_operational_cost":
        ax.set_yscale("symlog", linthresh=1.0)
    ax.set_xticks(list(x_lookup.values()), [str(value) for value in graph_sizes])
    ax.margins(x=0.08)
    ax.legend(frameon=False, ncol=4, fontsize=LEGEND_SIZE)
    _style_axes(ax)
    fig.tight_layout()
    fig.savefig(output_path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)


def _plot_grouped_bars(frame: pd.DataFrame, value_col: str, title: str, output_path: Path, dpi: int, ylabel: str) -> None:
    frame = _ordered_metric_frame(frame)
    scenarios = [scenario for scenario in DEFAULT_SCENARIOS if scenario in set(frame["scenario"].astype(str))]
    policies = [policy for policy in DEFAULT_POLICIES if policy in set(frame["policy"].astype(str))]
    width = 0.18
    x_positions = list(range(len(scenarios)))
    fig, ax = plt.subplots(figsize=(9.2, 4.8))
    for offset_idx, policy in enumerate(policies):
        values = []
        for scenario in scenarios:
            subset = frame[(frame["scenario"].astype(str) == scenario) & (frame["policy"].astype(str) == policy)]
            values.append(float(subset[value_col].iloc[0]) if not subset.empty else 0.0)
        shifted_positions = [x + ((offset_idx - 1.5) * width) for x in x_positions]
        ax.bar(shifted_positions, values, width=width, color=POLICY_COLORS.get(policy, "#111827"), label=_policy_label(policy))
    ax.set_xticks(x_positions, [_scenario_label(scenario).replace(" ", "\n") for scenario in scenarios])
    ax.set_title(title, fontsize=TITLE_SIZE)
    ax.set_ylabel(ylabel, fontsize=LABEL_SIZE)
    ax.legend(frameon=False, ncol=4, fontsize=LEGEND_SIZE)
    _style_axes(ax)
    fig.tight_layout()
    fig.savefig(output_path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)
