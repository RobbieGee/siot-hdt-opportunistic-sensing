from typing import Dict

import numpy as np
import pandas as pd

from sim_hdt_siot.config import (
    BUDGET_MATCHED_POLICIES,
    CONTEXT_VARIABLES,
    DEFAULT_POLICIES,
    DEFAULT_SCENARIOS,
    DISCOVERY_MODE_COMPARISON_POLICIES,
    POLICY_LABELS,
)


def compute_step_metrics(row: pd.Series) -> Dict[str, float]:
    variable_matches = {
        "place": float(row["true_place"] == row["estimated_place"]),
        "activity": float(row["true_activity"] == row["estimated_activity"]),
        "env_load": float(row["true_env_load"] == row["estimated_env_load"]),
        "resource_state": float(row["true_resource_state"] == row["estimated_resource_state"]),
    }
    variable_matches["mean_variable_accuracy"] = sum(variable_matches.values()) / len(variable_matches)
    variable_matches["overall_context_accuracy"] = float(all(value == 1.0 for value in variable_matches.values()))
    return variable_matches


def aggregate_metrics(step_results: pd.DataFrame, bootstrap_samples: int = 1000) -> pd.DataFrame:
    episode_summary = build_episode_summary(step_results)
    rows: list[Dict[str, float | str]] = []
    rng = np.random.default_rng(7)
    for (scenario, policy), group in episode_summary.groupby(["scenario", "policy"], observed=False):
        oca_low, oca_high = bootstrap_confidence_interval(group["overall_context_accuracy"].to_numpy(dtype=float), rng, bootstrap_samples)
        mva_low, mva_high = bootstrap_confidence_interval(group["mean_variable_accuracy"].to_numpy(dtype=float), rng, bootstrap_samples)
        conv_low, conv_high = bootstrap_confidence_interval(group["convergence_time"].dropna().to_numpy(dtype=float), rng, bootstrap_samples)
        row: Dict[str, float | str] = {
            "scenario": scenario,
            "policy": policy,
            "policy_label": POLICY_LABELS.get(str(policy), str(policy)),
            "discovery_mode": _dominant_text(group, "discovery_mode"),
            "place_accuracy": group["place"].mean(),
            "activity_accuracy": group["activity"].mean(),
            "env_load_accuracy": group["env_load"].mean(),
            "resource_state_accuracy": group["resource_state"].mean(),
            "mean_variable_accuracy": group["mean_variable_accuracy"].mean(),
            "mean_variable_accuracy_ci95_low": mva_low,
            "mean_variable_accuracy_ci95_high": mva_high,
            "overall_context_accuracy": group["overall_context_accuracy"].mean(),
            "overall_context_accuracy_ci95_low": oca_low,
            "overall_context_accuracy_ci95_high": oca_high,
            "avg_nodes_visited": group["nodes_visited"].mean(),
            "avg_discovered_candidate_count": group.get("discovered_candidate_count", pd.Series([0.0])).mean(),
            "avg_visited_node_count": group.get("visited_node_count", group["nodes_visited"]).mean(),
            "avg_traversed_edge_count": group.get("traversed_edge_count", pd.Series([0.0])).mean(),
            "avg_scored_candidate_count": group.get("scored_candidate_count", pd.Series([0.0])).mean(),
            "avg_candidate_variable_coverage_count": group.get("candidate_variable_coverage_count", pd.Series([0.0])).mean(),
            "candidate_all_variable_coverage_rate": group.get("all_variables_covered_by_candidates", pd.Series([0.0])).mean(),
            "avg_recruited_variable_coverage_count": group.get("recruited_variable_coverage_count", pd.Series([0.0])).mean(),
            "recruited_all_variable_coverage_rate": group.get("all_variables_covered_by_recruited_sources", pd.Series([0.0])).mean(),
            "avg_selected_source_count": group["selected_source_count"].mean(),
            "avg_estimated_latency": group["estimated_latency"].mean(),
            "avg_privacy_exposure_cost": group["privacy_exposure_cost"].mean(),
            "mean_discovery_cost": group.get("discovery_cost", pd.Series([0.0])).mean(),
            "mean_recruitment_cost": group.get("recruitment_cost", pd.Series([0.0])).mean(),
            "mean_selection_cost": group.get("selection_cost", pd.Series([0.0])).mean(),
            "mean_total_operational_cost": group.get("total_operational_cost", pd.Series([0.0])).mean(),
            "mean_latency_proxy": group["estimated_latency"].mean(),
            "mean_privacy_cost": group["privacy_exposure_cost"].mean(),
            "avg_ego_missing_rate": group["ego_missing_rate"].mean(),
            "convergence_time": group["convergence_time"].mean(),
            "convergence_time_ci95_low": conv_low,
            "convergence_time_ci95_high": conv_high,
            "adaptive_activation_rate": group.get("adaptive_activation_rate", pd.Series([0.0])).mean(),
            "robustness_vs_ego_missingness": compute_robustness_from_episode_summary(group),
        }
        row["OCA_per_source"] = row["overall_context_accuracy"] / max(float(row["avg_selected_source_count"]), 1.0)
        row["OCA_per_total_cost"] = row["overall_context_accuracy"] / max(float(row["mean_total_operational_cost"]), 1.0)
        for variable in CONTEXT_VARIABLES:
            values = group[f"macro_f1_{variable}"].to_numpy(dtype=float)
            low, high = bootstrap_confidence_interval(values, rng, bootstrap_samples)
            row[f"macro_f1_{variable}"] = group[f"macro_f1_{variable}"].mean()
            row[f"macro_f1_{variable}_ci95_low"] = low
            row[f"macro_f1_{variable}_ci95_high"] = high
        rows.append(row)
    return sort_scenario_policy_frame(pd.DataFrame(rows), ["scenario", "policy"]).reset_index(drop=True)


def build_episode_summary(step_results: pd.DataFrame) -> pd.DataFrame:
    rows: list[Dict[str, float | str | int]] = []
    group_columns = ["scenario", "policy", "episode"]
    if "seed" in step_results.columns:
        group_columns = ["scenario", "policy", "seed", "episode"]
    for keys, group in step_results.groupby(group_columns, observed=False):
        key_map = dict(zip(group_columns, keys if isinstance(keys, tuple) else (keys,)))
        row: Dict[str, float | str | int] = {
            "scenario": key_map["scenario"],
            "policy": key_map["policy"],
            "policy_label": POLICY_LABELS.get(str(key_map["policy"]), str(key_map["policy"])),
            "discovery_mode": _dominant_text(group, "discovery_mode"),
            "episode": int(key_map["episode"]),
            "place": group["place"].mean(),
            "activity": group["activity"].mean(),
            "env_load": group["env_load"].mean(),
            "resource_state": group["resource_state"].mean(),
            "mean_variable_accuracy": group["mean_variable_accuracy"].mean(),
            "overall_context_accuracy": group["overall_context_accuracy"].mean(),
            "nodes_visited": group["nodes_visited"].mean(),
            "visited_node_count": group["visited_node_count"].mean() if "visited_node_count" in group.columns else group["nodes_visited"].mean(),
            "traversed_edge_count": group["traversed_edge_count"].mean() if "traversed_edge_count" in group.columns else 0.0,
            "scored_candidate_count": group["scored_candidate_count"].mean() if "scored_candidate_count" in group.columns else 0.0,
            "candidate_variable_coverage_count": group["candidate_variable_coverage_count"].mean()
            if "candidate_variable_coverage_count" in group.columns
            else 0.0,
            "all_variables_covered_by_candidates": group["all_variables_covered_by_candidates"].mean()
            if "all_variables_covered_by_candidates" in group.columns
            else 0.0,
            "recruited_variable_coverage_count": group["recruited_variable_coverage_count"].mean()
            if "recruited_variable_coverage_count" in group.columns
            else 0.0,
            "all_variables_covered_by_recruited_sources": group["all_variables_covered_by_recruited_sources"].mean()
            if "all_variables_covered_by_recruited_sources" in group.columns
            else 0.0,
            "discovered_candidate_count": group["discovered_candidate_count"].mean()
            if "discovered_candidate_count" in group.columns
            else 0.0,
            "selected_source_count": group["selected_source_count"].mean(),
            "estimated_latency": group["estimated_latency"].mean(),
            "privacy_exposure_cost": group["privacy_exposure_cost"].mean(),
            "discovery_cost": group["discovery_cost"].mean() if "discovery_cost" in group.columns else 0.0,
            "recruitment_cost": group["recruitment_cost"].mean() if "recruitment_cost" in group.columns else 0.0,
            "selection_cost": group["selection_cost"].mean() if "selection_cost" in group.columns else 0.0,
            "total_operational_cost": group["total_operational_cost"].mean() if "total_operational_cost" in group.columns else 0.0,
            "ego_missing_rate": group["ego_missing_rate"].mean(),
            "convergence_time": compute_convergence_time(group),
            "adaptive_activation_rate": group["adaptive_triggered"].mean() if "adaptive_triggered" in group.columns else 0.0,
        }
        if "seed" in key_map:
            row["seed"] = int(key_map["seed"])
        for variable in CONTEXT_VARIABLES:
            row[f"macro_f1_{variable}"] = compute_macro_f1(group, variable)
        rows.append(row)
    sort_columns = ["scenario", "policy", "episode"]
    if "seed" in step_results.columns:
        sort_columns = ["scenario", "policy", "seed", "episode"]
    return sort_scenario_policy_frame(pd.DataFrame(rows), sort_columns).reset_index(drop=True)


def compute_macro_f1(group: pd.DataFrame, variable: str) -> float:
    labels = CONTEXT_VARIABLES[variable]
    scores = []
    true_col = f"true_{variable}"
    pred_col = f"estimated_{variable}"
    for label in labels:
        tp = float(((group[true_col] == label) & (group[pred_col] == label)).sum())
        fp = float(((group[true_col] != label) & (group[pred_col] == label)).sum())
        fn = float(((group[true_col] == label) & (group[pred_col] != label)).sum())
        precision = 0.0 if tp + fp == 0 else tp / (tp + fp)
        recall = 0.0 if tp + fn == 0 else tp / (tp + fn)
        if precision + recall == 0:
            scores.append(0.0)
        else:
            scores.append(2.0 * precision * recall / (precision + recall))
    return float(np.mean(scores))


def compute_convergence_time(group: pd.DataFrame) -> float:
    transition_lags: list[int] = []
    for _, episode_df in group.sort_values("step").groupby("episode"):
        episode_df = episode_df.reset_index(drop=True)
        transition_indices = []
        for idx in range(1, len(episode_df)):
            if any(
                episode_df.loc[idx, f"true_{variable}"] != episode_df.loc[idx - 1, f"true_{variable}"]
                for variable in CONTEXT_VARIABLES
            ):
                transition_indices.append(idx)
        for transition_idx in transition_indices:
            lag = 0
            while transition_idx + lag < len(episode_df):
                row = episode_df.loc[transition_idx + lag]
                if row["overall_context_accuracy"] == 1.0:
                    transition_lags.append(lag)
                    break
                lag += 1
    if not transition_lags:
        return float(len(group))
    return float(np.mean(transition_lags))


def compute_robustness_from_episode_summary(group: pd.DataFrame) -> float:
    if group["ego_missing_rate"].nunique() <= 1:
        return 0.0
    x = group["ego_missing_rate"].to_numpy()
    y = group["overall_context_accuracy"].to_numpy()
    slope = np.polyfit(x, y, deg=1)[0]
    return float(-slope)


def build_robustness_table(step_results: pd.DataFrame) -> pd.DataFrame:
    bins = pd.IntervalIndex.from_tuples(
        [(0.0, 0.25), (0.25, 0.50), (0.50, 0.75), (0.75, 1.01)],
        closed="left",
    )
    categories = pd.cut(step_results["ego_missing_rate"], bins)
    robustness = (
        step_results.assign(missingness_bin=categories.astype(str))
        .groupby(["scenario", "policy", "missingness_bin"], as_index=False, observed=False)
        .agg(
            overall_context_accuracy=("overall_context_accuracy", "mean"),
            observations=("overall_context_accuracy", "size"),
        )
    )
    robustness["policy_label"] = robustness["policy"].astype(str).map(POLICY_LABELS)
    return sort_scenario_policy_frame(robustness, ["scenario", "policy"]).reset_index(drop=True)


def build_overall_performance_table(metrics_df: pd.DataFrame) -> pd.DataFrame:
    columns = [
        "scenario",
        "policy",
        "policy_label",
        "overall_context_accuracy",
        "overall_context_accuracy_ci95_low",
        "overall_context_accuracy_ci95_high",
        "mean_variable_accuracy",
        "mean_variable_accuracy_ci95_low",
        "mean_variable_accuracy_ci95_high",
        "robustness_vs_ego_missingness",
    ]
    return sort_scenario_policy_frame(metrics_df[columns].copy(), ["scenario", "policy"]).reset_index(drop=True)


def build_per_variable_macro_f1_table(metrics_df: pd.DataFrame) -> pd.DataFrame:
    columns = ["scenario", "policy", "policy_label"]
    for variable in CONTEXT_VARIABLES:
        columns.extend([f"macro_f1_{variable}", f"macro_f1_{variable}_ci95_low", f"macro_f1_{variable}_ci95_high"])
    return sort_scenario_policy_frame(metrics_df[columns].copy(), ["scenario", "policy"]).reset_index(drop=True)


def build_convergence_table(metrics_df: pd.DataFrame) -> pd.DataFrame:
    columns = ["scenario", "policy", "policy_label", "convergence_time", "convergence_time_ci95_low", "convergence_time_ci95_high"]
    return sort_scenario_policy_frame(metrics_df[columns].copy(), ["scenario", "policy"]).reset_index(drop=True)


def build_overhead_table(metrics_df: pd.DataFrame) -> pd.DataFrame:
    columns = [
        "scenario",
        "policy",
        "policy_label",
        "discovery_mode",
        "avg_nodes_visited",
        "avg_discovered_candidate_count",
        "avg_visited_node_count",
        "avg_traversed_edge_count",
        "avg_scored_candidate_count",
        "avg_candidate_variable_coverage_count",
        "candidate_all_variable_coverage_rate",
        "avg_recruited_variable_coverage_count",
        "recruited_all_variable_coverage_rate",
        "avg_selected_source_count",
        "mean_discovery_cost",
        "mean_recruitment_cost",
        "mean_selection_cost",
        "mean_total_operational_cost",
        "avg_estimated_latency",
        "avg_privacy_exposure_cost",
        "adaptive_activation_rate",
    ]
    return sort_scenario_policy_frame(metrics_df[columns].copy(), ["scenario", "policy"]).reset_index(drop=True)


def build_statistical_summary(episode_summary: pd.DataFrame, bootstrap_samples: int = 1000) -> pd.DataFrame:
    metric_columns = [
        "overall_context_accuracy",
        "mean_variable_accuracy",
        "convergence_time",
        "nodes_visited",
        "visited_node_count",
        "traversed_edge_count",
        "scored_candidate_count",
        "discovered_candidate_count",
        "selected_source_count",
        "estimated_latency",
        "privacy_exposure_cost",
        "discovery_cost",
        "recruitment_cost",
        "selection_cost",
        "total_operational_cost",
        "ego_missing_rate",
    ] + [f"macro_f1_{variable}" for variable in CONTEXT_VARIABLES]
    rows: list[Dict[str, float | str]] = []
    rng = np.random.default_rng(13)
    for (scenario, policy), group in episode_summary.groupby(["scenario", "policy"], observed=False):
        for metric in metric_columns:
            series = group[metric].dropna()
            if series.empty:
                continue
            low, high = bootstrap_confidence_interval(series.to_numpy(dtype=float), rng, bootstrap_samples)
            rows.append(
                {
                    "scenario": scenario,
                    "policy": policy,
                    "policy_label": POLICY_LABELS.get(str(policy), str(policy)),
                    "metric": metric,
                    "mean": series.mean(),
                    "ci95_low": low,
                    "ci95_high": high,
                    "std": series.std(ddof=1) if len(series) > 1 else 0.0,
                    "median": series.median(),
                    "min": series.min(),
                    "max": series.max(),
                }
            )
    summary = pd.DataFrame(rows)
    return sort_scenario_policy_frame(summary, ["scenario", "policy"]).reset_index(drop=True)


def build_pairwise_policy_comparisons(episode_summary: pd.DataFrame, bootstrap_samples: int = 500) -> pd.DataFrame:
    rng = np.random.default_rng(7)
    rows: list[Dict[str, float | str]] = []
    metrics = ["overall_context_accuracy", "mean_variable_accuracy", "convergence_time"]
    for scenario, scenario_df in episode_summary.groupby("scenario", observed=False):
        policy_to_group = {policy: group.sort_values("episode") for policy, group in scenario_df.groupby("policy", observed=False)}
        available_policies = [policy for policy in DEFAULT_POLICIES if policy in policy_to_group]
        for idx, policy_a in enumerate(available_policies):
            for policy_b in available_policies[idx + 1 :]:
                index_columns = ["episode"]
                if "seed" in policy_to_group[policy_a].columns and "seed" in policy_to_group[policy_b].columns:
                    index_columns = ["seed", "episode"]
                episodes_a = policy_to_group[policy_a].set_index(index_columns)
                episodes_b = policy_to_group[policy_b].set_index(index_columns)
                shared_episodes = sorted(set(episodes_a.index).intersection(episodes_b.index))
                if not shared_episodes:
                    continue
                paired_a = episodes_a.loc[shared_episodes]
                paired_b = episodes_b.loc[shared_episodes]
                for metric in metrics:
                    diffs = (paired_a[metric] - paired_b[metric]).dropna().to_numpy(dtype=float)
                    if len(diffs) == 0:
                        continue
                    ci_low, ci_high = bootstrap_confidence_interval(diffs, rng, bootstrap_samples)
                    rows.append(
                        {
                            "scenario": scenario,
                            "metric": metric,
                            "policy_a": policy_a,
                            "policy_a_label": POLICY_LABELS.get(str(policy_a), str(policy_a)),
                            "policy_b": policy_b,
                            "policy_b_label": POLICY_LABELS.get(str(policy_b), str(policy_b)),
                            "mean_difference_a_minus_b": float(np.mean(diffs)),
                            "median_difference_a_minus_b": float(np.median(diffs)),
                            "ci95_low": ci_low,
                            "ci95_high": ci_high,
                            "episodes_compared": len(shared_episodes),
                        }
                    )
    comparisons = pd.DataFrame(rows)
    if comparisons.empty:
        return comparisons
    comparisons["scenario"] = pd.Categorical(comparisons["scenario"], categories=list(DEFAULT_SCENARIOS), ordered=True)
    comparisons["policy_a"] = pd.Categorical(comparisons["policy_a"], categories=list(DEFAULT_POLICIES), ordered=True)
    comparisons["policy_b"] = pd.Categorical(comparisons["policy_b"], categories=list(DEFAULT_POLICIES), ordered=True)
    return comparisons.sort_values(["scenario", "metric", "policy_a", "policy_b"]).reset_index(drop=True)


def bootstrap_confidence_interval(values: np.ndarray, rng: np.random.Generator, bootstrap_samples: int) -> tuple[float, float]:
    if len(values) == 0:
        return float("nan"), float("nan")
    if len(values) == 1:
        return float(values[0]), float(values[0])
    boot_means = []
    for _ in range(bootstrap_samples):
        sampled = rng.choice(values, size=len(values), replace=True)
        boot_means.append(float(np.mean(sampled)))
    low, high = np.quantile(np.asarray(boot_means), [0.025, 0.975])
    return float(low), float(high)


def build_per_variable_error_rates(step_results: pd.DataFrame, bootstrap_samples: int = 1000) -> pd.DataFrame:
    rows: list[Dict[str, object]] = []
    rng = np.random.default_rng(19)
    for (scenario, policy), group in step_results.groupby(["scenario", "policy"], observed=False):
        for variable in CONTEXT_VARIABLES:
            correct = group[variable].to_numpy(dtype=float)
            errors = 1.0 - correct
            low, high = bootstrap_confidence_interval(errors, rng, bootstrap_samples)
            rows.append(
                {
                    "scenario": scenario,
                    "policy": policy,
                    "policy_label": POLICY_LABELS.get(str(policy), str(policy)),
                    "variable": variable,
                    "error_rate": float(np.mean(errors)),
                    "ci95_low": low,
                    "ci95_high": high,
                    "observations": int(len(errors)),
                }
            )
    return sort_scenario_policy_frame(pd.DataFrame(rows), ["scenario", "policy", "variable"]).reset_index(drop=True)


def build_per_variable_macro_f1_long(episode_summary: pd.DataFrame, bootstrap_samples: int = 1000) -> pd.DataFrame:
    rows: list[Dict[str, object]] = []
    rng = np.random.default_rng(23)
    for (scenario, policy), group in episode_summary.groupby(["scenario", "policy"], observed=False):
        for variable in CONTEXT_VARIABLES:
            values = group[f"macro_f1_{variable}"].to_numpy(dtype=float)
            low, high = bootstrap_confidence_interval(values, rng, bootstrap_samples)
            rows.append(
                {
                    "scenario": scenario,
                    "policy": policy,
                    "policy_label": POLICY_LABELS.get(str(policy), str(policy)),
                    "variable": variable,
                    "macro_f1": float(np.mean(values)),
                    "ci95_low": low,
                    "ci95_high": high,
                    "aggregation_basis": "seed_episode_combinations",
                }
            )
    return sort_scenario_policy_frame(pd.DataFrame(rows), ["scenario", "policy", "variable"]).reset_index(drop=True)


def build_hardest_variable_by_scenario_policy(per_variable_macro_f1: pd.DataFrame) -> pd.DataFrame:
    rows: list[Dict[str, object]] = []
    for (scenario, policy), group in per_variable_macro_f1.groupby(["scenario", "policy"], observed=False):
        hardest = group.sort_values(["macro_f1", "variable"], ascending=[True, True]).iloc[0]
        rows.append(
            {
                "scenario": scenario,
                "policy": policy,
                "policy_label": POLICY_LABELS.get(str(policy), str(policy)),
                "hardest_variable": hardest["variable"],
                "macro_f1": hardest["macro_f1"],
                "ci95_low": hardest["ci95_low"],
                "ci95_high": hardest["ci95_high"],
            }
        )
    return sort_scenario_policy_frame(pd.DataFrame(rows), ["scenario", "policy"]).reset_index(drop=True)


def build_joint_error_pattern_summary(step_results: pd.DataFrame, top_n: int = 12) -> pd.DataFrame:
    records: list[Dict[str, object]] = []
    variable_names = list(CONTEXT_VARIABLES)
    working = step_results.copy()
    working["wrong_variables"] = working.apply(
        lambda row: tuple(variable for variable in variable_names if row[f"true_{variable}"] != row[f"estimated_{variable}"]),
        axis=1,
    )
    working["wrong_variable_count"] = working["wrong_variables"].map(len)
    working["failure_category"] = working["wrong_variable_count"].map(
        {
            0: "all_variables_correct",
            1: "only_one_variable_wrong",
            2: "two_variables_wrong",
            3: "three_variables_wrong",
            4: "all_variables_wrong",
        }
    )
    working["wrong_variable_combination"] = working["wrong_variables"].map(lambda values: "none" if not values else "+".join(values))
    for (scenario, policy), group in working.groupby(["scenario", "policy"], observed=False):
        total = max(1, len(group))
        combo_counts = (
            group.groupby(["failure_category", "wrong_variable_count", "wrong_variable_combination"], as_index=False, observed=False)
            .size()
            .rename(columns={"size": "count"})
            .sort_values(["count", "wrong_variable_combination"], ascending=[False, True])
            .head(top_n)
        )
        for _, row in combo_counts.iterrows():
            records.append(
                {
                    "summary_type": "wrong_variable_combination",
                    "scenario": scenario,
                    "policy": policy,
                    "policy_label": POLICY_LABELS.get(str(policy), str(policy)),
                    "failure_category": row["failure_category"],
                    "wrong_variable_count": int(row["wrong_variable_count"]),
                    "wrong_variable_combination": row["wrong_variable_combination"],
                    "true_context_tuple": "not_applicable",
                    "estimated_context_tuple": "not_applicable",
                    "count": int(row["count"]),
                    "rate": float(row["count"] / total),
                }
            )

        wrong_rows = group[group["wrong_variable_count"] > 0].copy()
        if wrong_rows.empty:
            continue
        wrong_rows["true_context_tuple"] = wrong_rows.apply(
            lambda row: "|".join(str(row[f"true_{variable}"]) for variable in variable_names), axis=1
        )
        wrong_rows["estimated_context_tuple"] = wrong_rows.apply(
            lambda row: "|".join(str(row[f"estimated_{variable}"]) for variable in variable_names), axis=1
        )
        tuple_counts = (
            wrong_rows.groupby(["true_context_tuple", "estimated_context_tuple"], as_index=False, observed=False)
            .size()
            .rename(columns={"size": "count"})
            .sort_values(["count", "true_context_tuple"], ascending=[False, True])
            .head(top_n)
        )
        for _, row in tuple_counts.iterrows():
            records.append(
                {
                    "summary_type": "misclassified_context_tuple",
                    "scenario": scenario,
                    "policy": policy,
                    "policy_label": POLICY_LABELS.get(str(policy), str(policy)),
                    "failure_category": "misclassified_context_tuple",
                    "wrong_variable_count": -1,
                    "wrong_variable_combination": "not_applicable",
                    "true_context_tuple": row["true_context_tuple"],
                    "estimated_context_tuple": row["estimated_context_tuple"],
                    "count": int(row["count"]),
                    "rate": float(row["count"] / total),
                }
            )
    return sort_scenario_policy_frame(pd.DataFrame(records), ["scenario", "policy", "summary_type"]).reset_index(drop=True)


def build_main_results_table(metrics_df: pd.DataFrame) -> pd.DataFrame:
    table = metrics_df[
        [
            "scenario",
            "policy",
            "policy_label",
            "overall_context_accuracy",
            "overall_context_accuracy_ci95_low",
            "overall_context_accuracy_ci95_high",
            "mean_variable_accuracy",
            "mean_variable_accuracy_ci95_low",
            "mean_variable_accuracy_ci95_high",
            "convergence_time",
            "convergence_time_ci95_low",
            "convergence_time_ci95_high",
        ]
    ].copy()
    table = table.rename(
        columns={
            "overall_context_accuracy": "OCA",
            "overall_context_accuracy_ci95_low": "OCA_ci95_low",
            "overall_context_accuracy_ci95_high": "OCA_ci95_high",
            "mean_variable_accuracy": "MVA",
            "mean_variable_accuracy_ci95_low": "MVA_ci95_low",
            "mean_variable_accuracy_ci95_high": "MVA_ci95_high",
        }
    )
    return sort_scenario_policy_frame(table, ["scenario", "policy"]).reset_index(drop=True)


def build_paper_overhead_table(metrics_df: pd.DataFrame) -> pd.DataFrame:
    table = metrics_df[
        [
        "scenario",
        "policy",
        "policy_label",
        "discovery_mode",
        "avg_nodes_visited",
        "avg_discovered_candidate_count",
        "avg_visited_node_count",
        "avg_traversed_edge_count",
        "avg_scored_candidate_count",
        "avg_candidate_variable_coverage_count",
        "candidate_all_variable_coverage_rate",
        "avg_recruited_variable_coverage_count",
        "recruited_all_variable_coverage_rate",
        "avg_selected_source_count",
            "mean_discovery_cost",
            "mean_recruitment_cost",
            "mean_selection_cost",
            "mean_total_operational_cost",
            "avg_estimated_latency",
            "avg_privacy_exposure_cost",
            "adaptive_activation_rate",
        ]
    ].copy()
    table = table.rename(
        columns={
            "avg_selected_source_count": "mean_sources",
            "mean_discovery_cost": "discovery_cost",
            "mean_recruitment_cost": "recruitment_cost",
            "mean_selection_cost": "selection_cost",
            "mean_total_operational_cost": "total_operational_cost",
            "avg_estimated_latency": "latency_proxy",
            "avg_privacy_exposure_cost": "privacy_cost",
        }
    )
    return sort_scenario_policy_frame(table, ["scenario", "policy"]).reset_index(drop=True)


def build_paper_per_variable_table(metrics_df: pd.DataFrame) -> pd.DataFrame:
    table = metrics_df[
        ["scenario", "policy", "policy_label", "macro_f1_place", "macro_f1_activity", "macro_f1_env_load", "macro_f1_resource_state"]
    ].copy()
    return sort_scenario_policy_frame(table, ["scenario", "policy"]).reset_index(drop=True)


def sort_scenario_policy_frame(frame: pd.DataFrame, sort_columns: list[str]) -> pd.DataFrame:
    ordered = frame.copy()
    if "scenario" in ordered.columns:
        present_scenarios = [str(value) for value in ordered["scenario"].dropna().astype(str).unique()]
        scenario_order = [scenario for scenario in DEFAULT_SCENARIOS if scenario in present_scenarios]
        scenario_order.extend(sorted(set(present_scenarios) - set(scenario_order)))
        ordered["scenario"] = pd.Categorical(ordered["scenario"].astype(str), categories=scenario_order, ordered=True)
    if "policy" in ordered.columns:
        present_policies = [str(value) for value in ordered["policy"].dropna().astype(str).unique()]
        policy_order = list(dict.fromkeys(list(DEFAULT_POLICIES) + list(BUDGET_MATCHED_POLICIES) + list(DISCOVERY_MODE_COMPARISON_POLICIES)))
        active_policy_order = [policy for policy in policy_order if policy in present_policies]
        active_policy_order.extend(sorted(set(present_policies) - set(active_policy_order)))
        ordered["policy"] = pd.Categorical(ordered["policy"].astype(str), categories=active_policy_order, ordered=True)
    return ordered.sort_values(sort_columns)


def _dominant_text(group: pd.DataFrame, column: str) -> str:
    if column not in group.columns or group[column].dropna().empty:
        return ""
    counts = group[column].astype(str).value_counts()
    return str(counts.index[0])
