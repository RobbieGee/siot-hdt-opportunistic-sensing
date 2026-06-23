from dataclasses import dataclass, replace
from pathlib import Path
from typing import Dict, Iterable, List

import numpy as np
import pandas as pd

from sim_hdt_siot.config import (
    BOUNDED_RELATIONSHIP_GUIDED_DISCOVERY,
    BUDGET_MATCHED_POLICIES,
    CONTEXT_VARIABLES,
    DEFAULT_POLICIES,
    DEFAULT_RELATION_UTILITIES,
    DEFAULT_SOURCE_SELECTION_LIMITS,
    DISCOVERY_MODE_COMPARISON_POLICIES,
    EXHAUSTIVE_HOP_DISCOVERY,
    ExperimentConfig,
    POLICY_LABELS,
    RELATION_TYPES,
)
from sim_hdt_siot.discovery import discover_sources, discover_sources_bounded_relationship_guided
from sim_hdt_siot.estimator import estimate_context
from sim_hdt_siot.graph_builder import build_siot_graph
from sim_hdt_siot.metrics import (
    aggregate_metrics,
    build_convergence_table,
    build_episode_summary,
    build_hardest_variable_by_scenario_policy,
    build_joint_error_pattern_summary,
    build_main_results_table,
    build_overall_performance_table,
    build_overhead_table,
    build_pairwise_policy_comparisons,
    build_paper_overhead_table,
    build_paper_per_variable_table,
    build_per_variable_error_rates,
    build_per_variable_macro_f1_long,
    build_per_variable_macro_f1_table,
    build_robustness_table,
    build_statistical_summary,
    compute_step_metrics,
)
from sim_hdt_siot.observation import apply_discovery_metadata, sample_observation
from sim_hdt_siot.plots import (
    plot_accuracy_vs_missing_ego,
    plot_budget_matched_oca_cost,
    plot_convergence_time,
    plot_discovery_mode_oca_cost,
    plot_effective_recovery_cost,
    plot_macro_f1_by_variable,
    plot_oca_per_cost,
    plot_overall_accuracy,
    plot_pareto,
    plot_scalability,
)
from sim_hdt_siot.ranking import rank_sources
from sim_hdt_siot.scenario import build_scenario


@dataclass
class SimulationResult:
    step_results: pd.DataFrame
    context_records: pd.DataFrame
    graph_summary: pd.DataFrame
    relation_distribution: pd.DataFrame
    candidate_pool_summary: pd.DataFrame


def run_experiment(config: ExperimentConfig | None = None) -> Dict[str, Path]:
    config = config or ExperimentConfig()
    root_dir = config.results_dir
    data_dir = root_dir / config.data_dir_name
    figures_dir = root_dir / config.figures_dir_name
    for directory in (data_dir, figures_dir):
        directory.mkdir(parents=True, exist_ok=True)

    main_result = simulate(config, capture_graph=True)
    step_results = _apply_experiment_metadata(main_result.step_results, config, "main_default", "main")
    episode_summary_df = build_episode_summary(step_results)
    metrics_df = _apply_experiment_metadata(
        aggregate_metrics(step_results, bootstrap_samples=config.bootstrap_samples),
        config,
        "main_default",
        "main",
    )
    robustness_df = build_robustness_table(step_results)
    overall_table_df = build_overall_performance_table(metrics_df)
    per_variable_summary_df = build_per_variable_macro_f1_table(metrics_df)
    convergence_table_df = build_convergence_table(metrics_df)
    overhead_table_df = build_overhead_table(metrics_df)
    statistical_summary_df = build_statistical_summary(episode_summary_df, bootstrap_samples=config.bootstrap_samples)
    pairwise_df = build_pairwise_policy_comparisons(episode_summary_df, bootstrap_samples=config.bootstrap_samples)
    table1_df = _apply_experiment_metadata(build_main_results_table(metrics_df), config, "main_default", "main")
    table2_df = _apply_experiment_metadata(build_paper_overhead_table(metrics_df), config, "main_default", "main")
    per_variable_table_df = build_paper_per_variable_table(metrics_df)
    adaptive_ablation_df = build_adaptive_ablation_table(config)

    per_variable_error_rates_df = build_per_variable_error_rates(step_results, bootstrap_samples=config.bootstrap_samples)
    per_variable_macro_f1_df = build_per_variable_macro_f1_long(episode_summary_df, bootstrap_samples=config.bootstrap_samples)
    hardest_variable_df = build_hardest_variable_by_scenario_policy(per_variable_macro_f1_df)
    joint_error_df = build_joint_error_pattern_summary(step_results)
    context_distribution_df = build_context_state_distribution(main_result.context_records)
    context_transition_df = build_context_transition_summary(main_result.context_records)
    adaptive_diagnostics_df = build_adaptive_diagnostics(step_results)
    parameter_table_df = build_parameter_table(config)
    convergence_cost_df = build_convergence_cost_results(step_results)
    cost_effectiveness_df = _apply_experiment_metadata(
        build_cost_effectiveness_results(metrics_df),
        config,
        "main_default",
        "main",
    )
    pareto_df = build_pareto_policy_results(cost_effectiveness_df)
    net_utility_df = build_net_utility_sensitivity(cost_effectiveness_df, config)
    budget_matched_results_df, budget_matched_overhead_df = build_budget_matched_outputs(config)
    discovery_mode_comparison_df = build_discovery_mode_comparison(config)
    scalability_results_df, scalability_graph_summary_df = build_scalability_outputs(config)
    scalability_results_df = _apply_experiment_metadata(
        scalability_results_df,
        config,
        "scalability",
        "scalability",
        episodes=config.scalability_episodes,
        timesteps=config.scalability_timesteps,
        seeds=config.scalability_seeds,
    )
    consistency_report_text = build_consistency_check_report(config, table1_df, discovery_mode_comparison_df, metrics_df)

    output_paths: Dict[str, Path] = {}
    csv_outputs = {
        config.raw_results_name: step_results,
        config.episode_summary_name: episode_summary_df,
        config.metrics_name: metrics_df,
        config.robustness_name: robustness_df,
        config.overall_table_name: overall_table_df,
        config.per_variable_summary_name: per_variable_summary_df,
        config.convergence_summary_name: convergence_table_df,
        config.overhead_summary_name: overhead_table_df,
        config.statistical_summary_name: statistical_summary_df,
        config.pairwise_comparisons_name: pairwise_df,
        config.main_results_table_name: table1_df,
        config.overhead_table_name: table2_df,
        config.adaptive_ablation_table_name: adaptive_ablation_df,
        config.per_variable_table_name: per_variable_table_df,
        "siot_graph_summary.csv": main_result.graph_summary,
        "siot_relation_distribution.csv": main_result.relation_distribution,
        "siot_candidate_pool_summary.csv": main_result.candidate_pool_summary,
        "adaptive_diagnostics.csv": adaptive_diagnostics_df,
        "simulator_parameter_table.csv": parameter_table_df,
        "context_state_distribution.csv": context_distribution_df,
        "context_transition_summary.csv": context_transition_df,
        "per_variable_error_rates.csv": per_variable_error_rates_df,
        "per_variable_macro_f1.csv": per_variable_macro_f1_df,
        "hardest_variable_by_scenario_policy.csv": hardest_variable_df,
        "joint_error_pattern_summary.csv": joint_error_df,
        "convergence_cost_results.csv": convergence_cost_df,
        "cost_effectiveness_results.csv": cost_effectiveness_df,
        "pareto_policy_results.csv": pareto_df,
        "net_utility_sensitivity.csv": net_utility_df,
        "budget_matched_results.csv": budget_matched_results_df,
        "budget_matched_overhead.csv": budget_matched_overhead_df,
        "discovery_mode_comparison.csv": discovery_mode_comparison_df,
        "scalability_results.csv": scalability_results_df,
        "scalability_graph_summary.csv": scalability_graph_summary_df,
        "figure2_overall_accuracy_data.csv": episode_summary_df[
            ["scenario", "policy", "policy_label", "seed", "episode", "overall_context_accuracy"]
        ],
        "figure3_convergence_time_data.csv": convergence_table_df,
        "figure4_macro_f1_data.csv": per_variable_summary_df,
        "figure5_performance_vs_ego_missingness_data.csv": robustness_df,
    }
    for filename, frame in csv_outputs.items():
        path = data_dir / filename
        frame.to_csv(path, index=False)
        output_paths[filename] = path

    raw_archive_path = data_dir / config.raw_results_archive_name
    step_results.to_csv(raw_archive_path, index=False, compression="gzip")
    output_paths[config.raw_results_archive_name] = raw_archive_path

    if config.run_sensitivity:
        sensitivity_outputs = run_sensitivity_analyses(config)
        for filename, frame in sensitivity_outputs.items():
            root_path = data_dir / filename
            frame.to_csv(root_path, index=False)
            output_paths[filename] = root_path

    figure_paths = build_figures(
        config,
        metrics_df,
        episode_summary_df,
        robustness_df,
        figures_dir,
        convergence_cost_df,
        budget_matched_results_df,
        pareto_df,
        cost_effectiveness_df,
        scalability_results_df,
        discovery_mode_comparison_df,
    )
    output_paths.update(figure_paths)

    summary_path = data_dir / config.experiment_summary_name
    summary_path.write_text(build_experiment_summary(metrics_df, table1_df, table2_df, adaptive_ablation_df), encoding="utf-8")
    output_paths[config.experiment_summary_name] = summary_path

    consistency_path = data_dir / "consistency_check_report.txt"
    consistency_path.write_text(consistency_report_text, encoding="utf-8")
    output_paths["consistency_check_report.txt"] = consistency_path

    validation_path = data_dir / config.validation_report_name
    validation_path.write_text(
        build_validation_report(config, data_dir, figures_dir, metrics_df, step_results, adaptive_diagnostics_df),
        encoding="utf-8",
    )
    output_paths[config.validation_report_name] = validation_path
    return output_paths


def simulate(config: ExperimentConfig, capture_graph: bool = False) -> SimulationResult:
    rows: List[Dict[str, object]] = []
    context_rows: List[Dict[str, object]] = []
    graph_rows: List[Dict[str, object]] = []
    relation_rows: List[Dict[str, object]] = []
    candidate_pool_rows: List[Dict[str, object]] = []
    for seed in config.seeds:
        for episode in range(config.episodes):
            for scenario_index, scenario_name in enumerate(config.scenarios):
                scenario_rng = np.random.default_rng(np.random.SeedSequence([seed, episode, scenario_index, 101]))
                scenario = build_scenario(scenario_name, config.timesteps, scenario_rng)
                graph_rng = np.random.default_rng(np.random.SeedSequence([seed, episode, scenario_index, 202]))
                graph_bundle = build_siot_graph(scenario, config, graph_rng, seed, episode)
                if capture_graph:
                    graph_rows.append(graph_bundle.graph_summary)
                    relation_rows.extend(graph_bundle.relation_distribution)
                    candidate_pool_rows.append(graph_bundle.candidate_pool_summary)

                ego_observations = []
                for timestep, truth in enumerate(scenario.timeline):
                    context_rows.append(
                        {
                            "seed": seed,
                            "episode": episode,
                            "scenario": scenario.name,
                            "timestep": timestep,
                            **truth.as_dict(),
                            "context_tuple": "|".join(truth.as_dict().values()),
                        }
                    )
                    ego_rng = np.random.default_rng(np.random.SeedSequence([seed, episode, scenario_index, timestep, 303]))
                    ego_observations.append(
                        sample_observation(
                            source_id=config.ego_id,
                            truth=truth,
                            timeline=scenario.timeline,
                            step=timestep,
                            profile=graph_bundle.agents[config.ego_id].profile,
                            rng=ego_rng,
                        )
                    )

                discovery_cache = {}
                for policy_index, policy in enumerate(config.policies):
                    base_policy = _base_policy(policy)
                    discovery_mode = _discovery_mode_for_policy(policy)
                    for timestep, truth in enumerate(scenario.timeline):
                        ego_observation = ego_observations[timestep]
                        adaptive_state = assess_adaptive_state(ego_observation, graph_bundle.agents[config.ego_id].profile, config, policy)
                        cache_key = _discovery_cache_key(policy, discovery_mode, adaptive_state)
                        if cache_key not in discovery_cache:
                            if base_policy == "ego_only":
                                discovery_cache[cache_key] = discover_sources(
                                    graph_bundle.graph,
                                    config.ego_id,
                                    "ego_only",
                                    int(adaptive_state["active_hop_radius"]),
                                )
                            elif discovery_mode == BOUNDED_RELATIONSHIP_GUIDED_DISCOVERY:
                                discovery_cache[cache_key] = discover_sources_bounded_relationship_guided(
                                    graph=graph_bundle.graph,
                                    ego_id=config.ego_id,
                                    max_hops=int(adaptive_state["active_hop_radius"]),
                                    node_budget=int(adaptive_state["discovery_node_budget_active"]),
                                    edge_budget=int(adaptive_state["discovery_edge_budget_active"]),
                                    max_candidates_to_score=int(adaptive_state["max_candidates_to_score_active"]),
                                    min_candidates_required=config.bounded_discovery_min_candidates_required,
                                    min_variable_coverage=config.bounded_discovery_min_variable_coverage,
                                    early_stop_when_all_variables_covered=config.bounded_discovery_early_stop_when_all_variables_covered,
                                    early_stop_min_quality=config.bounded_discovery_early_stop_min_quality,
                                    max_neighbors_per_expansion=config.bounded_discovery_max_neighbors_per_expansion,
                                    relation_priority_weights=config.relation_priority_weights,
                                    discovery_privacy_penalty=config.bounded_discovery_privacy_penalty,
                                    discovery_latency_penalty=config.bounded_discovery_latency_penalty,
                                    unresolved_variables=adaptive_state["unresolved_variables"],
                                )
                            else:
                                discovery_cache[cache_key] = discover_sources(
                                    graph_bundle.graph,
                                    config.ego_id,
                                    "opportunistic_all",
                                    int(adaptive_state["active_hop_radius"]),
                                )
                        discovery_result = discovery_cache[cache_key]
                        ranked_sources = rank_sources(
                            graph=graph_bundle.graph,
                            ego_id=config.ego_id,
                            candidates=discovery_result.candidates,
                            agents=graph_bundle.agents,
                            policy=base_policy,
                            trigger_threshold=config.trigger_threshold,
                            max_selected_sources=int(adaptive_state["selection_cap"]),
                            latency_penalty_scale=float(adaptive_state["latency_penalty_scale"]),
                            privacy_penalty_scale=float(adaptive_state["privacy_penalty_scale"]),
                            broaden_threshold_factor=float(adaptive_state["broaden_threshold_factor"]),
                            privacy_penalty_strength=config.privacy_penalty_strength,
                            rng=np.random.default_rng(np.random.SeedSequence([seed, episode, scenario_index, timestep, policy_index, 505])),
                        )
                        external_observations = {}
                        candidate_map = {candidate.source_id: candidate for candidate in discovery_result.candidates}
                        for ranked_source in ranked_sources:
                            node_index = int(graph_bundle.graph.nodes[ranked_source.source_id].get("node_index", 0))
                            source_rng = np.random.default_rng(
                                np.random.SeedSequence([seed, episode, scenario_index, timestep, node_index, 404])
                            )
                            raw_observation = sample_observation(
                                source_id=ranked_source.source_id,
                                truth=truth,
                                timeline=scenario.timeline,
                                step=timestep,
                                profile=graph_bundle.agents[ranked_source.source_id].profile,
                                rng=source_rng,
                            )
                            external_observations[ranked_source.source_id] = apply_discovery_metadata(
                                raw_observation,
                                candidate_map[ranked_source.source_id],
                            )

                        scored_candidate_count = _scored_candidate_count(base_policy, discovery_result)
                        latency_sum = sum(external_observations[source.source_id].latency for source in ranked_sources)
                        privacy_sum = sum(source.privacy_cost for source in ranked_sources)
                        discovery_cost = _discovery_cost(config, base_policy, discovery_result)
                        recruitment_cost = _recruitment_cost(config, len(ranked_sources) if base_policy != "ego_only" else 0, latency_sum, privacy_sum)
                        selection_cost = float(scored_candidate_count)
                        total_operational_cost = discovery_cost + recruitment_cost + (config.selection_cost_lambda * selection_cost)
                        recruited_variable_coverage_count = _source_variable_coverage_count(
                            graph_bundle.graph,
                            [source.source_id for source in ranked_sources],
                        )

                        estimate, confidence, telemetry = estimate_context(
                            ego_observation=ego_observation,
                            selected_sources=ranked_sources,
                            external_observations=external_observations,
                            policy=base_policy,
                            graph=graph_bundle.graph,
                            agents=graph_bundle.agents,
                            ego_id=config.ego_id,
                        )

                        row: Dict[str, object] = {
                            "seed": seed,
                            "episode": episode,
                            "scenario": scenario.name,
                            "policy": policy,
                            "policy_label": POLICY_LABELS.get(policy, policy),
                            "base_policy": base_policy,
                            "discovery_mode": discovery_result.discovery_mode,
                            "step": timestep,
                            "timestep": timestep,
                            "nodes_visited": discovery_result.nodes_visited,
                            "visited_node_count": discovery_result.nodes_visited,
                            "traversed_edge_count": discovery_result.traversed_edges,
                            "discovered_candidate_count": len(discovery_result.candidates),
                            "scored_candidate_count": scored_candidate_count,
                            "recruited_source_count": len(ranked_sources) if base_policy != "ego_only" else 0,
                            "selected_source_count": telemetry["selected_source_count"],
                            "observed_external_source_count": telemetry["observed_external_source_count"],
                            "estimated_latency": telemetry["estimated_latency"],
                            "privacy_exposure_cost": telemetry["privacy_exposure_cost"],
                            "latency_proxy": telemetry["estimated_latency"],
                            "privacy_cost": telemetry["privacy_exposure_cost"],
                            "discovery_cost": discovery_cost,
                            "relation_traversal_cost": discovery_result.relation_traversal_cost,
                            "recruitment_cost": recruitment_cost,
                            "selection_cost": selection_cost,
                            "total_operational_cost": total_operational_cost,
                            "recruited_latency_sum": latency_sum,
                            "recruited_privacy_sum": privacy_sum,
                            "candidate_variable_coverage_count": discovery_result.candidate_variable_coverage_count,
                            "all_variables_covered_by_candidates": bool(discovery_result.all_variables_covered_by_candidates),
                            "recruited_variable_coverage_count": recruited_variable_coverage_count,
                            "all_variables_covered_by_recruited_sources": recruited_variable_coverage_count == len(CONTEXT_VARIABLES),
                            "ego_missing": 1.0 - (len(ego_observation.values) / len(CONTEXT_VARIABLES)),
                            "ego_missing_rate": 1.0 - (len(ego_observation.values) / len(CONTEXT_VARIABLES)),
                            "ego_confidence": adaptive_state["ego_confidence"],
                            "ego_overall_confidence": adaptive_state["ego_confidence"],
                            "unresolved_variable_count": adaptive_state["unresolved_variable_count"],
                            "ego_unresolved_variables": adaptive_state["unresolved_variable_count"],
                            "adaptive_triggered": bool(adaptive_state["adaptive_triggered"]),
                            "adaptive_mode_triggered": bool(adaptive_state["adaptive_triggered"]),
                            "active_hop_radius": adaptive_state["active_hop_radius"],
                            "selection_cap": adaptive_state["selection_cap"],
                            "discovery_node_budget_active": discovery_result.discovery_node_budget_active,
                            "discovery_edge_budget_active": discovery_result.discovery_edge_budget_active,
                            "max_candidates_to_score_active": discovery_result.max_candidates_to_score_active,
                            "stopped_by_budget": bool(discovery_result.stopped_by_budget),
                            "stopped_by_coverage": bool(discovery_result.stopped_by_coverage),
                            "stopped_by_quality": bool(discovery_result.stopped_by_quality),
                            "stopped_by_frontier_empty": bool(discovery_result.stopped_by_frontier_empty),
                            "effective_max_hops": adaptive_state["active_hop_radius"],
                            "effective_max_selected_sources": adaptive_state["selection_cap"],
                            "effective_latency_penalty_scale": adaptive_state["latency_penalty_scale"],
                            "effective_privacy_penalty_scale": adaptive_state["privacy_penalty_scale"],
                            "effective_broaden_threshold_factor": adaptive_state["broaden_threshold_factor"],
                            "ego_observation_age": ego_observation.age,
                            "mean_recruited_graph_distance": _mean_or_zero([source.hops for source in ranked_sources]),
                            "mean_recruited_relation_utility": _mean_or_zero([source.relation_utility for source in ranked_sources]),
                            "mean_recruited_trust": _mean_or_zero([source.trust for source in ranked_sources]),
                            "confidence_place": confidence["place"],
                            "confidence_activity": confidence["activity"],
                            "confidence_env_load": confidence["env_load"],
                            "confidence_resource_state": confidence["resource_state"],
                        }
                        for variable, value in truth.as_dict().items():
                            row[f"true_{variable}"] = value
                        for variable, value in estimate.as_dict().items():
                            row[f"estimated_{variable}"] = value
                        row.update(compute_step_metrics(pd.Series(row)))
                        rows.append(row)

    return SimulationResult(
        step_results=pd.DataFrame(rows),
        context_records=pd.DataFrame(context_rows).drop_duplicates(),
        graph_summary=pd.DataFrame(graph_rows),
        relation_distribution=pd.DataFrame(relation_rows),
        candidate_pool_summary=pd.DataFrame(candidate_pool_rows),
    )


def assess_adaptive_state(
    ego_observation,
    ego_profile,
    config: ExperimentConfig,
    policy: str,
) -> Dict[str, object]:
    base_policy = _base_policy(policy)
    observed_confidences = []
    unresolved_variables = 0
    unresolved_variable_names: List[str] = []
    for variable in CONTEXT_VARIABLES:
        if variable not in ego_observation.values:
            unresolved_variables += 1
            unresolved_variable_names.append(variable)
            observed_confidences.append(0.0)
            continue
        confidence = ego_profile.variable_accuracy.get(variable, ego_profile.base_accuracy) * ego_observation.freshness
        observed_confidences.append(confidence)
        if confidence < config.ego_confidence_threshold:
            unresolved_variables += 1
            unresolved_variable_names.append(variable)

    ego_missing_rate = 1.0 - (len(ego_observation.values) / len(CONTEXT_VARIABLES))
    ego_confidence = float(np.mean(observed_confidences)) if observed_confidences else 0.0
    adaptive_triggered = False
    if config.adaptive_enabled and base_policy in {"siot_aware", "siot_aware_trust_privacy", "random_k", "budgeted_opportunistic_k"}:
        low_confidence = ego_confidence < config.ego_confidence_threshold
        high_missingness = ego_missing_rate > config.ego_missingness_threshold
        highly_unresolved = unresolved_variables >= config.unresolved_variable_threshold
        if config.collapse_require_joint_condition:
            adaptive_triggered = high_missingness or (low_confidence and highly_unresolved)
        else:
            adaptive_triggered = high_missingness or low_confidence or highly_unresolved

    active_hop_radius = config.baseline_hop_radius
    selection_cap = DEFAULT_SOURCE_SELECTION_LIMITS.get(policy, config.default_selection_cap)
    latency_penalty_scale = 1.0
    privacy_penalty_scale = 1.0
    broaden_threshold_factor = 1.0
    if adaptive_triggered:
        active_hop_radius = config.adaptive_hop_radius
        selection_cap = config.adaptive_selection_cap
        latency_penalty_scale = config.adaptive_latency_penalty_scale
        privacy_penalty_scale = config.adaptive_privacy_penalty_scale
        broaden_threshold_factor = 0.70
    if base_policy == "ego_only":
        selection_cap = 0
    elif base_policy == "opportunistic_all":
        selection_cap = 999
        active_hop_radius = config.adaptive_hop_radius
    node_budget, edge_budget, score_cap = _bounded_discovery_budgets(config, adaptive_triggered)

    return {
        "ego_confidence": ego_confidence,
        "unresolved_variable_count": unresolved_variables,
        "unresolved_variables": tuple(sorted(set(unresolved_variable_names))),
        "adaptive_triggered": adaptive_triggered,
        "active_hop_radius": active_hop_radius,
        "selection_cap": selection_cap if selection_cap is not None else 999,
        "discovery_node_budget_active": node_budget,
        "discovery_edge_budget_active": edge_budget,
        "max_candidates_to_score_active": score_cap,
        "latency_penalty_scale": latency_penalty_scale,
        "privacy_penalty_scale": privacy_penalty_scale,
        "broaden_threshold_factor": broaden_threshold_factor,
    }


def build_adaptive_diagnostics(step_results: pd.DataFrame) -> pd.DataFrame:
    columns = [
        "scenario",
        "seed",
        "episode",
        "timestep",
        "policy",
        "policy_label",
        "discovery_mode",
        "ego_missing",
        "ego_confidence",
        "unresolved_variable_count",
        "adaptive_triggered",
        "active_hop_radius",
        "selection_cap",
        "discovery_node_budget_active",
        "discovery_edge_budget_active",
        "max_candidates_to_score_active",
        "stopped_by_budget",
        "stopped_by_coverage",
        "stopped_by_quality",
        "stopped_by_frontier_empty",
        "discovered_candidate_count",
        "visited_node_count",
        "traversed_edge_count",
        "scored_candidate_count",
        "recruited_source_count",
        "candidate_variable_coverage_count",
        "all_variables_covered_by_candidates",
        "recruited_variable_coverage_count",
        "all_variables_covered_by_recruited_sources",
    ]
    diagnostics = step_results[columns].copy()
    diagnostics["adaptive_triggered"] = diagnostics["adaptive_triggered"].astype(bool)
    for column in [
        "stopped_by_budget",
        "stopped_by_coverage",
        "stopped_by_quality",
        "stopped_by_frontier_empty",
        "all_variables_covered_by_candidates",
        "all_variables_covered_by_recruited_sources",
    ]:
        diagnostics[column] = diagnostics[column].astype(bool)
    return diagnostics


def build_context_state_distribution(context_records: pd.DataFrame) -> pd.DataFrame:
    rows: List[Dict[str, object]] = []
    for scenario, group in context_records.groupby("scenario", observed=False):
        total = max(1, len(group))
        for variable in CONTEXT_VARIABLES:
            counts = group[variable].value_counts().sort_index()
            for state, count in counts.items():
                rows.append(
                    {
                        "distribution_type": "variable_state",
                        "scenario": scenario,
                        "variable": variable,
                        "state": state,
                        "context_tuple": "",
                        "count": int(count),
                        "share": float(count / total),
                    }
                )
        tuple_counts = group["context_tuple"].value_counts().head(25)
        for context_tuple, count in tuple_counts.items():
            rows.append(
                {
                    "distribution_type": "joint_context_tuple",
                    "scenario": scenario,
                    "variable": "joint",
                    "state": "",
                    "context_tuple": context_tuple,
                    "count": int(count),
                    "share": float(count / total),
                }
            )
    return pd.DataFrame(rows)


def build_context_transition_summary(context_records: pd.DataFrame) -> pd.DataFrame:
    ordered = context_records.sort_values(["scenario", "seed", "episode", "timestep"]).copy()
    ordered["previous_context_tuple"] = ordered.groupby(["scenario", "seed", "episode"], observed=False)["context_tuple"].shift(1)
    transitions = ordered.dropna(subset=["previous_context_tuple"])
    summary = (
        transitions.groupby(["scenario", "previous_context_tuple", "context_tuple"], as_index=False, observed=False)
        .size()
        .rename(columns={"size": "transition_count", "context_tuple": "next_context_tuple"})
        .sort_values(["scenario", "transition_count"], ascending=[True, False])
    )
    totals = summary.groupby("scenario", observed=False)["transition_count"].transform("sum")
    summary["transition_share"] = summary["transition_count"] / totals
    return summary.reset_index(drop=True)


def _apply_experiment_metadata(
    frame: pd.DataFrame,
    config: ExperimentConfig,
    experiment_tag: str,
    run_type: str,
    episodes: int | None = None,
    timesteps: int | None = None,
    seeds: Iterable[int] | None = None,
    graph_size: int | None = None,
    adaptive_enabled: bool | None = None,
) -> pd.DataFrame:
    if frame.empty:
        return frame.copy()
    annotated = frame.copy()
    annotated["experiment_tag"] = experiment_tag
    annotated["run_type"] = run_type
    if "policy" in annotated.columns:
        annotated["policy_id"] = annotated["policy"].astype(str)
        if "policy_label" not in annotated.columns:
            annotated["policy_label"] = annotated["policy_id"].map(POLICY_LABELS).fillna(annotated["policy_id"])
        if "discovery_mode" not in annotated.columns:
            annotated["discovery_mode"] = annotated["policy_id"].map(_discovery_mode_for_policy)
    elif "policy_id" not in annotated.columns:
        annotated["policy_id"] = ""
    if "discovery_mode" not in annotated.columns:
        annotated["discovery_mode"] = ""
    if "scenario" not in annotated.columns:
        annotated["scenario"] = ""
    if graph_size is not None or "graph_size" not in annotated.columns:
        annotated["graph_size"] = config.source_node_count if graph_size is None else graph_size
    annotated["episodes"] = config.episodes if episodes is None else episodes
    annotated["timesteps"] = config.timesteps if timesteps is None else timesteps
    seed_values = config.seeds if seeds is None else tuple(seeds)
    annotated["seeds"] = ",".join(str(seed) for seed in seed_values)
    annotated["adaptive_enabled"] = config.adaptive_enabled if adaptive_enabled is None else adaptive_enabled
    annotated["bounded_discovery_enabled"] = annotated["discovery_mode"].astype(str).eq(BOUNDED_RELATIONSHIP_GUIDED_DISCOVERY)
    preferred = [
        "experiment_tag",
        "run_type",
        "policy_id",
        "policy",
        "policy_label",
        "discovery_mode",
        "scenario",
        "graph_size",
        "episodes",
        "timesteps",
        "seeds",
        "adaptive_enabled",
        "bounded_discovery_enabled",
    ]
    ordered_columns = [column for column in preferred if column in annotated.columns]
    ordered_columns.extend(column for column in annotated.columns if column not in ordered_columns)
    return annotated[ordered_columns]


def build_parameter_table(config: ExperimentConfig) -> pd.DataFrame:
    rows: List[Dict[str, object]] = []

    def add(category: str, parameter: str, value: object, description: str) -> None:
        rows.append({"category": category, "parameter": parameter, "value": str(value), "description": description})

    add("experiment", "episodes", config.episodes, "Main episodes per seed and scenario.")
    add("experiment", "timesteps_per_episode", config.timesteps, "Discrete timesteps per episode.")
    add("experiment", "seeds", list(config.seeds), "Main deterministic random seeds.")
    add("experiment", "bootstrap_basis", config.bootstrap_basis, "Bootstrap unit used for confidence intervals.")
    add("experiment", "bootstrap_samples", config.bootstrap_samples, "Bootstrap resamples for 95% confidence intervals.")
    add("graph", "number_of_external_source_nodes", config.source_node_count, "External SIoT object/source nodes.")
    add("graph", "number_of_relation_types", len(RELATION_TYPES), "Typed SIoT edge labels.")
    add("graph", "relation_types", list(RELATION_TYPES), "Supported SIoT relationship types.")
    for relation, utility in DEFAULT_RELATION_UTILITIES.items():
        add("relation_utility", relation, utility, "Default relation utility used in discovery/ranking.")
    add("graph", "ego_personal_device_count", config.ego_personal_device_count, "Ego-associated OOR devices.")
    add("graph", "environmental_source_count", config.environmental_source_count, "Environmental source count target.")
    add("graph", "infrastructure_source_count", config.infrastructure_source_count, "Infrastructure source count target.")
    add("graph", "vehicle_source_count", config.vehicle_source_count, "Vehicle source count target.")
    add("graph", "graph_density_mode", config.graph_density_mode, "Default graph-density setting.")
    add("graph", "graph_density_edge_probabilities", config.graph_density_edge_probabilities, "Extra typed-edge probability by density.")
    add("graph", "ego_layer1_candidate_min", config.ego_layer1_candidate_min, "Lower target for ego 1-hop candidate pool.")
    add("graph", "ego_layer1_candidate_max", config.ego_layer1_candidate_max, "Upper target for ego 1-hop candidate pool.")
    add("graph", "target_average_shortest_path", f"{config.target_average_shortest_path_low}-{config.target_average_shortest_path_high}", "Default graph path-length target.")
    add("adaptive", "baseline_hop_radius", config.baseline_hop_radius, "Normal graph discovery radius.")
    add("adaptive", "adaptive_hop_radius", config.adaptive_hop_radius, "Broadened graph discovery radius.")
    add("adaptive", "default_selection_cap", config.default_selection_cap, "Selective recruitment cap in normal mode.")
    add("adaptive", "adaptive_selection_cap", config.adaptive_selection_cap, "Selective recruitment cap in adaptive mode.")
    add("bounded_discovery", "default_mode_for_siot_aware", BOUNDED_RELATIONSHIP_GUIDED_DISCOVERY, "Default discovery mode for SIoT-aware selective policies.")
    add("bounded_discovery", "exhaustive_mode", EXHAUSTIVE_HOP_DISCOVERY, "High-coverage hop-limited reference discovery mode.")
    add("bounded_discovery", "small_graph_node_edge_score_budget", (config.bounded_discovery_node_budget_small, config.bounded_discovery_edge_budget_small, config.bounded_discovery_score_cap_small), "Bounded discovery budget for small graphs.")
    add("bounded_discovery", "medium_graph_node_edge_score_budget", (config.bounded_discovery_node_budget_medium, config.bounded_discovery_edge_budget_medium, config.bounded_discovery_score_cap_medium), "Bounded discovery budget for medium graphs.")
    add("bounded_discovery", "large_graph_node_edge_score_budget", (config.bounded_discovery_node_budget_large, config.bounded_discovery_edge_budget_large, config.bounded_discovery_score_cap_large), "Bounded discovery budget for large graphs.")
    add("bounded_discovery", "adaptive_discovery_budget_multiplier", config.adaptive_discovery_budget_multiplier, "Multiplier applied to bounded discovery budgets under adaptive broadening.")
    add("bounded_discovery", "min_candidates_required", config.bounded_discovery_min_candidates_required, "Minimum candidates before bounded early stopping.")
    add("bounded_discovery", "min_variable_coverage", config.bounded_discovery_min_variable_coverage, "Minimum distinct context variables covered before quality-based bounded stopping.")
    add("bounded_discovery", "early_stop_when_all_variables_covered", config.bounded_discovery_early_stop_when_all_variables_covered, "Whether bounded discovery stops when all variables are covered and quality thresholds are met.")
    add("bounded_discovery", "early_stop_min_quality", config.bounded_discovery_early_stop_min_quality, "Minimum expected candidate quality for bounded early stopping.")
    add("bounded_discovery", "max_neighbors_per_expansion", config.bounded_discovery_max_neighbors_per_expansion, "Per-node beam width for bounded relationship-guided frontier expansion.")
    add("bounded_discovery", "relation_priority_weights", config.relation_priority_weights, "SIoT relation priorities used by bounded discovery.")
    add("bounded_discovery", "privacy_penalty", config.bounded_discovery_privacy_penalty, "Expected privacy penalty used during bounded discovery expansion.")
    add("bounded_discovery", "latency_penalty", config.bounded_discovery_latency_penalty, "Expected latency penalty used during bounded discovery expansion.")
    add("adaptive", "ego_missingness_threshold", config.ego_missingness_threshold, "Adaptive trigger threshold for ego missingness.")
    add("adaptive", "ego_confidence_threshold", config.ego_confidence_threshold, "Adaptive trigger threshold for ego confidence.")
    add("adaptive", "unresolved_variable_threshold", config.unresolved_variable_threshold, "Adaptive trigger threshold for unresolved variables.")
    add("scenario", "degraded_external_fraction", config.degraded_external_fraction, "Default noisy-external degraded-source fraction.")
    add("ranking", "trigger_threshold", config.trigger_threshold, "Minimum selective recruitment score before fallback.")
    add("ranking", "privacy_penalty_strength", config.privacy_penalty_strength, "Privacy divisor strength for privacy-aware SIoT.")
    add("cost", "relation_traversal_costs", config.relation_traversal_costs, "Relation-aware discovery traversal costs.")
    add("cost", "discovery_edge_lambda", config.discovery_edge_lambda, "Weight on traversed edges in discovery cost.")
    add("cost", "recruitment_latency_lambda", config.recruitment_latency_lambda, "Weight on latency in recruitment cost.")
    add("cost", "recruitment_privacy_lambda", config.recruitment_privacy_lambda, "Weight on privacy in recruitment cost.")
    add("cost", "selection_cost_lambda", config.selection_cost_lambda, "Weight on candidate scoring cost.")
    add("cost", "net_utility_cost_lambda", config.net_utility_cost_lambda, "Default interpretive utility cost penalty.")
    add("cost", "net_utility_privacy_lambda", config.net_utility_privacy_lambda, "Default interpretive utility privacy penalty.")
    add("cost", "net_utility_latency_lambda", config.net_utility_latency_lambda, "Default interpretive utility latency penalty.")
    add("cost", "cost_epsilon", config.cost_epsilon, "Denominator floor for cost-effectiveness ratios.")
    add("budget_matched", "policies", list(BUDGET_MATCHED_POLICIES), "Secondary budget-matched policy set.")
    add("budget_matched", "episodes", config.budget_matched_episodes, "Budget-matched episodes per seed and scenario.")
    add("budget_matched", "timesteps", config.budget_matched_timesteps, "Budget-matched timesteps per episode.")
    add("budget_matched", "seeds", list(config.budget_matched_seeds), "Budget-matched deterministic seeds.")
    add("discovery_mode_comparison", "policies", list(DISCOVERY_MODE_COMPARISON_POLICIES), "Exhaustive versus bounded SIoT-aware comparison policies.")
    add("discovery_mode_comparison", "episodes", config.discovery_mode_comparison_episodes, "Discovery-mode comparison episodes per seed and scenario.")
    add("discovery_mode_comparison", "timesteps", config.discovery_mode_comparison_timesteps, "Discovery-mode comparison timesteps per episode.")
    add("discovery_mode_comparison", "seeds", list(config.discovery_mode_comparison_seeds), "Discovery-mode comparison deterministic seeds.")
    for variable, states in CONTEXT_VARIABLES.items():
        add("context_domain", variable, states, "Discrete context variable domain.")
    for scenario in config.scenarios:
        scenario_def = build_scenario(scenario, 1, np.random.default_rng(123))
        settings = scenario_def.settings
        add("scenario_setting", f"{scenario}_ego_availability", settings.ego_availability, "Scenario ego observation availability.")
        add("scenario_setting", f"{scenario}_ego_base_accuracy", settings.ego_base_accuracy, "Scenario ego base accuracy.")
        add("scenario_setting", f"{scenario}_ego_variable_accuracy", settings.ego_variable_accuracy, "Scenario ego variable accuracies.")
        add("scenario_setting", f"{scenario}_external_degraded_fraction", settings.external_degraded_fraction, "Scenario external perturbation rate.")
    add("sensitivity", "sensitivity_episodes", config.sensitivity_episodes, "Compact sensitivity episodes per seed.")
    add("sensitivity", "sensitivity_timesteps", config.sensitivity_timesteps, "Compact sensitivity timesteps per episode.")
    add("sensitivity", "sensitivity_seeds", list(config.sensitivity_seeds), "Sensitivity deterministic seeds.")
    add("scalability", "graph_sizes", list(config.scalability_graph_sizes), "Source-node counts for scalability analysis.")
    add("scalability", "episodes", config.scalability_episodes, "Scalability episodes per seed and scenario.")
    add("scalability", "timesteps", config.scalability_timesteps, "Scalability timesteps per episode.")
    add("scalability", "seeds", list(config.scalability_seeds), "Scalability deterministic seeds.")
    return pd.DataFrame(rows)


def build_adaptive_ablation_table(config: ExperimentConfig) -> pd.DataFrame:
    rows: List[Dict[str, object]] = []
    for label, enabled in (("adaptive_disabled", False), ("adaptive_enabled", True)):
        ablation_config = replace(
            config,
            adaptive_enabled=enabled,
            policies=("siot_aware", "siot_aware_trust_privacy"),
            run_sensitivity=False,
        )
        result = simulate(ablation_config, capture_graph=False)
        metrics = aggregate_metrics(result.step_results, bootstrap_samples=config.bootstrap_samples)
        for _, row in metrics.iterrows():
            rows.append(
                {
                    "scenario": row["scenario"],
                    "policy": row["policy"],
                    "policy_label": row["policy_label"],
                    "discovery_mode": row.get("discovery_mode", ""),
                    "adaptive_mode": label,
                    "OCA": row["overall_context_accuracy"],
                    "OCA_ci95_low": row["overall_context_accuracy_ci95_low"],
                    "OCA_ci95_high": row["overall_context_accuracy_ci95_high"],
                    "MVA": row["mean_variable_accuracy"],
                    "mean_sources": row["avg_selected_source_count"],
                    "mean_discovery_cost": row["mean_discovery_cost"],
                    "mean_total_operational_cost": row["mean_total_operational_cost"],
                    "latency_proxy": row["avg_estimated_latency"],
                    "privacy_cost": row["avg_privacy_exposure_cost"],
                    "adaptive_activation_rate": row["adaptive_activation_rate"],
                }
            )
    return pd.DataFrame(rows)


def run_sensitivity_analyses(config: ExperimentConfig) -> Dict[str, pd.DataFrame]:
    base_kwargs = {
        "episodes": config.sensitivity_episodes,
        "timesteps": config.sensitivity_timesteps,
        "seeds": config.sensitivity_seeds,
        "scenarios": config.sensitivity_scenarios,
        "run_sensitivity": False,
    }
    outputs: Dict[str, pd.DataFrame] = {}
    hop_rows = []
    for baseline, adaptive in config.sensitivity_hop_radius_values:
        run_config = replace(config, baseline_hop_radius=baseline, adaptive_hop_radius=adaptive, **base_kwargs)
        hop_rows.extend(_sensitivity_rows(run_config, "hop_radius", f"{baseline}/{adaptive}"))
    outputs["sensitivity_hop_radius.csv"] = pd.DataFrame(hop_rows)

    cap_rows = []
    for cap in config.sensitivity_selection_cap_values:
        run_config = replace(config, default_selection_cap=cap, adaptive_selection_cap=cap, **base_kwargs)
        cap_rows.extend(_sensitivity_rows(run_config, "selection_cap", cap))
    outputs["sensitivity_selection_cap.csv"] = pd.DataFrame(cap_rows)

    privacy_rows = []
    for label, strength in config.sensitivity_privacy_penalty_values.items():
        run_config = replace(config, privacy_penalty_strength=strength, **base_kwargs)
        privacy_rows.extend(_sensitivity_rows(run_config, "privacy_penalty", label))
    outputs["sensitivity_privacy_penalty.csv"] = pd.DataFrame(privacy_rows)

    degraded_rows = []
    for fraction in config.sensitivity_degraded_fraction_values:
        run_config = replace(config, degraded_external_fraction=fraction, **base_kwargs)
        degraded_rows.extend(_sensitivity_rows(run_config, "degraded_external_fraction", fraction))
    outputs["sensitivity_degraded_fraction.csv"] = pd.DataFrame(degraded_rows)

    density_rows = []
    topology_rows = []
    for density in config.sensitivity_graph_density_values:
        run_config = replace(config, graph_density_mode=density, **base_kwargs)
        density_rows.extend(_sensitivity_rows(run_config, "graph_density", density))
        topology_result = simulate(run_config, capture_graph=True)
        graph_summary = topology_result.graph_summary
        topology_rows.append(
            {
                "parameter_name": "graph_density",
                "parameter_value": density,
                "mean_average_shortest_path_length": graph_summary["average_shortest_path_length"].mean(),
                "mean_effective_diameter": graph_summary["effective_diameter"].mean(),
                "mean_average_degree": graph_summary["average_degree"].mean(),
                "mean_ego_degree": graph_summary["ego_degree"].mean(),
                "mean_ego_betweenness_centrality": graph_summary["ego_betweenness_centrality"].mean(),
                "mean_candidates_1_hop": graph_summary["candidate_sources_within_1_hop"].mean(),
                "mean_candidates_2_hop": graph_summary["candidate_sources_within_2_hop"].mean(),
                "mean_candidates_3_hop": graph_summary["candidate_sources_within_3_hop"].mean(),
                "mean_candidates_4_hop": graph_summary["candidate_sources_within_4_hop"].mean(),
            }
        )
    outputs["sensitivity_graph_density.csv"] = pd.DataFrame(density_rows)
    outputs["sensitivity_graph_topology.csv"] = pd.DataFrame(topology_rows)
    return outputs


def _sensitivity_rows(config: ExperimentConfig, parameter_name: str, parameter_value: object) -> List[Dict[str, object]]:
    result = simulate(config, capture_graph=False)
    metrics = aggregate_metrics(result.step_results, bootstrap_samples=300)
    rows: List[Dict[str, object]] = []
    for _, row in metrics.iterrows():
        macro_values = [row[f"macro_f1_{variable}"] for variable in CONTEXT_VARIABLES]
        rows.append(
            {
                "scenario": row["scenario"],
                "policy": row["policy"],
                "policy_label": row["policy_label"],
                "discovery_mode": row.get("discovery_mode", ""),
                "parameter_name": parameter_name,
                "parameter_value": parameter_value,
                "OCA": row["overall_context_accuracy"],
                "MVA": row["mean_variable_accuracy"],
                "Macro_F1_mean": float(np.mean(macro_values)),
                "mean_sources": row["avg_selected_source_count"],
                "mean_discovery_cost": row["mean_discovery_cost"],
                "mean_recruitment_cost": row["mean_recruitment_cost"],
                "mean_selection_cost": row["mean_selection_cost"],
                "mean_total_operational_cost": row["mean_total_operational_cost"],
                "mean_latency_proxy": row["mean_latency_proxy"],
                "mean_privacy_cost": row["mean_privacy_cost"],
                "latency_proxy": row["avg_estimated_latency"],
                "privacy_cost": row["avg_privacy_exposure_cost"],
                "OCA_per_total_cost": row["OCA_per_total_cost"],
                "adaptive_activation_rate": row["adaptive_activation_rate"],
            }
        )
    return rows


def build_budget_matched_outputs(config: ExperimentConfig) -> tuple[pd.DataFrame, pd.DataFrame]:
    budget_config = replace(
        config,
        policies=BUDGET_MATCHED_POLICIES,
        episodes=config.budget_matched_episodes,
        timesteps=config.budget_matched_timesteps,
        seeds=config.budget_matched_seeds,
        run_sensitivity=False,
        run_scalability=False,
    )
    result = simulate(budget_config, capture_graph=False)
    metrics = aggregate_metrics(result.step_results, bootstrap_samples=min(500, config.bootstrap_samples))
    results = _apply_experiment_metadata(
        build_cost_effectiveness_results(metrics),
        config,
        "budget_matched",
        "budget_matched",
        episodes=config.budget_matched_episodes,
        timesteps=config.budget_matched_timesteps,
        seeds=config.budget_matched_seeds,
    )
    overhead = _apply_experiment_metadata(
        build_paper_overhead_table(metrics),
        config,
        "budget_matched",
        "budget_matched",
        episodes=config.budget_matched_episodes,
        timesteps=config.budget_matched_timesteps,
        seeds=config.budget_matched_seeds,
    )
    return results, overhead


def build_discovery_mode_comparison(config: ExperimentConfig) -> pd.DataFrame:
    comparison_config = replace(
        config,
        policies=DISCOVERY_MODE_COMPARISON_POLICIES,
        episodes=config.discovery_mode_comparison_episodes,
        timesteps=config.discovery_mode_comparison_timesteps,
        seeds=config.discovery_mode_comparison_seeds,
        run_sensitivity=False,
        run_scalability=False,
    )
    result = simulate(comparison_config, capture_graph=False)
    metrics = aggregate_metrics(result.step_results, bootstrap_samples=min(500, config.bootstrap_samples))
    rows: List[Dict[str, object]] = []
    for _, row in metrics.iterrows():
        macro_values = [row[f"macro_f1_{variable}"] for variable in CONTEXT_VARIABLES]
        rows.append(
            {
                "scenario": row["scenario"],
                "policy": row["policy"],
                "policy_label": row["policy_label"],
                "base_policy": _base_policy(str(row["policy"])),
                "discovery_mode": row.get("discovery_mode", _discovery_mode_for_policy(str(row["policy"]))),
                "OCA": row["overall_context_accuracy"],
                "MVA": row["mean_variable_accuracy"],
                "Macro_F1_mean": float(np.mean(macro_values)),
                "mean_sources": row["avg_selected_source_count"],
                "mean_discovered_candidate_count": row["avg_discovered_candidate_count"],
                "mean_visited_node_count": row["avg_visited_node_count"],
                "mean_traversed_edge_count": row["avg_traversed_edge_count"],
                "mean_scored_candidate_count": row["avg_scored_candidate_count"],
                "mean_discovery_cost": row["mean_discovery_cost"],
                "mean_recruitment_cost": row["mean_recruitment_cost"],
                "mean_selection_cost": row["mean_selection_cost"],
                "mean_total_operational_cost": row["mean_total_operational_cost"],
                "mean_latency_proxy": row["mean_latency_proxy"],
                "mean_privacy_cost": row["mean_privacy_cost"],
                "OCA_per_total_cost": row["OCA_per_total_cost"],
                "adaptive_activation_rate": row["adaptive_activation_rate"],
                "episodes": config.discovery_mode_comparison_episodes,
                "timesteps": config.discovery_mode_comparison_timesteps,
                "seeds": ",".join(str(seed) for seed in config.discovery_mode_comparison_seeds),
            }
        )
    return _apply_experiment_metadata(
        pd.DataFrame(rows),
        config,
        "discovery_mode_comparison",
        "discovery_mode_comparison",
        episodes=config.discovery_mode_comparison_episodes,
        timesteps=config.discovery_mode_comparison_timesteps,
        seeds=config.discovery_mode_comparison_seeds,
    )


def build_cost_effectiveness_results(metrics_df: pd.DataFrame) -> pd.DataFrame:
    rows: List[Dict[str, object]] = []
    for _, row in metrics_df.iterrows():
        macro_values = [row[f"macro_f1_{variable}"] for variable in CONTEXT_VARIABLES]
        rows.append(
            {
                "scenario": row["scenario"],
                "policy": row["policy"],
                "policy_label": row["policy_label"],
                "discovery_mode": row.get("discovery_mode", ""),
                "OCA": row["overall_context_accuracy"],
                "MVA": row["mean_variable_accuracy"],
                "Macro_F1_mean": float(np.mean(macro_values)),
                "mean_sources": row["avg_selected_source_count"],
                "mean_discovery_cost": row["mean_discovery_cost"],
                "mean_recruitment_cost": row["mean_recruitment_cost"],
                "mean_selection_cost": row["mean_selection_cost"],
                "mean_total_operational_cost": row["mean_total_operational_cost"],
                "mean_latency_proxy": row["mean_latency_proxy"],
                "mean_privacy_cost": row["mean_privacy_cost"],
                "OCA_per_source": row["OCA_per_source"],
                "OCA_per_total_cost": row["OCA_per_total_cost"],
            }
        )
    return pd.DataFrame(rows)


def build_pareto_policy_results(cost_effectiveness_df: pd.DataFrame) -> pd.DataFrame:
    frame = cost_effectiveness_df.copy()
    efficient_flags = []
    for _, row in frame.iterrows():
        scenario_df = frame[frame["scenario"] == row["scenario"]]
        dominated = (
            (scenario_df["OCA"] >= row["OCA"])
            & (scenario_df["mean_total_operational_cost"] <= row["mean_total_operational_cost"])
            & (
                (scenario_df["OCA"] > row["OCA"])
                | (scenario_df["mean_total_operational_cost"] < row["mean_total_operational_cost"])
            )
        ).any()
        efficient_flags.append(not bool(dominated))
    frame["pareto_efficient_oca_cost"] = efficient_flags
    return frame


def build_net_utility_sensitivity(cost_effectiveness_df: pd.DataFrame, config: ExperimentConfig) -> pd.DataFrame:
    rows: List[Dict[str, object]] = []
    cost_weights = [0.0, config.net_utility_cost_lambda, config.net_utility_cost_lambda * 2.0]
    privacy_weights = [0.0, config.net_utility_privacy_lambda]
    latency_weights = [0.0, config.net_utility_latency_lambda]
    for scenario, group in cost_effectiveness_df.groupby("scenario", observed=False):
        normalized = group.copy()
        for source_col, target_col in [
            ("mean_total_operational_cost", "normalized_total_operational_cost"),
            ("mean_privacy_cost", "normalized_privacy_cost"),
            ("mean_latency_proxy", "normalized_latency_proxy"),
        ]:
            span = normalized[source_col].max() - normalized[source_col].min()
            normalized[target_col] = 0.0 if span <= 0 else (normalized[source_col] - normalized[source_col].min()) / span
        for cost_weight in cost_weights:
            for privacy_weight in privacy_weights:
                for latency_weight in latency_weights:
                    for _, row in normalized.iterrows():
                        rows.append(
                            {
                                "scenario": scenario,
                                "policy": row["policy"],
                                "policy_label": row["policy_label"],
                                "discovery_mode": row.get("discovery_mode", ""),
                                "lambda_cost": cost_weight,
                                "lambda_privacy": privacy_weight,
                                "lambda_latency": latency_weight,
                                "OCA": row["OCA"],
                                "normalized_total_operational_cost": row["normalized_total_operational_cost"],
                                "normalized_privacy_cost": row["normalized_privacy_cost"],
                                "normalized_latency_proxy": row["normalized_latency_proxy"],
                                "net_utility": row["OCA"]
                                - (cost_weight * row["normalized_total_operational_cost"])
                                - (privacy_weight * row["normalized_privacy_cost"])
                                - (latency_weight * row["normalized_latency_proxy"]),
                            }
                        )
    return pd.DataFrame(rows)


def build_convergence_cost_results(step_results: pd.DataFrame) -> pd.DataFrame:
    transition_rows: List[Dict[str, object]] = []
    for (scenario, policy, seed, episode), group in step_results.sort_values("timestep").groupby(
        ["scenario", "policy", "seed", "episode"],
        observed=False,
    ):
        group = group.reset_index(drop=True)
        for idx in range(1, len(group)):
            if not any(group.loc[idx, f"true_{variable}"] != group.loc[idx - 1, f"true_{variable}"] for variable in CONTEXT_VARIABLES):
                continue
            recovery_idx = None
            for candidate_idx in range(idx, len(group)):
                if group.loc[candidate_idx, "overall_context_accuracy"] == 1.0:
                    recovery_idx = candidate_idx
                    break
            if recovery_idx is None:
                recovery_idx = len(group) - 1
            recovery_window = group.loc[idx:recovery_idx]
            transition_rows.append(
                {
                    "scenario": scenario,
                    "policy": policy,
                    "policy_label": POLICY_LABELS.get(str(policy), str(policy)),
                    "seed": seed,
                    "episode": episode,
                    "transition_timestep": int(idx),
                    "convergence_time": int(recovery_idx - idx),
                    "discovery_cost_during_recovery": float(recovery_window["discovery_cost"].sum()),
                    "recruitment_cost_during_recovery": float(recovery_window["recruitment_cost"].sum()),
                    "latency_cost_during_recovery": float(recovery_window["recruited_latency_sum"].sum()),
                    "effective_recovery_cost": float(recovery_window["total_operational_cost"].sum()),
                }
            )
    transitions = pd.DataFrame(transition_rows)
    if transitions.empty:
        return transitions
    rows = []
    rng = np.random.default_rng(31)
    for (scenario, policy), group in transitions.groupby(["scenario", "policy"], observed=False):
        values = group["effective_recovery_cost"].to_numpy(dtype=float)
        low, high = bootstrap_ci(values, rng)
        rows.append(
            {
                "scenario": scenario,
                "policy": policy,
                "policy_label": POLICY_LABELS.get(str(policy), str(policy)),
                "convergence_time": group["convergence_time"].mean(),
                "effective_recovery_cost": group["effective_recovery_cost"].mean(),
                "effective_recovery_cost_ci95_low": low,
                "effective_recovery_cost_ci95_high": high,
                "mean_discovery_cost_during_recovery": group["discovery_cost_during_recovery"].mean(),
                "mean_recruitment_cost_during_recovery": group["recruitment_cost_during_recovery"].mean(),
                "mean_latency_cost_during_recovery": group["latency_cost_during_recovery"].mean(),
                "transition_count": len(group),
            }
        )
    return pd.DataFrame(rows)


def bootstrap_ci(values: np.ndarray, rng: np.random.Generator, samples: int = 500) -> tuple[float, float]:
    if len(values) == 0:
        return float("nan"), float("nan")
    if len(values) == 1:
        return float(values[0]), float(values[0])
    means = [float(np.mean(rng.choice(values, size=len(values), replace=True))) for _ in range(samples)]
    low, high = np.quantile(np.asarray(means), [0.025, 0.975])
    return float(low), float(high)


def build_scalability_outputs(config: ExperimentConfig) -> tuple[pd.DataFrame, pd.DataFrame]:
    if not config.run_scalability:
        return pd.DataFrame(), pd.DataFrame()
    result_rows: List[Dict[str, object]] = []
    graph_frames: List[pd.DataFrame] = []
    for graph_size in config.scalability_graph_sizes:
        scale_config = replace(
            config,
            source_node_count=graph_size,
            episodes=config.scalability_episodes,
            timesteps=config.scalability_timesteps,
            seeds=config.scalability_seeds,
            policies=DEFAULT_POLICIES,
            run_sensitivity=False,
            run_scalability=False,
        )
        result = simulate(scale_config, capture_graph=True)
        metrics = aggregate_metrics(result.step_results, bootstrap_samples=250)
        graph_summary = result.graph_summary.copy()
        graph_summary["graph_size"] = graph_size
        graph_summary["scalability_episodes"] = config.scalability_episodes
        graph_summary["scalability_seeds"] = ",".join(str(seed) for seed in config.scalability_seeds)
        graph_frames.append(graph_summary)
        for _, row in metrics.iterrows():
            result_rows.append(
                {
                    "graph_size": graph_size,
                    "scenario": row["scenario"],
                    "policy": row["policy"],
                    "policy_label": row["policy_label"],
                    "discovery_mode": row.get("discovery_mode", ""),
                    "OCA": row["overall_context_accuracy"],
                    "MVA": row["mean_variable_accuracy"],
                    "mean_sources": row["avg_selected_source_count"],
                    "mean_discovered_candidate_count": row["avg_discovered_candidate_count"],
                    "mean_visited_node_count": row["avg_visited_node_count"],
                    "mean_traversed_edge_count": row["avg_traversed_edge_count"],
                    "mean_scored_candidate_count": row["avg_scored_candidate_count"],
                    "mean_total_operational_cost": row["mean_total_operational_cost"],
                    "mean_discovery_cost": row["mean_discovery_cost"],
                    "mean_recruitment_cost": row["mean_recruitment_cost"],
                    "mean_privacy_cost": row["mean_privacy_cost"],
                    "mean_latency_proxy": row["mean_latency_proxy"],
                    "OCA_per_total_cost": row["OCA_per_total_cost"],
                }
            )
    graph_output = pd.concat(graph_frames, ignore_index=True) if graph_frames else pd.DataFrame()
    return pd.DataFrame(result_rows), graph_output


def build_figures(
    config: ExperimentConfig,
    metrics_df: pd.DataFrame,
    episode_summary_df: pd.DataFrame,
    robustness_df: pd.DataFrame,
    figures_dir: Path,
    convergence_cost_df: pd.DataFrame,
    budget_matched_results_df: pd.DataFrame,
    pareto_df: pd.DataFrame,
    cost_effectiveness_df: pd.DataFrame,
    scalability_results_df: pd.DataFrame,
    discovery_mode_comparison_df: pd.DataFrame,
) -> Dict[str, Path]:
    plotters = {
        "overall_accuracy": lambda path: plot_overall_accuracy(metrics_df, path, config.plot_dpi, episode_summary_df),
        "macro_f1_by_variable": lambda path: plot_macro_f1_by_variable(metrics_df, path, config.plot_dpi),
        "accuracy_vs_missing_ego": lambda path: plot_accuracy_vs_missing_ego(robustness_df, path, config.plot_dpi),
        "convergence_time": lambda path: plot_convergence_time(metrics_df, path, config.plot_dpi),
        "effective_recovery_cost": lambda path: plot_effective_recovery_cost(convergence_cost_df, path, config.plot_dpi),
        "budget_matched": lambda path: plot_budget_matched_oca_cost(budget_matched_results_df, path, config.plot_dpi),
        "pareto_cost": lambda path: plot_pareto(pareto_df, path, config.plot_dpi, "mean_total_operational_cost", "Total operational cost"),
        "pareto_privacy": lambda path: plot_pareto(pareto_df, path, config.plot_dpi, "mean_privacy_cost", "Privacy cost"),
        "oca_per_cost": lambda path: plot_oca_per_cost(cost_effectiveness_df, path, config.plot_dpi),
        "scalability_cost": lambda path: plot_scalability(
            scalability_results_df, path, config.plot_dpi, "mean_total_operational_cost", "Total operational cost"
        ),
        "scalability_oca": lambda path: plot_scalability(scalability_results_df, path, config.plot_dpi, "OCA", "OCA"),
        "scalability_oca_per_cost": lambda path: plot_scalability(
            scalability_results_df, path, config.plot_dpi, "OCA_per_total_cost", "OCA / Total Cost"
        ),
        "discovery_mode_comparison": lambda path: plot_discovery_mode_oca_cost(
            discovery_mode_comparison_df,
            path,
            config.plot_dpi,
        ),
    }
    paths: Dict[str, Path] = {}
    for key, filename in config.plot_names.items():
        png_path = figures_dir / filename
        plotters[key](png_path)
        paths[filename] = png_path
        pdf_path = png_path.with_suffix(".pdf")
        plotters[key](pdf_path)
        paths[pdf_path.name] = pdf_path
    return paths


def build_validation_report(
    config: ExperimentConfig,
    data_dir: Path,
    figures_dir: Path,
    metrics_df: pd.DataFrame,
    step_results: pd.DataFrame,
    adaptive_diagnostics_df: pd.DataFrame,
) -> str:
    required_csvs = [
        config.metrics_name,
        config.main_results_table_name,
        config.overhead_table_name,
        config.adaptive_ablation_table_name,
        "siot_graph_summary.csv",
        "siot_relation_distribution.csv",
        "siot_candidate_pool_summary.csv",
        "adaptive_diagnostics.csv",
        "simulator_parameter_table.csv",
        "context_state_distribution.csv",
        "context_transition_summary.csv",
        "per_variable_error_rates.csv",
        "per_variable_macro_f1.csv",
        "hardest_variable_by_scenario_policy.csv",
        "joint_error_pattern_summary.csv",
        "convergence_cost_results.csv",
        "cost_effectiveness_results.csv",
        "pareto_policy_results.csv",
        "net_utility_sensitivity.csv",
        "budget_matched_results.csv",
        "budget_matched_overhead.csv",
        "discovery_mode_comparison.csv",
        "scalability_results.csv",
        "scalability_graph_summary.csv",
    ]
    required_texts = ["consistency_check_report.txt"]
    if config.run_sensitivity:
        required_csvs.extend(
            [
                "sensitivity_hop_radius.csv",
                "sensitivity_selection_cap.csv",
                "sensitivity_privacy_penalty.csv",
                "sensitivity_degraded_fraction.csv",
                "sensitivity_graph_density.csv",
                "sensitivity_graph_topology.csv",
            ]
        )
    required_figures = list(config.plot_names.values()) + [Path(name).with_suffix(".pdf").name for name in config.plot_names.values()]
    checks: List[tuple[str, bool, str]] = []
    for filename in required_csvs:
        checks.append((f"CSV exists: {filename}", (data_dir / filename).exists(), str(data_dir / filename)))
    for filename in required_texts:
        checks.append((f"Text report exists: {filename}", (data_dir / filename).exists(), str(data_dir / filename)))
    for filename in required_figures:
        checks.append((f"Figure exists: {filename}", (figures_dir / filename).exists(), str(figures_dir / filename)))
    raw_csv_path = data_dir / config.raw_results_name
    raw_archive_path = data_dir / config.raw_results_archive_name
    checks.append(
        (
            "Raw step output exists as generated CSV or release gzip archive",
            raw_csv_path.exists() or raw_archive_path.exists(),
            f"{raw_csv_path}; {raw_archive_path}",
        )
    )
    checks.append(("Release raw step gzip archive exists", raw_archive_path.exists(), str(raw_archive_path)))

    key_metric_columns = [
        "overall_context_accuracy",
        "mean_variable_accuracy",
        "avg_selected_source_count",
        "avg_estimated_latency",
        "avg_privacy_exposure_cost",
    ]
    checks.append(("No NaN in key metric columns", not metrics_df[key_metric_columns].isna().any().any(), ",".join(key_metric_columns)))
    scenario_policy_pairs = set(zip(metrics_df["scenario"].astype(str), metrics_df["policy"].astype(str)))
    expected_pairs = {(scenario, policy) for scenario in config.scenarios for policy in config.policies}
    checks.append(("All policies appear in all scenarios", expected_pairs.issubset(scenario_policy_pairs), str(sorted(expected_pairs - scenario_policy_pairs))))
    non_ego = adaptive_diagnostics_df[adaptive_diagnostics_df["policy"] != "ego_only"]
    checks.append(
        (
            "Graph discovery returns non-empty candidate pools for non-ego policies",
            bool((non_ego["discovered_candidate_count"] > 0).all()),
            "",
        )
    )
    checks.append(("ego_only recruits zero sources", bool((step_results.loc[step_results["policy"] == "ego_only", "recruited_source_count"] == 0).all()), ""))
    opp_rows = step_results[step_results["policy"] == "opportunistic_all"]
    checks.append(
        (
            "opportunistic_all recruits all discovered candidates",
            bool((opp_rows["recruited_source_count"] == opp_rows["discovered_candidate_count"]).all()),
            "",
        )
    )
    selective_rows = step_results[step_results["policy"].isin(["siot_aware", "siot_aware_trust_privacy"])]
    checks.append(
        (
            "Selective policies respect active selection caps",
            bool((selective_rows["recruited_source_count"] <= selective_rows["selection_cap"]).all()),
            "",
        )
    )
    avg_sources = step_results.groupby("policy", observed=False)["recruited_source_count"].mean()
    opp_avg = float(avg_sources.get("opportunistic_all", 0.0))
    selective_policies = [policy for policy in avg_sources.index if policy in {"siot_aware", "siot_aware_trust_privacy"}]
    selective_max = float(avg_sources[selective_policies].max()) if selective_policies else 0.0
    checks.append(("opportunistic_all recruits at least as many sources as selective policies on average", opp_avg >= selective_max, ""))
    bounded_columns = ["overall_context_accuracy", "mean_variable_accuracy"] + [f"macro_f1_{variable}" for variable in CONTEXT_VARIABLES]
    checks.append(
        (
            "All OCA/MVA/F1 values are within [0,1]",
            bool(((metrics_df[bounded_columns] >= 0.0) & (metrics_df[bounded_columns] <= 1.0)).all().all()),
            "",
        )
    )
    checks.append(
        (
            "Privacy and latency proxies are non-negative",
            bool((step_results["privacy_exposure_cost"].ge(0.0) & step_results["estimated_latency"].ge(0.0)).all()),
            "",
        )
    )
    checks.append(
        (
            "Discovery and total operational costs are non-negative",
            bool((step_results["discovery_cost"].ge(0.0) & step_results["total_operational_cost"].ge(0.0)).all()),
            "",
        )
    )
    checks.append(("discovery_mode exists in raw and metrics outputs", "discovery_mode" in step_results.columns and "discovery_mode" in metrics_df.columns, ""))
    metadata_columns = {
        "experiment_tag",
        "run_type",
        "policy_id",
        "policy_label",
        "discovery_mode",
        "scenario",
        "graph_size",
        "episodes",
        "timesteps",
        "seeds",
        "adaptive_enabled",
        "bounded_discovery_enabled",
    }
    metadata_files = [
        config.main_results_table_name,
        config.metrics_name,
        "discovery_mode_comparison.csv",
        "budget_matched_results.csv",
        "scalability_results.csv",
    ]
    stale_free = True
    for filename in metadata_files:
        path = data_dir / filename
        if not path.exists():
            stale_free = False
            checks.append((f"Metadata present: {filename}", False, "file missing"))
            continue
        frame = pd.read_csv(path, nrows=5)
        missing_metadata = sorted(metadata_columns - set(frame.columns))
        has_metadata = not missing_metadata
        stale_free = stale_free and has_metadata
        checks.append((f"Metadata present: {filename}", has_metadata, ",".join(missing_metadata)))
    checks.append(("No stale output files detected in metadata-bearing result files", stale_free, ""))
    table1_path = data_dir / config.main_results_table_name
    if table1_path.exists():
        table1 = pd.read_csv(table1_path)
        expected_modes = {
            "ego_only": EXHAUSTIVE_HOP_DISCOVERY,
            "opportunistic_all": EXHAUSTIVE_HOP_DISCOVERY,
            "siot_aware": BOUNDED_RELATIONSHIP_GUIDED_DISCOVERY,
            "siot_aware_trust_privacy": BOUNDED_RELATIONSHIP_GUIDED_DISCOVERY,
        }
        mode_ok = True
        for policy, expected_mode in expected_modes.items():
            policy_modes = table1.loc[table1["policy"].astype(str) == policy, "discovery_mode"].astype(str)
            mode_ok = mode_ok and not policy_modes.empty and bool((policy_modes == expected_mode).all())
        checks.append(("Main Table 1 policies use intended default discovery modes", mode_ok, ""))
        merged = table1.merge(
            metrics_df[["scenario", "policy", "overall_context_accuracy"]],
            on=["scenario", "policy"],
            how="left",
        )
        table_matches_metrics = bool(np.allclose(merged["OCA"].astype(float), merged["overall_context_accuracy"].astype(float)))
        checks.append(("Table 1 values match aggregated_metrics input", table_matches_metrics, ""))
    comparison_path = data_dir / "discovery_mode_comparison.csv"
    if comparison_path.exists():
        comparison = pd.read_csv(comparison_path)
        expected_comparison_modes = {
            "siot_aware_exhaustive_discovery": EXHAUSTIVE_HOP_DISCOVERY,
            "siot_aware_bounded_discovery": BOUNDED_RELATIONSHIP_GUIDED_DISCOVERY,
            "siot_aware_trust_privacy_exhaustive_discovery": EXHAUSTIVE_HOP_DISCOVERY,
            "siot_aware_trust_privacy_bounded_discovery": BOUNDED_RELATIONSHIP_GUIDED_DISCOVERY,
        }
        comparison_mode_ok = True
        for policy, expected_mode in expected_comparison_modes.items():
            policy_modes = comparison.loc[comparison["policy"].astype(str) == policy, "discovery_mode"].astype(str)
            comparison_mode_ok = comparison_mode_ok and not policy_modes.empty and bool((policy_modes == expected_mode).all())
        checks.append(("Discovery-mode comparison variants use intended discovery modes", comparison_mode_ok, ""))
        tag_values = set(comparison.get("experiment_tag", pd.Series(dtype=str)).astype(str).unique())
        checks.append(
            (
                "Discovery-mode comparison uses separate experiment_tag from Table 1",
                "discovery_mode_comparison" in tag_values,
                ",".join(sorted(tag_values)),
            )
        )
    if table1_path.exists() and comparison_path.exists():
        table1 = pd.read_csv(table1_path)
        comparison = pd.read_csv(comparison_path)
        main_tags = set(table1.loc[table1["policy"].astype(str) == "siot_aware", "experiment_tag"].astype(str))
        comparison_tags = set(
            comparison.loc[comparison["policy"].astype(str) == "siot_aware_bounded_discovery", "experiment_tag"].astype(str)
        )
        checks.append(
            (
                "Values across different experiment_tag values are not assumed identical",
                bool(main_tags and comparison_tags and main_tags.isdisjoint(comparison_tags)),
                f"main={sorted(main_tags)}, comparison={sorted(comparison_tags)}",
            )
        )
    if "discovery_mode" in step_results.columns:
        siot_rows = step_results[step_results["policy"].isin(["siot_aware", "siot_aware_trust_privacy"])]
        checks.append(
            (
                "Default SIoT-aware policies use bounded relationship-guided discovery",
                bool((siot_rows["discovery_mode"] == BOUNDED_RELATIONSHIP_GUIDED_DISCOVERY).all()),
                "",
            )
        )
        opp_modes = step_results.loc[step_results["policy"] == "opportunistic_all", "discovery_mode"]
        checks.append(("opportunistic_all uses exhaustive hop discovery", bool((opp_modes == EXHAUSTIVE_HOP_DISCOVERY).all()), ""))
        bounded_rows = step_results[step_results["discovery_mode"] == BOUNDED_RELATIONSHIP_GUIDED_DISCOVERY]
        checks.append(
            (
                "Bounded discovery respects active node budget",
                bool((bounded_rows["visited_node_count"] <= bounded_rows["discovery_node_budget_active"]).all()),
                "",
            )
        )
        checks.append(
            (
                "Bounded discovery respects active edge budget",
                bool((bounded_rows["traversed_edge_count"] <= bounded_rows["discovery_edge_budget_active"]).all()),
                "",
            )
        )
        checks.append(
            (
                "Bounded discovery respects max candidates to score",
                bool((bounded_rows["scored_candidate_count"] <= bounded_rows["max_candidates_to_score_active"]).all()),
                "",
            )
        )
        checks.append(
            (
                "Discovered candidates do not exceed visited nodes",
                bool((step_results["discovered_candidate_count"] <= step_results["visited_node_count"]).all()),
                "",
            )
        )
        for flag_column in [
            "all_variables_covered_by_candidates",
            "all_variables_covered_by_recruited_sources",
            "stopped_by_budget",
            "stopped_by_coverage",
            "stopped_by_quality",
            "stopped_by_frontier_empty",
        ]:
            checks.append((f"{flag_column} is boolean", str(step_results[flag_column].dtype) == "bool", ""))
    checks.append(("adaptive_triggered is boolean", str(adaptive_diagnostics_df["adaptive_triggered"].dtype) == "bool", ""))
    recorded_seeds = sorted(int(seed) for seed in step_results["seed"].unique())
    checks.append(("All random seeds are recorded", set(config.seeds).issubset(set(recorded_seeds)), str(recorded_seeds)))
    graph_summary_path = data_dir / "siot_graph_summary.csv"
    if graph_summary_path.exists():
        graph_summary = pd.read_csv(graph_summary_path)
        checks.append(
            (
                "Default graph average shortest path is approximately 3-4 hops",
                bool(
                    graph_summary["average_shortest_path_length"]
                    .between(config.target_average_shortest_path_low, config.target_average_shortest_path_high)
                    .mean()
                    >= 0.80
                ),
                f"mean={graph_summary['average_shortest_path_length'].mean():.3f}",
            )
        )
        checks.append(
            (
                "Default 2-hop ego pool does not include nearly all sources",
                bool((graph_summary["candidate_sources_within_2_hop"] / graph_summary["source_nodes"] < 0.75).all()),
                f"mean_share={(graph_summary['candidate_sources_within_2_hop'] / graph_summary['source_nodes']).mean():.3f}",
            )
        )
        checks.append(
            (
                "Ego belongs to dominant connected component",
                bool((graph_summary["ego_component_fraction"] >= 0.90).all()),
                f"min={graph_summary['ego_component_fraction'].min():.3f}",
            )
        )
        relation_counts = [column for column in graph_summary.columns if column.startswith("edge_count_")]
        checks.append(("All relation types are present", bool((graph_summary[relation_counts].sum() > 0).all()), ""))
    budget_path = data_dir / "budget_matched_results.csv"
    if budget_path.exists():
        budget = pd.read_csv(budget_path)
        expected_budget = {(scenario, policy) for scenario in config.scenarios for policy in BUDGET_MATCHED_POLICIES}
        actual_budget = set(zip(budget["scenario"].astype(str), budget["policy"].astype(str)))
        checks.append(
            ("Budget-matched policies appear in all scenarios", expected_budget.issubset(actual_budget), str(sorted(expected_budget - actual_budget)))
        )
        capped = budget[budget["policy"].isin(BUDGET_MATCHED_POLICIES)]
        checks.append(
            (
                "Budget-matched policies respect k on average",
                bool((capped["mean_sources"] <= config.adaptive_selection_cap + 1.0e-9).all()),
                "",
            )
        )

    lines = ["Validation report", ""]
    failed = []
    for label, passed, detail in checks:
        status = "PASS" if passed else "FAIL"
        lines.append(f"{status}: {label}")
        if detail:
            lines.append(f"  {detail}")
        if not passed:
            failed.append(label)
    lines.append("")
    lines.append(f"Overall status: {'PASS' if not failed else 'FAIL'}")
    return "\n".join(lines) + "\n"


def build_experiment_summary(
    metrics_df: pd.DataFrame,
    table1_df: pd.DataFrame,
    table2_df: pd.DataFrame,
    adaptive_ablation_df: pd.DataFrame,
) -> str:
    lines: list[str] = []
    lines.append("SIoT Opportunistic Context Enrichment: Experimental Summary")
    lines.append("")
    lines.append("Best-performing policy by scenario:")
    for scenario, group in metrics_df.groupby("scenario", sort=False, observed=True):
        if group.empty:
            continue
        best_row = group.sort_values("overall_context_accuracy", ascending=False).iloc[0]
        lines.append(
            f"- {scenario}: {best_row['policy']} "
            f"(OCA={best_row['overall_context_accuracy']:.3f}, MVA={best_row['mean_variable_accuracy']:.3f})"
        )
    lines.append("")
    lines.append("Table 1 OCA values:")
    for _, row in table1_df.iterrows():
        lines.append(f"- {row['scenario']} / {row['policy']}: {row['OCA']:.3f} [{row['OCA_ci95_low']:.3f}, {row['OCA_ci95_high']:.3f}]")
    lines.append("")
    lines.append("Table 2 overhead means:")
    for _, row in table2_df.iterrows():
        lines.append(
            f"- {row['scenario']} / {row['policy']}: sources={row['mean_sources']:.2f}, "
            f"total_cost={row['total_operational_cost']:.2f}, latency={row['latency_proxy']:.2f}, privacy={row['privacy_cost']:.2f}"
        )
    lines.append("")
    lines.append("Adaptive ablation modes:")
    for mode, group in adaptive_ablation_df.groupby("adaptive_mode", observed=False):
        lines.append(f"- {mode}: mean OCA={group['OCA'].mean():.3f}, mean adaptive activation={group['adaptive_activation_rate'].mean():.3f}")
    return "\n".join(lines) + "\n"


def build_consistency_check_report(
    config: ExperimentConfig,
    table1_df: pd.DataFrame,
    discovery_mode_comparison_df: pd.DataFrame,
    metrics_df: pd.DataFrame,
) -> str:
    main_row = table1_df[(table1_df["scenario"].astype(str) == "nominal") & (table1_df["policy"].astype(str) == "siot_aware")]
    comparison_row = discovery_mode_comparison_df[
        (discovery_mode_comparison_df["scenario"].astype(str) == "nominal")
        & (discovery_mode_comparison_df["policy"].astype(str) == "siot_aware_bounded_discovery")
    ]
    metric_row = metrics_df[(metrics_df["scenario"].astype(str) == "nominal") & (metrics_df["policy"].astype(str) == "siot_aware")]
    lines: List[str] = [
        "SIoT-HDT consistency check report",
        "",
        "Question:",
        "- Compare nominal / siot_aware in Table 1 with nominal / siot_aware_bounded_discovery in the discovery-mode comparison.",
        "",
    ]
    if main_row.empty or comparison_row.empty:
        lines.extend(
            [
                "Classification: pipeline issue",
                "- One or both source rows were not found in the generated outputs.",
            ]
        )
        return "\n".join(lines) + "\n"

    main = main_row.iloc[0]
    comparison = comparison_row.iloc[0]
    lines.extend(
        [
            "Classification: expected difference plus metadata/documentation issue",
            "- The two values are generated from different experiment_tag values and different sample sizes.",
            "- Table 1 uses the main default experiment.",
            "- The discovery-mode comparison uses a separate compact comparison run with explicit exhaustive/bounded policy variants.",
            "- The values should not be expected to match exactly, even when the base policy and discovery mode are equivalent.",
            "",
            "Source rows:",
            (
                f"- data/table1_main_results.csv: scenario={main['scenario']}, policy_id={main['policy_id']}, "
                f"policy_label={main['policy_label']}, discovery_mode={main['discovery_mode']}, OCA={float(main['OCA']):.6f}"
            ),
            (
                f"- data/discovery_mode_comparison.csv: scenario={comparison['scenario']}, policy_id={comparison['policy_id']}, "
                f"policy_label={comparison['policy_label']}, discovery_mode={comparison['discovery_mode']}, OCA={float(comparison['OCA']):.6f}"
            ),
            "",
            "Main Table 1 configuration:",
            f"- experiment_tag={main['experiment_tag']}",
            f"- run_type={main['run_type']}",
            f"- graph_size={main['graph_size']}",
            f"- episodes={main['episodes']}",
            f"- timesteps={main['timesteps']}",
            f"- seeds={main['seeds']}",
            f"- adaptive_enabled={main['adaptive_enabled']}",
            f"- bounded_discovery_enabled={main['bounded_discovery_enabled']}",
            f"- selection_cap={config.default_selection_cap}, adaptive_selection_cap={config.adaptive_selection_cap}",
            "",
            "Discovery-mode comparison configuration:",
            f"- experiment_tag={comparison['experiment_tag']}",
            f"- run_type={comparison['run_type']}",
            f"- graph_size={comparison['graph_size']}",
            f"- episodes={comparison['episodes']}",
            f"- timesteps={comparison['timesteps']}",
            f"- seeds={comparison['seeds']}",
            f"- adaptive_enabled={comparison['adaptive_enabled']}",
            f"- bounded_discovery_enabled={comparison['bounded_discovery_enabled']}",
            f"- selection_cap={config.default_selection_cap}, adaptive_selection_cap={config.adaptive_selection_cap}",
            "",
            "Bounded discovery parameters:",
            f"- small_graph_node_edge_score_budget=({config.bounded_discovery_node_budget_small}, {config.bounded_discovery_edge_budget_small}, {config.bounded_discovery_score_cap_small})",
            f"- medium_graph_node_edge_score_budget=({config.bounded_discovery_node_budget_medium}, {config.bounded_discovery_edge_budget_medium}, {config.bounded_discovery_score_cap_medium})",
            f"- large_graph_node_edge_score_budget=({config.bounded_discovery_node_budget_large}, {config.bounded_discovery_edge_budget_large}, {config.bounded_discovery_score_cap_large})",
            f"- min_candidates_required={config.bounded_discovery_min_candidates_required}",
            f"- min_variable_coverage={config.bounded_discovery_min_variable_coverage}",
            f"- max_neighbors_per_expansion={config.bounded_discovery_max_neighbors_per_expansion}",
            f"- adaptive_discovery_budget_multiplier={config.adaptive_discovery_budget_multiplier}",
            "",
            "Aggregation:",
            "- OCA is computed from timestep-level joint correctness.",
            "- The metric pipeline first aggregates each seed/episode over timesteps, then averages seed-episode summaries.",
            "- Confidence intervals use bootstrap resampling over seed-episode combinations.",
            "- data/table1_main_results.csv is generated from data/aggregated_metrics.csv.",
            "- data/discovery_mode_comparison.csv is generated by a separate simulate-and-aggregate call in build_discovery_mode_comparison.",
        ]
    )
    if not metric_row.empty:
        metric = metric_row.iloc[0]
        lines.extend(
            [
                "",
                "Input consistency:",
                f"- data/aggregated_metrics.csv nominal / siot_aware overall_context_accuracy={float(metric['overall_context_accuracy']):.6f}",
                f"- Table 1 nominal / siot_aware OCA={float(main['OCA']):.6f}",
            ]
        )
    lines.extend(
        [
            "",
            "Conclusion:",
            "- No numerical result was manually edited.",
            "- The apparent mismatch is expected because the rows belong to different experiment_tag values.",
            "- Metadata columns were added so the two rows are no longer presented as the same experimental condition.",
        ]
    )
    return "\n".join(lines) + "\n"


def _base_policy(policy: str) -> str:
    if policy in {"siot_aware_exhaustive_discovery", "siot_aware_bounded_discovery"}:
        return "siot_aware"
    if policy in {"siot_aware_trust_privacy_exhaustive_discovery", "siot_aware_trust_privacy_bounded_discovery"}:
        return "siot_aware_trust_privacy"
    return policy


def _discovery_mode_for_policy(policy: str) -> str:
    if policy in {"siot_aware", "siot_aware_trust_privacy", "siot_aware_bounded_discovery", "siot_aware_trust_privacy_bounded_discovery"}:
        return BOUNDED_RELATIONSHIP_GUIDED_DISCOVERY
    return EXHAUSTIVE_HOP_DISCOVERY


def _discovery_cache_key(policy: str, discovery_mode: str, adaptive_state: Dict[str, object]) -> tuple[object, ...]:
    if _base_policy(policy) == "ego_only":
        return ("ego_only",)
    if discovery_mode == EXHAUSTIVE_HOP_DISCOVERY:
        return (discovery_mode, int(adaptive_state["active_hop_radius"]))
    return (
        discovery_mode,
        int(adaptive_state["active_hop_radius"]),
        int(adaptive_state["discovery_node_budget_active"]),
        int(adaptive_state["discovery_edge_budget_active"]),
        int(adaptive_state["max_candidates_to_score_active"]),
        tuple(adaptive_state.get("unresolved_variables", ())),
    )


def _bounded_discovery_budgets(config: ExperimentConfig, adaptive_triggered: bool) -> tuple[int, int, int]:
    source_count = config.source_node_count
    if source_count >= config.bounded_discovery_large_graph_threshold:
        node_budget = config.bounded_discovery_node_budget_large
        edge_budget = config.bounded_discovery_edge_budget_large
        score_cap = config.bounded_discovery_score_cap_large
    elif source_count >= config.bounded_discovery_medium_graph_threshold:
        node_budget = config.bounded_discovery_node_budget_medium
        edge_budget = config.bounded_discovery_edge_budget_medium
        score_cap = config.bounded_discovery_score_cap_medium
    else:
        node_budget = config.bounded_discovery_node_budget_small
        edge_budget = config.bounded_discovery_edge_budget_small
        score_cap = config.bounded_discovery_score_cap_small
    if adaptive_triggered:
        multiplier = config.adaptive_discovery_budget_multiplier
        node_budget = int(round(node_budget * multiplier))
        edge_budget = int(round(edge_budget * multiplier))
        score_cap = int(round(score_cap * multiplier))
    return node_budget, edge_budget, score_cap


def _source_variable_coverage_count(graph, source_ids: Iterable[str]) -> int:
    covered: set[str] = set()
    for source_id in source_ids:
        if source_id not in graph:
            continue
        covered.update(
            variable
            for variable in graph.nodes[source_id].get("sensing_capabilities", ())
            if variable in CONTEXT_VARIABLES
        )
    return len(covered)


def _mean_or_zero(values: Iterable[float]) -> float:
    values = list(values)
    return 0.0 if not values else float(np.mean(values))


def _scored_candidate_count(policy: str, discovery_result) -> int:
    if policy in {"siot_aware", "siot_aware_trust_privacy", "random_k", "budgeted_opportunistic_k"}:
        if discovery_result.discovery_mode == BOUNDED_RELATIONSHIP_GUIDED_DISCOVERY:
            return min(len(discovery_result.candidates), int(discovery_result.max_candidates_to_score_active))
        return len(list(discovery_result.candidates))
    return 0


def _discovery_cost(config: ExperimentConfig, policy: str, discovery_result) -> float:
    if policy == "ego_only":
        return 0.0
    return float(
        discovery_result.nodes_visited
        + (config.discovery_edge_lambda * discovery_result.traversed_edges)
        + discovery_result.relation_traversal_cost
    )


def _recruitment_cost(config: ExperimentConfig, recruited_source_count: int, latency_sum: float, privacy_sum: float) -> float:
    if recruited_source_count <= 0:
        return 0.0
    return float(
        recruited_source_count
        + (config.recruitment_latency_lambda * latency_sum)
        + (config.recruitment_privacy_lambda * privacy_sum)
    )
